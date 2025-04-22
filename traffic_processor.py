#!/usr/bin/env python3

"""
NeuraShield Network Traffic Processor
This module processes network traffic and extracts features for the NeuraShield threat detection system.
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Tuple
from threading import Thread, Event
from queue import Queue
import subprocess
from datetime import datetime
import tempfile

# Try importing common Python packet capture libraries
try:
    import pyshark
    PYSHARK_AVAILABLE = True
except ImportError:
    PYSHARK_AVAILABLE = False

try:
    from scapy.all import sniff, IP, TCP, UDP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs/traffic.log"))
    ]
)

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"), exist_ok=True)

class FlowKey:
    """Class representing a network flow key (5-tuple)"""
    
    def __init__(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int, protocol: int):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
    
    def __eq__(self, other):
        if not isinstance(other, FlowKey):
            return False
        return (self.src_ip == other.src_ip and
                self.dst_ip == other.dst_ip and
                self.src_port == other.src_port and
                self.dst_port == other.dst_port and
                self.protocol == other.protocol)
    
    def __hash__(self):
        return hash((self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol))
    
    def __str__(self):
        proto_name = {6: 'TCP', 17: 'UDP'}.get(self.protocol, str(self.protocol))
        return f"{self.src_ip}:{self.src_port} -> {self.dst_ip}:{self.dst_port} ({proto_name})"

class FlowStats:
    """Class for tracking flow statistics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.packet_count = 0
        self.byte_count = 0
        self.src_bytes = 0
        self.dst_bytes = 0
        self.timestamps = []
        self.packet_sizes = []
        self.ttl_values = []
        self.tcp_window_sizes = []
        self.tcp_flags = []
    
    def update(self, packet_size: int, timestamp: float, ttl: Optional[int] = None, 
              window_size: Optional[int] = None, tcp_flags: Optional[int] = None, 
              is_src_to_dst: bool = True):
        """Update flow statistics with a new packet"""
        self.packet_count += 1
        self.byte_count += packet_size
        self.last_time = timestamp
        
        if is_src_to_dst:
            self.src_bytes += packet_size
        else:
            self.dst_bytes += packet_size
        
        self.timestamps.append(timestamp)
        self.packet_sizes.append(packet_size)
        
        if ttl is not None:
            self.ttl_values.append(ttl)
        
        if window_size is not None:
            self.tcp_window_sizes.append(window_size)
        
        if tcp_flags is not None:
            self.tcp_flags.append(tcp_flags)
    
    def get_duration(self) -> float:
        """Get flow duration in seconds"""
        return self.last_time - self.start_time if self.packet_count > 1 else 0
    
    def get_packet_rate(self) -> float:
        """Get packet rate (packets per second)"""
        duration = self.get_duration()
        return self.packet_count / duration if duration > 0 else 0
    
    def get_byte_rate(self) -> float:
        """Get byte rate (bytes per second)"""
        duration = self.get_duration()
        return self.byte_count / duration if duration > 0 else 0
    
    def get_avg_packet_size(self) -> float:
        """Get average packet size"""
        return self.byte_count / self.packet_count if self.packet_count > 0 else 0
    
    def get_avg_ttl(self) -> float:
        """Get average TTL value"""
        return np.mean(self.ttl_values) if self.ttl_values else 0
    
    def get_avg_window_size(self) -> float:
        """Get average TCP window size"""
        return np.mean(self.tcp_window_sizes) if self.tcp_window_sizes else 0
    
    def get_inter_arrival_times(self) -> List[float]:
        """Get packet inter-arrival times"""
        if len(self.timestamps) <= 1:
            return [0]
        return [self.timestamps[i] - self.timestamps[i-1] for i in range(1, len(self.timestamps))]
    
    def get_avg_inter_arrival_time(self) -> float:
        """Get average packet inter-arrival time"""
        iat = self.get_inter_arrival_times()
        return np.mean(iat) if len(iat) > 0 else 0

class NetworkTrafficProcessor:
    """Main class for processing network traffic and extracting features"""
    
    def __init__(self, interface: str = None, pcap_file: str = None, 
                flow_timeout: int = 120, batch_size: int = 100,
                feature_output_dir: str = None):
        """
        Initialize the traffic processor.
        
        Args:
            interface: Network interface to capture from
            pcap_file: PCAP file to read from
            flow_timeout: Flow timeout in seconds
            batch_size: Batch size for feature extraction
            feature_output_dir: Directory to output feature files
        """
        self.interface = interface
        self.pcap_file = pcap_file
        self.flow_timeout = flow_timeout
        self.batch_size = batch_size
        
        if feature_output_dir:
            self.feature_output_dir = feature_output_dir
        else:
            self.feature_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features")
        
        os.makedirs(self.feature_output_dir, exist_ok=True)
        
        # Initialize flow tracking
        self.flow_stats = {}
        self.complete_flows = []
        self.flow_queue = Queue()
        
        # Thread control
        self.stop_event = Event()
        self.capture_thread = None
        self.process_thread = None
    
    def _check_capture_capabilities(self) -> bool:
        """Check if packet capture libraries are available"""
        if not (PYSHARK_AVAILABLE or SCAPY_AVAILABLE):
            logging.error("No packet capture libraries available. Install either PyShark or Scapy.")
            return False
        
        if self.interface and not os.path.exists(f"/sys/class/net/{self.interface}"):
            logging.error(f"Network interface {self.interface} not found")
            return False
        
        if self.pcap_file and not os.path.exists(self.pcap_file):
            logging.error(f"PCAP file {self.pcap_file} not found")
            return False
        
        return True
    
    def _process_packet_scapy(self, packet) -> Optional[Tuple[FlowKey, int, float, Optional[int], Optional[int], Optional[int], bool]]:
        """
        Process a packet captured with Scapy.
        
        Returns a tuple of (flow_key, packet_size, timestamp, ttl, window_size, tcp_flags, is_src_to_dst)
        """
        if IP not in packet:
            return None
        
        # Extract IP information
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        protocol = packet[IP].proto
        ttl = packet[IP].ttl
        
        # Extract port information and determine if this is TCP or UDP
        if TCP in packet:
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
            window_size = packet[TCP].window
            tcp_flags = packet[TCP].flags
        elif UDP in packet:
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
            window_size = None
            tcp_flags = None
        else:
            # Not TCP or UDP, skip
            return None
        
        # Create flow key (we use source->dest ordering)
        flow_key = FlowKey(src_ip, dst_ip, src_port, dst_port, protocol)
        
        # Determine packet size and direction
        packet_size = len(packet)
        is_src_to_dst = True  # With our flow key definition, this is always true
        
        # Get timestamp
        timestamp = float(packet.time)
        
        return (flow_key, packet_size, timestamp, ttl, window_size, tcp_flags, is_src_to_dst)
    
    def _process_packet_pyshark(self, packet) -> Optional[Tuple[FlowKey, int, float, Optional[int], Optional[int], Optional[int], bool]]:
        """
        Process a packet captured with PyShark.
        
        Returns a tuple of (flow_key, packet_size, timestamp, ttl, window_size, tcp_flags, is_src_to_dst)
        """
        try:
            if not hasattr(packet, 'ip'):
                return None
            
            # Extract IP information
            src_ip = packet.ip.src
            dst_ip = packet.ip.dst
            protocol = int(packet.ip.proto)
            ttl = int(packet.ip.ttl) if hasattr(packet.ip, 'ttl') else None
            
            # Extract port information and determine if this is TCP or UDP
            if hasattr(packet, 'tcp'):
                src_port = int(packet.tcp.srcport)
                dst_port = int(packet.tcp.dstport)
                window_size = int(packet.tcp.window_size) if hasattr(packet.tcp, 'window_size') else None
                tcp_flags = int(packet.tcp.flags, 16) if hasattr(packet.tcp, 'flags') else None
            elif hasattr(packet, 'udp'):
                src_port = int(packet.udp.srcport)
                dst_port = int(packet.udp.dstport)
                window_size = None
                tcp_flags = None
            else:
                # Not TCP or UDP, skip
                return None
            
            # Create flow key (we use source->dest ordering)
            flow_key = FlowKey(src_ip, dst_ip, src_port, dst_port, protocol)
            
            # Determine packet size and direction
            packet_size = int(packet.length) if hasattr(packet, 'length') else 0
            is_src_to_dst = True  # With our flow key definition, this is always true
            
            # Get timestamp
            timestamp = float(packet.sniff_timestamp) if hasattr(packet, 'sniff_timestamp') else time.time()
            
            return (flow_key, packet_size, timestamp, ttl, window_size, tcp_flags, is_src_to_dst)
        
        except Exception as e:
            logging.error(f"Error processing packet with PyShark: {str(e)}")
            return None
    
    def _process_packet(self, packet, library: str = 'scapy'):
        """
        Process a packet and update flow statistics.
        
        Args:
            packet: Packet object (from Scapy or PyShark)
            library: Library used to capture the packet ('scapy' or 'pyshark')
        """
        try:
            # Process packet based on the library used
            if library == 'scapy':
                result = self._process_packet_scapy(packet)
            else:  # pyshark
                result = self._process_packet_pyshark(packet)
            
            if result is None:
                return
            
            flow_key, packet_size, timestamp, ttl, window_size, tcp_flags, is_src_to_dst = result
            
            # Update flow statistics
            if flow_key in self.flow_stats:
                self.flow_stats[flow_key].update(
                    packet_size, timestamp, ttl, window_size, tcp_flags, is_src_to_dst
                )
            else:
                # Create new flow
                self.flow_stats[flow_key] = FlowStats()
                self.flow_stats[flow_key].update(
                    packet_size, timestamp, ttl, window_size, tcp_flags, is_src_to_dst
                )
            
            # Check for flow timeout
            current_time = time.time()
            expired_flows = []
            
            for fk, fs in self.flow_stats.items():
                if current_time - fs.last_time > self.flow_timeout:
                    expired_flows.append(fk)
            
            # Process expired flows
            for fk in expired_flows:
                fs = self.flow_stats.pop(fk)
                self._extract_flow_features(fk, fs)
        
        except Exception as e:
            logging.error(f"Error processing packet: {str(e)}")
    
    def _extract_flow_features(self, flow_key: FlowKey, flow_stats: FlowStats) -> Dict[str, float]:
        """
        Extract features from a flow for the NeuraShield model.
        
        Args:
            flow_key: Flow key
            flow_stats: Flow statistics
            
        Returns:
            Dictionary of features
        """
        # Extract basic flow features
        duration = flow_stats.get_duration()
        packet_rate = flow_stats.get_packet_rate()
        byte_rate = flow_stats.get_byte_rate()
        avg_packet_size = flow_stats.get_avg_packet_size()
        avg_ttl = flow_stats.get_avg_ttl()
        avg_window_size = flow_stats.get_avg_window_size()
        avg_iat = flow_stats.get_avg_inter_arrival_time()
        
        # Ratio of source to destination bytes
        if flow_stats.dst_bytes > 0:
            src_dst_ratio = flow_stats.src_bytes / flow_stats.dst_bytes
        else:
            src_dst_ratio = float(flow_stats.src_bytes) if flow_stats.src_bytes > 0 else 0
        
        # Create feature vector in the model's expected format
        features = {
            "feature_0": duration,           # Flow duration
            "feature_1": flow_stats.byte_count,     # Total bytes
            "feature_2": flow_stats.packet_count,   # Total packets
            "feature_3": byte_rate,          # Byte rate
            "feature_4": avg_ttl,            # Average TTL
            "feature_5": avg_window_size,    # Average window size
            "feature_6": avg_packet_size,    # Average packet size
            "feature_7": avg_iat             # Average inter-arrival time
        }
        
        # Add raw flow data for reference
        flow_data = {
            "src_ip": flow_key.src_ip,
            "dst_ip": flow_key.dst_ip,
            "src_port": flow_key.src_port,
            "dst_port": flow_key.dst_port,
            "protocol": flow_key.protocol,
            "protocol_name": {6: 'TCP', 17: 'UDP'}.get(flow_key.protocol, str(flow_key.protocol)),
            "timestamp": datetime.now().isoformat(),
            "raw_features": {
                "duration": duration,
                "packet_count": flow_stats.packet_count,
                "byte_count": flow_stats.byte_count,
                "packet_rate": packet_rate,
                "byte_rate": byte_rate,
                "avg_packet_size": avg_packet_size,
                "src_bytes": flow_stats.src_bytes,
                "dst_bytes": flow_stats.dst_bytes,
                "src_dst_ratio": src_dst_ratio,
                "avg_ttl": avg_ttl,
                "avg_window_size": avg_window_size,
                "avg_inter_arrival_time": avg_iat
            }
        }
        
        # Save complete flow data with features
        complete_flow = {
            "flow_key": str(flow_key),
            "features": features,
            "flow_data": flow_data
        }
        
        self.complete_flows.append(complete_flow)
        self.flow_queue.put(complete_flow)
        
        # If we've reached batch size, save to file
        if len(self.complete_flows) >= self.batch_size:
            self._save_features_batch()
        
        return features
    
    def _save_features_batch(self):
        """Save a batch of features to a file"""
        if not self.complete_flows:
            return
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.feature_output_dir, f"flow_features_{timestamp}.json")
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(self.complete_flows, f)
        
        logging.info(f"Saved {len(self.complete_flows)} flows to {filename}")
        
        # Clear complete flows
        self.complete_flows = []
    
    def _capture_packets_scapy(self):
        """Capture packets using Scapy"""
        try:
            if self.pcap_file:
                # Read from PCAP file
                logging.info(f"Reading packets from {self.pcap_file}")
                sniff(offline=self.pcap_file, prn=lambda x: self._process_packet(x, 'scapy'), store=0)
            else:
                # Capture from interface
                logging.info(f"Capturing packets from interface {self.interface}")
                sniff(iface=self.interface, prn=lambda x: self._process_packet(x, 'scapy'), 
                     store=0, stop_filter=lambda x: self.stop_event.is_set())
        except Exception as e:
            logging.error(f"Error capturing packets with Scapy: {str(e)}")
    
    def _capture_packets_pyshark(self):
        """Capture packets using PyShark"""
        try:
            # Import pyshark here to handle import errors gracefully
            import pyshark
            
            if self.pcap_file:
                # Read from PCAP file
                logging.info(f"Reading packets from {self.pcap_file}")
                cap = pyshark.FileCapture(self.pcap_file, keep_packets=False)
                for packet in cap:
                    if self.stop_event.is_set():
                        break
                    self._process_packet(packet, 'pyshark')
                cap.close()
            else:
                # Capture from interface
                logging.info(f"Capturing packets from interface {self.interface}")
                cap = pyshark.LiveCapture(interface=self.interface)
                cap.sniff(timeout=0)  # Start capturing
                
                for packet in cap.sniff_continuously():
                    if self.stop_event.is_set():
                        break
                    self._process_packet(packet, 'pyshark')
                cap.close()
        except Exception as e:
            logging.error(f"Error capturing packets with PyShark: {str(e)}")
    
    def _process_flows(self):
        """Process flows from the queue and extract features"""
        while not self.stop_event.is_set() or not self.flow_queue.empty():
            try:
                # Get flow from queue with timeout
                try:
                    flow = self.flow_queue.get(timeout=1.0)
                except:
                    continue
                
                # Process flow
                logging.debug(f"Processing flow: {flow['flow_key']}")
                
                # Mark as done
                self.flow_queue.task_done()
            except Exception as e:
                logging.error(f"Error processing flow: {str(e)}")
    
    def start(self):
        """Start the traffic processor"""
        if not self._check_capture_capabilities():
            return False
        
        # Reset state
        self.flow_stats = {}
        self.complete_flows = []
        self.stop_event.clear()
        
        # Start packet capture thread
        if SCAPY_AVAILABLE:
            self.capture_thread = Thread(target=self._capture_packets_scapy, daemon=True)
        elif PYSHARK_AVAILABLE:
            self.capture_thread = Thread(target=self._capture_packets_pyshark, daemon=True)
        else:
            logging.error("No packet capture library available")
            return False
        
        self.capture_thread.start()
        
        # Start flow processing thread
        self.process_thread = Thread(target=self._process_flows, daemon=True)
        self.process_thread.start()
        
        logging.info("Traffic processor started")
        return True
    
    def stop(self):
        """Stop the traffic processor"""
        self.stop_event.set()
        
        if self.capture_thread:
            self.capture_thread.join(timeout=5.0)
        
        if self.process_thread:
            self.process_thread.join(timeout=5.0)
        
        # Save any remaining flows
        self._save_features_batch()
        
        logging.info("Traffic processor stopped")
    
    def extract_features_from_pcap(self, pcap_file: str) -> List[Dict[str, Any]]:
        """
        Extract features from a PCAP file.
        
        Args:
            pcap_file: Path to PCAP file
            
        Returns:
            List of feature dictionaries
        """
        if not os.path.exists(pcap_file):
            logging.error(f"PCAP file {pcap_file} not found")
            return []
        
        # Set PCAP file
        self.pcap_file = pcap_file
        
        # Reset state
        self.flow_stats = {}
        self.complete_flows = []
        self.stop_event.clear()
        
        # Start capture thread (will process the PCAP and then exit)
        if SCAPY_AVAILABLE:
            self._capture_packets_scapy()
        elif PYSHARK_AVAILABLE:
            self._capture_packets_pyshark()
        else:
            logging.error("No packet capture library available")
            return []
        
        # Save any remaining flows
        self._save_features_batch()
        
        logging.info(f"Extracted features from {pcap_file}")
        return self.complete_flows

def main():
    """Run the traffic processor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuraShield Network Traffic Processor')
    parser.add_argument('--interface', '-i', type=str, help='Network interface to capture from')
    parser.add_argument('--pcap', '-p', type=str, help='PCAP file to read from')
    parser.add_argument('--timeout', '-t', type=int, default=120, help='Flow timeout in seconds')
    parser.add_argument('--batch-size', '-b', type=int, default=100, help='Batch size for feature extraction')
    parser.add_argument('--output-dir', '-o', type=str, help='Directory to output feature files')
    
    args = parser.parse_args()
    
    if not args.interface and not args.pcap:
        parser.error('Either --interface or --pcap must be specified')
    
    # Create traffic processor
    processor = NetworkTrafficProcessor(
        interface=args.interface,
        pcap_file=args.pcap,
        flow_timeout=args.timeout,
        batch_size=args.batch_size,
        feature_output_dir=args.output_dir
    )
    
    try:
        # Start processor
        if processor.start():
            logging.info("Press Ctrl+C to stop")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping traffic processor")
    finally:
        processor.stop()

if __name__ == "__main__":
    main() 