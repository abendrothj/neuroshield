/**
 * Represents a security event stored on the blockchain
 */
export interface SecurityEvent {
  id: string;
  timestamp: string;
  type: string;
  details: string;
  ipfsHash?: string;
}

/**
 * Event types categorization
 */
export enum EventType {
  INFO = "INFO",
  WARNING = "WARNING",
  CRITICAL = "CRITICAL",
  ERROR = "ERROR"
}

/**
 * Represents an AI detection that can be submitted to the blockchain
 */
export interface AIDetection {
  timestamp: string;
  threat_type: string;
  confidence: number;
  affected_system: string;
  source_ip: string;
  details?: Record<string, any>;
} 