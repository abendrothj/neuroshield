#!/usr/bin/env python3

"""
NeuraShield Dataset Downloader
This script downloads and preprocesses cybersecurity datasets for transfer learning
"""

import os
import sys
import argparse
import logging
import requests
import zipfile
import tarfile
import gzip
import shutil
import pandas as pd
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Dataset URLs and information
DATASETS = {
    'unsw-nb15': {
        'name': 'UNSW-NB15',
        'url': 'https://research.unsw.edu.au/projects/unsw-nb15-dataset/data/UNSW-NB15_1.csv',
        'files': [
            'https://research.unsw.edu.au/projects/unsw-nb15-dataset/data/UNSW-NB15_1.csv',
            'https://research.unsw.edu.au/projects/unsw-nb15-dataset/data/UNSW-NB15_2.csv',
            'https://research.unsw.edu.au/projects/unsw-nb15-dataset/data/UNSW-NB15_3.csv',
            'https://research.unsw.edu.au/projects/unsw-nb15-dataset/data/UNSW-NB15_4.csv',
            'https://research.unsw.edu.au/projects/unsw-nb15-dataset/data/UNSW-NB15_training-set.csv',
            'https://research.unsw.edu.au/projects/unsw-nb15-dataset/data/UNSW-NB15_testing-set.csv'
        ],
        'description': 'Modern network traffic with normal and attack behaviors'
    },
    'cicids2017': {
        'name': 'CIC-IDS-2017',
        'url': 'https://www.unb.ca/cic/datasets/ids-2017.html',
        'description': 'Network traffic with modern attacks like DoS, DDoS, Brute Force, XSS, SQL Injection'
    },
    'nsl-kdd': {
        'name': 'NSL-KDD',
        'url': 'https://www.unb.ca/cic/datasets/nsl.html',
        'files': [
            'https://github.com/defcom17/NSL_KDD/raw/master/KDDTrain%2B.txt',
            'https://github.com/defcom17/NSL_KDD/raw/master/KDDTest%2B.txt'
        ],
        'description': 'Improved version of the KDD Cup 1999 dataset'
    },
    'cse-cic-ids2018': {
        'name': 'CSE-CIC-IDS2018',
        'url': 'https://www.unb.ca/cic/datasets/ids-2018.html',
        'description': 'Network traffic with various attack scenarios including cryptomining'
    },
    'iot-23': {
        'name': 'IoT-23',
        'url': 'https://www.stratosphereips.org/datasets-iot23',
        'description': 'IoT traffic including malicious and benign samples from IoT devices'
    },
    'ctu-13': {
        'name': 'CTU-13',
        'url': 'https://www.stratosphereips.org/datasets-ctu13',
        'description': 'Botnet traffic captures in 13 different scenarios'
    }
}

def download_file(url, destination):
    """
    Download a file from URL to destination with progress bar
    
    Args:
        url: URL to download from
        destination: Local path to save the file
    """
    logging.info(f"Downloading {url} to {destination}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    with open(destination, 'wb') as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))


def extract_archive(archive_path, extract_dir):
    """
    Extract an archive file
    
    Args:
        archive_path: Path to the archive file
        extract_dir: Directory to extract to
    """
    logging.info(f"Extracting {archive_path} to {extract_dir}")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_dir)
    elif archive_path.endswith('.gz') and not archive_path.endswith('.tar.gz'):
        output_path = archive_path[:-3]  # Remove .gz extension
        with gzip.open(archive_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        logging.warning(f"Unsupported archive format: {archive_path}")


def download_unsw_nb15(data_dir):
    """Download UNSW-NB15 dataset"""
    dataset_dir = os.path.join(data_dir, 'UNSW_NB15')
    os.makedirs(dataset_dir, exist_ok=True)
    
    for url in DATASETS['unsw-nb15']['files']:
        filename = os.path.basename(url)
        destination = os.path.join(dataset_dir, filename)
        
        if not os.path.exists(destination):
            try:
                download_file(url, destination)
            except Exception as e:
                logging.error(f"Error downloading {url}: {str(e)}")
                continue
    
    # Check if we have the required training and testing files
    required_files = ['UNSW-NB15_training-set.csv', 'UNSW-NB15_testing-set.csv']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(dataset_dir, f))]
    
    if missing_files:
        logging.warning(f"Missing required files: {missing_files}")
        logging.warning("You may need to download these files manually from the UNSW-NB15 website")
    else:
        logging.info("UNSW-NB15 dataset downloaded successfully")
    
    return dataset_dir


def download_nslkdd(data_dir):
    """Download NSL-KDD dataset"""
    dataset_dir = os.path.join(data_dir, 'NSL_KDD')
    os.makedirs(dataset_dir, exist_ok=True)
    
    for url in DATASETS['nsl-kdd']['files']:
        filename = os.path.basename(url).replace('%2B', '+')
        destination = os.path.join(dataset_dir, filename)
        
        if not os.path.exists(destination):
            try:
                download_file(url, destination)
            except Exception as e:
                logging.error(f"Error downloading {url}: {str(e)}")
                continue
    
    # Check if we have the required training and testing files
    required_files = ['KDDTrain+.txt', 'KDDTest+.txt']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(dataset_dir, f))]
    
    if missing_files:
        logging.warning(f"Missing required files: {missing_files}")
        logging.warning("You may need to download these files manually from the NSL-KDD website")
    else:
        logging.info("NSL-KDD dataset downloaded successfully")
    
    return dataset_dir


def download_cicids2017(data_dir):
    """
    Note: CIC-IDS2017 requires manual download from their website
    This function just creates the directory and provides instructions
    """
    dataset_dir = os.path.join(data_dir, 'CIC_IDS_2017')
    os.makedirs(dataset_dir, exist_ok=True)
    
    instructions_file = os.path.join(dataset_dir, 'DOWNLOAD_INSTRUCTIONS.txt')
    with open(instructions_file, 'w') as f:
        f.write("CIC-IDS2017 Dataset Download Instructions\n")
        f.write("=======================================\n\n")
        f.write("The CIC-IDS2017 dataset requires registration and manual download from:\n")
        f.write("https://www.unb.ca/cic/datasets/ids-2017.html\n\n")
        f.write("After downloading, place the CSV files in this directory.\n")
        f.write("The files should have names like 'Monday-WorkingHours.pcap_ISCX.csv', etc.\n")
    
    logging.info(f"Created directory for CIC-IDS2017 at {dataset_dir}")
    logging.info("CIC-IDS2017 requires manual download. See DOWNLOAD_INSTRUCTIONS.txt in the dataset directory.")
    
    return dataset_dir


def list_available_datasets():
    """List all available datasets with descriptions"""
    print("\nAvailable Datasets:")
    print("===================")
    
    for key, info in DATASETS.items():
        print(f"\n{info['name']} ({key})")
        print("-" * (len(info['name']) + len(key) + 3))
        print(f"Description: {info['description']}")
        print(f"Source URL: {info['url']}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Download cybersecurity datasets for transfer learning')
    
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS.keys()),
                        help='Datasets to download')
    parser.add_argument('--all', action='store_true',
                        help='Download all available datasets')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory to store datasets')
    parser.add_argument('--list', action='store_true',
                        help='List available datasets')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_datasets()
        sys.exit(0)
    
    if not args.datasets and not args.all:
        parser.error("No datasets specified. Use --datasets or --all")
    
    return args


def main():
    """Main function"""
    args = parse_arguments()
    
    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Determine which datasets to download
    if args.all:
        datasets_to_download = list(DATASETS.keys())
    else:
        datasets_to_download = args.datasets
    
    # Download selected datasets
    downloaded_dirs = {}
    
    for dataset in datasets_to_download:
        try:
            if dataset == 'unsw-nb15':
                downloaded_dirs[dataset] = download_unsw_nb15(args.data_dir)
            elif dataset == 'nsl-kdd':
                downloaded_dirs[dataset] = download_nslkdd(args.data_dir)
            elif dataset == 'cicids2017':
                downloaded_dirs[dataset] = download_cicids2017(args.data_dir)
            else:
                logging.warning(f"Download method not implemented for {dataset}")
                logging.warning(f"Please visit {DATASETS[dataset]['url']} to download manually")
        except Exception as e:
            logging.error(f"Error downloading {dataset}: {str(e)}")
    
    # Print summary
    print("\nDownload Summary:")
    print("================")
    
    for dataset, directory in downloaded_dirs.items():
        print(f"{DATASETS[dataset]['name']}: {directory}")
    
    print("\nNotes:")
    print("- Some datasets may require manual downloads due to website registration requirements.")
    print("- Check the dataset directories for any DOWNLOAD_INSTRUCTIONS.txt files.")
    print("- Use these dataset paths with multi_dataset_learning.py for transfer learning.")


if __name__ == "__main__":
    main() 