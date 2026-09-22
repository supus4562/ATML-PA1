"""utilities/download_cifar.py — Fast CIFAR-10 & CIFAR-100 Downloader.

Bypasses the heavily throttled University of Toronto server (~50 KB/s)
by downloading verified official tarballs from high-speed CDN mirrors
and extracting them into the standard directory structure expected by torchvision.

Usage:
  python utilities/download_cifar.py --dest ./data/cifar
"""
from __future__ import annotations

import argparse
import hashlib
import os
import tarfile
import urllib.request

DATASETS = {
    'cifar-10': {
        'url': 'https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz',
        'filename': 'cifar-10-python.tar.gz',
        'extracted_dir': 'cifar-10-batches-py',
        'md5': 'c58f30108f718f92721af3b95e74349a'
    },
    'cifar-100': {
        'url': 'https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-python.tar.gz',
        'filename': 'cifar-100-python.tar.gz',
        'extracted_dir': 'cifar-100-python',
        'md5': 'eb9058c3a382ffc7106e4002c42a8d85'
    }
}


try:
    from tqdm import tqdm

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

except ImportError:
    class DownloadProgressBar:
        """Fallback progress reporter using standard library when tqdm is not installed."""
        def __init__(self, unit='B', unit_scale=True, miniters=1, desc=""):
            self.desc = desc
            self.last_printed = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            print()

        def update_to(self, b=1, bsize=1, tsize=None):
            downloaded = b * bsize
            if tsize:
                percent = downloaded / tsize * 100
                mb = downloaded / (1024 * 1024)
                total_mb = tsize / (1024 * 1024)
                # Print every ~10% or at 100%
                if int(percent) >= self.last_printed + 10 or downloaded >= tsize:
                    self.last_printed = int(percent)
                    print(f"\r{self.desc}: {mb:.1f}/{total_mb:.1f} MB ({percent:.1f}%)", end="", flush=True)


def verify_md5(file_path: str, expected_md5: str) -> bool:
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == expected_md5


def download_and_extract(name: str, info: dict, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    target_extracted = os.path.join(dest_dir, info['extracted_dir'])
    
    if os.path.exists(target_extracted):
        print(f"[{name}] Already extracted at {target_extracted}. Skipping.")
        return

    tarball_path = os.path.join(dest_dir, info['filename'])
    if not (os.path.exists(tarball_path) and verify_md5(tarball_path, info['md5'])):
        print(f"[{name}] Downloading from fast mirror: {info['url']}...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=info['filename']) as t:
            urllib.request.urlretrieve(info['url'], filename=tarball_path, reporthook=t.update_to)
            
        print(f"[{name}] Verifying MD5 checksum...")
        if not verify_md5(tarball_path, info['md5']):
            raise RuntimeError(f"MD5 mismatch for {tarball_path}")
        print(f"[{name}] MD5 verified successfully!")
    else:
        print(f"[{name}] Valid tarball already found at {tarball_path}")

    print(f"[{name}] Extracting to {dest_dir}...")
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)
    print(f"[{name}] Extraction complete -> {target_extracted}\n")


def main():
    parser = argparse.ArgumentParser(description="Fast CIFAR-10 & CIFAR-100 Downloader")
    parser.add_argument('--dest', default='./data/cifar', help="Destination directory (default: ./data/cifar)")
    args = parser.parse_args()

    print(f"=== Fast CIFAR Dataset Downloader ===")
    print(f"Target Directory: {os.path.abspath(args.dest)}\n")

    for name, info in DATASETS.items():
        download_and_extract(name, info, args.dest)

    print("All CIFAR datasets are downloaded, verified, and ready for Task 4!")


if __name__ == '__main__':
    main()
