#!/usr/bin/env python3
"""
Download benchmark datasets for SuperAnimal behavior analysis.

Supports:
- Sample videos (mouse, dog, horse)
- AP-10K (multi-species animal pose)
- COCO Keypoints (human pose)
- MARS (multi-animal social behavior)
- UCLA Mouse Dataset

Usage:
    python -m src.scripts.download_datasets                    # Download all samples
    python -m src.scripts.download_datasets --dataset ap10k    # Download specific dataset
    python -m src.scripts.download_datasets --list             # List available datasets
"""
import argparse
import sys
import os
import urllib.request
import zipfile
import tarfile
import json
from pathlib import Path
from typing import Optional, Dict, List
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.downloader import VideoDownloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# Benchmark dataset configurations
BENCHMARK_DATASETS = {
    "ap10k": {
        "name": "AP-10K (Animal Pose)",
        "description": "Multi-species animal pose dataset with 17 keypoints",
        "url": "https://github.com/AlexTheBad/AP-10K",
        "download_urls": [
            # Note: Actual download requires manual registration
            # These are placeholder/sample URLs
        ],
        "format": "coco",
        "keypoints": 17,
        "species": ["dog", "cat", "horse", "cow", "sheep", "elephant"],
        "license": "Apache 2.0",
        "size": "~2GB",
        "manual_download": True,
        "instructions": """
AP-10K requires manual download:
1. Visit https://github.com/AlexTheBad/AP-10K
2. Follow download instructions
3. Extract to data/external/ap10k/
""",
    },
    "coco_pose": {
        "name": "COCO Keypoints",
        "description": "Human pose estimation benchmark with 17 keypoints",
        "url": "https://cocodataset.org/#download",
        "download_urls": [],
        "format": "coco",
        "keypoints": 17,
        "species": ["human"],
        "license": "CC BY 4.0",
        "size": "~20GB (val2017)",
        "manual_download": True,
        "instructions": """
COCO requires manual download:
1. Visit https://cocodataset.org/#download
2. Download val2017.zip and annotations_trainval2017.zip
3. Extract to data/external/coco/
""",
    },
    "mars": {
        "name": "MARS (Mouse Social Behavior)",
        "description": "Multi-animal resident-intruder social interaction dataset",
        "url": "https://data.caltech.edu/records/s0vdx-0k302",
        "download_urls": [],
        "format": "custom",
        "keypoints": 7,
        "num_animals": 2,
        "classes": ["attack", "mount", "investigation", "other"],
        "species": ["mouse"],
        "license": "CC BY 4.0",
        "size": "~5GB",
        "manual_download": True,
        "instructions": """
MARS dataset requires manual download:
1. Visit https://data.caltech.edu/records/s0vdx-0k302
2. Download the dataset files
3. Extract to data/external/mars/
4. Run preprocessing: python -m src.data.preprocessing.mars_preprocess
""",
    },
    "ucla_mouse": {
        "name": "UCLA Mouse Dataset",
        "description": "Mouse behavior recordings for action recognition",
        "url": "https://github.com/",
        "download_urls": [],
        "format": "video",
        "species": ["mouse"],
        "license": "Research only",
        "size": "~1GB",
        "manual_download": True,
        "instructions": """
UCLA Mouse dataset:
1. Contact dataset authors for access
2. Download and extract to data/external/ucla/
""",
    },
}


class DatasetDownloader:
    """Download and prepare benchmark datasets."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.external_dir = self.data_dir / "external"
        self.external_dir.mkdir(parents=True, exist_ok=True)
        self.video_downloader = VideoDownloader(str(self.data_dir))

    def list_datasets(self) -> None:
        """List all available datasets."""
        print("\n" + "=" * 70)
        print("Available Datasets")
        print("=" * 70)

        # Sample videos (auto-downloadable)
        print("\n[SAMPLE VIDEOS] (Auto-download available)")
        print("-" * 50)
        for name, info in self.video_downloader.SAMPLE_VIDEOS.items():
            print(f"  {name}:")
            print(f"    Species: {info.get('species', 'unknown')}")
            print(f"    Description: {info.get('description', '')}")

        # Benchmark datasets
        print("\n[BENCHMARK DATASETS]")
        print("-" * 50)
        for key, info in BENCHMARK_DATASETS.items():
            manual = "(Manual download)" if info.get("manual_download") else "(Auto-download)"
            print(f"  {key}: {info['name']} {manual}")
            print(f"    Species: {', '.join(info.get('species', []))}")
            print(f"    Keypoints: {info.get('keypoints', 'N/A')}")
            print(f"    Size: {info.get('size', 'Unknown')}")
            print(f"    License: {info.get('license', 'Unknown')}")
            print()

    def download_samples(self) -> Dict[str, Optional[Path]]:
        """Download all sample videos."""
        logger.info("Downloading sample videos...")
        return self.video_downloader.download_all_samples()

    def download_dataset(self, dataset_name: str) -> bool:
        """Download a specific dataset."""
        if dataset_name not in BENCHMARK_DATASETS:
            logger.error(f"Unknown dataset: {dataset_name}")
            logger.info(f"Available: {list(BENCHMARK_DATASETS.keys())}")
            return False

        info = BENCHMARK_DATASETS[dataset_name]
        logger.info(f"\nDataset: {info['name']}")
        logger.info(f"Description: {info['description']}")

        if info.get("manual_download"):
            logger.warning("This dataset requires manual download.")
            print(info.get("instructions", ""))
            return False

        # Auto-download if URLs available
        download_urls = info.get("download_urls", [])
        if not download_urls:
            logger.warning("No download URLs available.")
            return False

        output_dir = self.external_dir / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for url in download_urls:
            filename = url.split("/")[-1]
            filepath = output_dir / filename

            if filepath.exists():
                logger.info(f"Already exists: {filepath}")
                continue

            logger.info(f"Downloading: {url}")
            try:
                urllib.request.urlretrieve(url, filepath)
                logger.info(f"Downloaded: {filepath}")

                # Extract if archive
                if filepath.suffix in [".zip", ".gz", ".tar"]:
                    self._extract_archive(filepath, output_dir)
            except Exception as e:
                logger.error(f"Failed to download: {e}")
                return False

        return True

    def _extract_archive(self, filepath: Path, output_dir: Path) -> None:
        """Extract archive file."""
        logger.info(f"Extracting: {filepath}")

        if filepath.suffix == ".zip":
            with zipfile.ZipFile(filepath, "r") as zf:
                zf.extractall(output_dir)
        elif filepath.suffix == ".gz" and ".tar" in filepath.suffixes:
            with tarfile.open(filepath, "r:gz") as tf:
                tf.extractall(output_dir)
        elif filepath.suffix == ".tar":
            with tarfile.open(filepath, "r") as tf:
                tf.extractall(output_dir)

        logger.info(f"Extracted to: {output_dir}")

    def prepare_dataset(self, dataset_name: str) -> bool:
        """Prepare dataset for pipeline use (preprocessing)."""
        output_dir = self.external_dir / dataset_name

        if not output_dir.exists():
            logger.error(f"Dataset not found: {output_dir}")
            logger.info("Please download the dataset first.")
            return False

        # Dataset-specific preprocessing
        if dataset_name == "mars":
            return self._prepare_mars(output_dir)
        elif dataset_name == "ap10k":
            return self._prepare_ap10k(output_dir)
        elif dataset_name == "coco_pose":
            return self._prepare_coco(output_dir)

        logger.info(f"No preprocessing required for: {dataset_name}")
        return True

    def _prepare_mars(self, data_dir: Path) -> bool:
        """Prepare MARS dataset."""
        logger.info("Preparing MARS dataset...")
        # Preprocessing would be done by mars_preprocess.py
        logger.info("Run: python -m src.data.preprocessing.mars_preprocess")
        return True

    def _prepare_ap10k(self, data_dir: Path) -> bool:
        """Prepare AP-10K dataset."""
        logger.info("Preparing AP-10K dataset...")
        # Convert to unified format if needed
        return True

    def _prepare_coco(self, data_dir: Path) -> bool:
        """Prepare COCO dataset."""
        logger.info("Preparing COCO dataset...")
        return True

    def get_dataset_path(self, dataset_name: str) -> Optional[Path]:
        """Get path to downloaded dataset."""
        if dataset_name in self.video_downloader.SAMPLE_VIDEOS:
            return self.video_downloader.download_sample(dataset_name)

        dataset_dir = self.external_dir / dataset_name
        if dataset_dir.exists():
            return dataset_dir

        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for SuperAnimal behavior analysis"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Specific dataset to download (e.g., ap10k, mars, coco_pose)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets",
    )
    parser.add_argument(
        "--samples",
        action="store_true",
        help="Download sample videos only",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all auto-downloadable datasets",
    )
    parser.add_argument(
        "--prepare",
        type=str,
        help="Prepare/preprocess a downloaded dataset",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory (default: data)",
    )

    args = parser.parse_args()

    downloader = DatasetDownloader(args.data_dir)

    if args.list:
        downloader.list_datasets()
        return

    if args.samples or (not args.dataset and not args.all and not args.prepare):
        logger.info("=" * 60)
        logger.info("SuperAnimal PoC - Dataset Downloader")
        logger.info("=" * 60)

        results = downloader.download_samples()

        logger.info("\nDownload Summary:")
        for name, path in results.items():
            status = "OK" if path else "FAILED"
            logger.info(f"  {name}: {status}")

        return

    if args.dataset:
        downloader.download_dataset(args.dataset)
        return

    if args.prepare:
        downloader.prepare_dataset(args.prepare)
        return

    if args.all:
        logger.info("Downloading all available datasets...")
        downloader.download_samples()

        for name in BENCHMARK_DATASETS:
            if not BENCHMARK_DATASETS[name].get("manual_download"):
                downloader.download_dataset(name)


if __name__ == "__main__":
    main()
