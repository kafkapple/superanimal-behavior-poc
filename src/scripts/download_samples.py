#!/usr/bin/env python3
"""
Download sample videos for SuperAnimal behavior analysis.
Run this script to pre-download all sample videos.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.downloader import VideoDownloader
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Download all sample videos."""
    logger.info("SuperAnimal PoC - Sample Video Downloader")
    logger.info("=" * 50)

    data_dir = project_root / "data"
    downloader = VideoDownloader(str(data_dir))

    logger.info("\nDownloading sample videos...")
    results = downloader.download_all_samples()

    logger.info("\n" + "=" * 50)
    logger.info("Download Summary:")
    logger.info("=" * 50)

    for name, path in results.items():
        if path and path.exists():
            info = downloader.get_video_info(path)
            logger.info(f"\n{name}:")
            logger.info(f"  Path: {path}")
            logger.info(f"  Resolution: {info['width']}x{info['height']}")
            logger.info(f"  FPS: {info['fps']:.1f}")
            logger.info(f"  Frames: {info['frame_count']}")
            logger.info(f"  Duration: {info['duration']:.1f}s")
        else:
            logger.error(f"\n{name}: FAILED")

    logger.info("\n" + "=" * 50)
    logger.info("Done!")


if __name__ == "__main__":
    main()
