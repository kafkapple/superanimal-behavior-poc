"""Utility functions for SuperAnimal behavior analysis."""
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    return logging.getLogger("superanimal")


def get_device(device: str = "auto") -> str:
    """
    Determine the best available device for inference.

    Supports:
        - 'auto': Automatically detect best available device
        - 'cuda': NVIDIA GPU (Linux/Windows)
        - 'mps': Apple Silicon GPU (macOS M1/M2/M3)
        - 'cpu': CPU fallback

    Args:
        device: Device preference ('auto', 'cuda', 'mps', 'cpu')

    Returns:
        Selected device string
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not available, using CPU")
        return "cpu"

    # Validate explicitly specified device
    if device != "auto":
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to auto-detect")
        elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available, falling back to auto-detect")
        else:
            logger.info(f"Using specified device: {device}")
            return device

    # Auto-detect best available device
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Auto-detected CUDA device: {device_name}")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Auto-detected Apple MPS (Metal Performance Shaders)")
        return "mps"

    logger.info("No GPU available, using CPU")
    return "cpu"
