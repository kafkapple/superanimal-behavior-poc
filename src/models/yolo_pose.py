"""
YOLO Pose Model Wrapper.

Provides YOLO-based pose estimation for comparison with SuperAnimal.
Supports YOLOv8-pose for animals and humans.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


# YOLO Pose keypoint definitions
YOLO_ANIMAL_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Mapping from YOLO keypoints to SuperAnimal-compatible names
YOLO_TO_SUPERANIMAL_MAPPING = {
    # For quadrupeds (approximate mapping)
    "nose": "nose",
    "left_eye": "left_eye",
    "right_eye": "right_eye",
    "left_ear": "left_earbase",
    "right_ear": "right_earbase",
    "left_shoulder": "left_front_elbow",
    "right_shoulder": "right_front_elbow",
    "left_hip": "left_back_elbow",
    "right_hip": "right_back_elbow",
    "left_ankle": "left_front_paw",
    "right_ankle": "right_front_paw",
}


class YOLOPosePredictor:
    """
    YOLO Pose model wrapper for animal pose estimation.

    Uses ultralytics YOLOv8-pose or custom animal pose models.
    """

    def __init__(
        self,
        model_name: str = "yolov8n-pose",
        device: str = "auto",
        conf_threshold: float = 0.5,
    ):
        """
        Initialize YOLO Pose predictor.

        Args:
            model_name: YOLO model name (yolov8n-pose, yolov8s-pose, etc.)
            device: Device to use (auto, cuda, mps, cpu)
            conf_threshold: Confidence threshold for detections
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.model = None
        self.device = self._get_device(device)
        self._load_model()

    def _get_device(self, device: str) -> str:
        """Determine device to use."""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return device

    def _load_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO model: {self.model_name}")
            self.model = YOLO(f"{self.model_name}.pt")
            self.model.to(self.device)
            logger.info(f"YOLO model loaded on {self.device}")

        except ImportError:
            logger.warning("ultralytics not installed. Install with: pip install ultralytics")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

    def predict_video(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Path] = None,
        max_frames: Optional[int] = None,
    ) -> Dict:
        """
        Run pose estimation on video.

        Args:
            video_path: Path to video file
            output_dir: Output directory for results
            max_frames: Maximum frames to process

        Returns:
            Dictionary with keypoints and metadata
        """
        if self.model is None:
            logger.error("YOLO model not loaded")
            return {"keypoints": None, "metadata": {}}

        import cv2

        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        logger.info(f"Processing {total_frames} frames with YOLO Pose...")

        all_keypoints = []

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            results = self.model(frame, verbose=False, conf=self.conf_threshold)

            # Extract keypoints
            if len(results) > 0 and results[0].keypoints is not None:
                kps = results[0].keypoints.data.cpu().numpy()
                if len(kps) > 0:
                    # Take first detection
                    keypoints = kps[0]  # (17, 3) - x, y, conf
                    all_keypoints.append(keypoints)
                else:
                    all_keypoints.append(np.zeros((17, 3)))
            else:
                all_keypoints.append(np.zeros((17, 3)))

        cap.release()

        keypoints = np.array(all_keypoints)

        return {
            "keypoints": keypoints,
            "keypoint_names": YOLO_ANIMAL_KEYPOINTS,
            "metadata": {
                "fps": fps,
                "total_frames": len(keypoints),
                "width": width,
                "height": height,
                "model": self.model_name,
            }
        }

    def get_keypoint_names(self) -> List[str]:
        """Get keypoint names."""
        return YOLO_ANIMAL_KEYPOINTS


class MMPosePredictor:
    """
    MMPose model wrapper for animal pose estimation.

    Uses MMPose animal pose models (AP-10K trained).
    """

    def __init__(
        self,
        config: str = "td-hm_hrnet-w32_8xb64-210e_ap10k-256x256",
        checkpoint: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize MMPose predictor.

        Args:
            config: MMPose config name
            checkpoint: Path to checkpoint (auto-download if None)
            device: Device to use
        """
        self.config = config
        self.checkpoint = checkpoint
        self.model = None
        self.device = device
        self._load_model()

    def _load_model(self):
        """Load MMPose model."""
        try:
            from mmpose.apis import init_model, inference_topdown
            from mmdet.apis import init_detector, inference_detector

            # This would require proper MMPose setup
            logger.info(f"MMPose model loading: {self.config}")
            # Model initialization would go here

        except ImportError:
            logger.warning("mmpose not installed. Install with: pip install mmpose mmdet")
            self.model = None

    def predict_video(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Path] = None,
        max_frames: Optional[int] = None,
    ) -> Dict:
        """Run pose estimation on video."""
        logger.warning("MMPose video prediction not fully implemented")
        return {"keypoints": None, "metadata": {"model": "mmpose"}}


# ============================================================
# Model Registry
# ============================================================

POSE_MODELS = {
    "yolo_pose": YOLOPosePredictor,
    "mmpose": MMPosePredictor,
}


def get_pose_model(name: str, **kwargs):
    """Get pose model by name."""
    if name not in POSE_MODELS:
        raise ValueError(f"Unknown pose model: {name}. Available: {list(POSE_MODELS.keys())}")
    return POSE_MODELS[name](**kwargs)


def check_available_models() -> Dict[str, bool]:
    """Check which pose models are available."""
    available = {}

    # Check YOLO
    try:
        from ultralytics import YOLO
        available["yolo_pose"] = True
    except ImportError:
        available["yolo_pose"] = False

    # Check MMPose
    try:
        import mmpose
        available["mmpose"] = True
    except ImportError:
        available["mmpose"] = False

    # SuperAnimal is always available (main model)
    available["superanimal"] = True

    return available
