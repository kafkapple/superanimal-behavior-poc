"""
SuperAnimal keypoint prediction module.
Wraps DeepLabCut's SuperAnimal models for pose estimation.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check if DeepLabCut is available
DLC_AVAILABLE = False
try:
    import deeplabcut
    DLC_AVAILABLE = hasattr(deeplabcut, 'video_inference_superanimal')
except Exception:
    pass


class SuperAnimalPredictor:
    """
    Wrapper for DeepLabCut SuperAnimal models.
    Supports both TopViewMouse and Quadruped models.
    """

    SUPPORTED_MODELS = {
        "topviewmouse": {
            "superanimal_name": "superanimal_topviewmouse",
            "model_name": "hrnet_w32",
            "detector_name": "fasterrcnn_resnet50_fpn_v2",
            "num_keypoints": 27,
        },
        "quadruped": {
            "superanimal_name": "superanimal_quadruped",
            "model_name": "hrnet_w32",
            "detector_name": "fasterrcnn_resnet50_fpn_v2",
            "num_keypoints": 39,
        },
    }

    def __init__(
        self,
        model_type: str = "topviewmouse",
        model_name: str = "hrnet_w32",
        video_adapt: bool = False,
        scale_list: Optional[List[int]] = None,
        device: str = "auto",
        use_keypoints: Optional[List[str]] = None,
    ):
        """
        Initialize SuperAnimal predictor.

        Args:
            model_type: Either 'topviewmouse' or 'quadruped'
            model_name: Model architecture ('hrnet_w32' recommended)
            video_adapt: Enable video adaptation for better accuracy
            scale_list: List of scales for multi-scale inference
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
            use_keypoints: Optional list of keypoint names to use (filters output)
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Supported: {list(self.SUPPORTED_MODELS.keys())}")

        self.model_type = model_type
        self.model_config = self.SUPPORTED_MODELS[model_type]
        self.superanimal_name = self.model_config["superanimal_name"]
        self.model_name = model_name
        self.video_adapt = video_adapt
        self.scale_list = scale_list
        self.device = self._setup_device(device)
        self.use_keypoints = use_keypoints  # Filter to specific keypoints

        logger.info(f"Initialized SuperAnimalPredictor: {self.superanimal_name} ({self.model_name})")
        if use_keypoints:
            logger.info(f"Using filtered keypoints: {len(use_keypoints)} of {self.model_config['num_keypoints']}")
        if not DLC_AVAILABLE:
            logger.warning("DeepLabCut not fully available. Will use mock predictions for demo.")

    def _setup_device(self, device: str) -> str:
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
        import torch

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

    def predict_video(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
    ) -> Dict:
        """
        Run pose estimation on a video.

        Args:
            video_path: Path to input video
            output_dir: Directory for output files (optional)
            max_frames: Maximum frames to process (for PoC)

        Returns:
            Dictionary with keypoint predictions and metadata
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if output_dir is None:
            output_dir = video_path.parent
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running pose estimation on: {video_path}")
        logger.info(f"Model: {self.superanimal_name}, Device: {self.device}")

        # Check if DLC is available
        if not DLC_AVAILABLE:
            logger.warning("DeepLabCut not available. Generating mock predictions for demo.")
            return self._generate_mock_predictions(video_path, max_frames)

        import deeplabcut
        import cv2

        # If max_frames specified, create a trimmed video first
        # DLC doesn't support max_frames directly
        actual_video_path = video_path
        temp_video_path = None

        if max_frames is not None:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if max_frames < total_frames:
                logger.info(f"Trimming video to {max_frames} frames (original: {total_frames})")
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                temp_video_path = output_dir / f"{video_path.stem}_trimmed_{max_frames}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))

                for i in range(max_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)

                out.release()
                actual_video_path = temp_video_path
                logger.info(f"Created trimmed video: {temp_video_path}")

            cap.release()

        # Prepare inference arguments for DLC 3.0 API
        detector_name = self.model_config.get("detector_name", "fasterrcnn_resnet50_fpn_v2")
        inference_kwargs = {
            "videos": [str(actual_video_path)],
            "superanimal_name": self.superanimal_name,
            "model_name": self.model_name,
            "detector_name": detector_name,
            "video_adapt": self.video_adapt,
            "dest_folder": str(output_dir),
            "device": self.device,  # Pass device for GPU acceleration
        }

        if self.scale_list is not None:
            inference_kwargs["scale_list"] = self.scale_list

        logger.info(f"DLC inference config: superanimal={self.superanimal_name}, model={self.model_name}, detector={detector_name}, device={self.device}")

        # Run DeepLabCut inference
        try:
            deeplabcut.video_inference_superanimal(**inference_kwargs)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            logger.warning("Falling back to mock predictions for demo.")
            if temp_video_path and temp_video_path.exists():
                temp_video_path.unlink()
            return self._generate_mock_predictions(video_path, max_frames)

        # Load results (use actual_video_path for finding output files)
        results = self._load_predictions(actual_video_path, output_dir)
        results["video_path"] = str(video_path)  # Return original video path
        results["model_type"] = self.model_type
        results["model_name"] = self.model_name

        # Clean up temp video if created
        if temp_video_path and temp_video_path.exists():
            # Keep trimmed video for reference, or delete if not needed
            # temp_video_path.unlink()
            pass

        return results

    def predict_images(
        self,
        image_paths: Union[str, Path, List[Union[str, Path]]],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """
        Run pose estimation on images by creating a temporary video.

        DLC SuperAnimal only supports video inference, so images are
        converted to a single-frame video for processing.

        Args:
            image_paths: Single image path or list of image paths
            output_dir: Directory for output files

        Returns:
            Dictionary with keypoint predictions and metadata
        """
        import cv2

        # Normalize to list
        if isinstance(image_paths, (str, Path)):
            image_paths = [image_paths]
        image_paths = [Path(p) for p in image_paths]

        # Validate
        for img_path in image_paths:
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")

        if output_dir is None:
            output_dir = image_paths[0].parent
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running pose estimation on {len(image_paths)} images")

        # Read first image for dimensions
        first_img = cv2.imread(str(image_paths[0]))
        if first_img is None:
            raise ValueError(f"Could not read image: {image_paths[0]}")
        height, width = first_img.shape[:2]

        # Create temporary video from images (1 fps so each frame = 1 image)
        temp_video_path = output_dir / f"_temp_images_{len(image_paths)}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_video_path), fourcc, 1.0, (width, height))

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is not None:
                # Resize if needed
                if img.shape[:2] != (height, width):
                    img = cv2.resize(img, (width, height))
                out.write(img)
        out.release()

        # Run video inference
        try:
            results = self.predict_video(
                video_path=temp_video_path,
                output_dir=output_dir,
                max_frames=None,
            )
            results["image_paths"] = [str(p) for p in image_paths]
            results["source_type"] = "images"
        finally:
            # Clean up temp video
            if temp_video_path.exists():
                temp_video_path.unlink()

        return results

    def _generate_mock_predictions(
        self,
        video_path: Path,
        max_frames: Optional[int] = None,
    ) -> Dict:
        """Generate realistic mock keypoint predictions for demo purposes."""
        import cv2

        # Get video properties
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        num_frames = min(total_frames, max_frames or 500)
        num_keypoints = self.model_config["num_keypoints"]

        logger.info(f"Generating mock predictions for {num_frames} frames, {num_keypoints} keypoints")

        # Generate smooth trajectory using sine waves
        t = np.linspace(0, 4 * np.pi, num_frames)
        center_x = width / 2 + (width * 0.3) * np.sin(t) + np.random.randn(num_frames) * 3
        center_y = height / 2 + (height * 0.25) * np.cos(t * 0.7) + np.random.randn(num_frames) * 3

        # Generate keypoints around center with realistic offsets
        keypoints = np.zeros((num_frames, num_keypoints, 3))
        keypoint_names = self.get_keypoint_names()

        for i, name in enumerate(keypoint_names):
            # Assign consistent offset per keypoint
            np.random.seed(hash(name) % (2**32))
            offset_x = np.random.randn() * 15
            offset_y = np.random.randn() * 15

            # Add small temporal noise
            keypoints[:, i, 0] = center_x + offset_x + np.random.randn(num_frames) * 2
            keypoints[:, i, 1] = center_y + offset_y + np.random.randn(num_frames) * 2
            keypoints[:, i, 2] = 0.85 + np.random.rand(num_frames) * 0.15  # confidence

        return {
            "keypoints": keypoints,
            "keypoints_df": None,
            "metadata": {
                "num_frames": num_frames,
                "num_keypoints": num_keypoints,
                "mock_data": True,
            },
            "video_path": str(video_path),
            "model_type": self.model_type,
            "model_name": self.model_name,
        }

    def _load_predictions(self, video_path: Path, output_dir: Path) -> Dict:
        """Load prediction results from DeepLabCut output."""
        # DeepLabCut saves results with specific naming convention
        video_name = video_path.stem

        # Find the output file (h5 or csv)
        h5_files = list(output_dir.glob(f"{video_name}*.h5"))
        csv_files = list(output_dir.glob(f"{video_name}*.csv"))

        keypoints_df = None
        if h5_files:
            keypoints_df = pd.read_hdf(h5_files[0])
            logger.info(f"Loaded predictions from: {h5_files[0]}")
        elif csv_files:
            keypoints_df = pd.read_csv(csv_files[0], header=[0, 1, 2], index_col=0)
            logger.info(f"Loaded predictions from: {csv_files[0]}")

        if keypoints_df is None:
            logger.warning("No prediction files found, returning empty results")
            return {"keypoints": None, "metadata": {}}

        # Parse keypoints into structured format
        keypoints = self._parse_keypoints(keypoints_df)

        return {
            "keypoints": keypoints,
            "keypoints_df": keypoints_df,
            "metadata": {
                "num_frames": len(keypoints_df),
                "num_keypoints": self.model_config["num_keypoints"],
            },
        }

    def _parse_keypoints(self, df: pd.DataFrame) -> np.ndarray:
        """
        Parse keypoints DataFrame into numpy array.
        Supports both DLC 2.x (3-level) and DLC 3.0 (4-level multi-animal) formats.

        Returns:
            Array of shape (num_frames, num_keypoints, 3) for [x, y, confidence]
        """
        try:
            scorer = df.columns.get_level_values(0)[0]
            num_levels = df.columns.nlevels

            # DLC 3.0 multi-animal format: scorer -> individuals -> bodyparts -> coords
            if num_levels == 4:
                individual = df.columns.get_level_values(1)[0]  # Use first animal (animal0)
                animal_df = df[scorer][individual]
                bodyparts = animal_df.columns.get_level_values(0).unique()
                logger.info(f"Parsing DLC 3.0 multi-animal format (individual: {individual})")
            # DLC 2.x format: scorer -> bodyparts -> coords
            else:
                animal_df = df[scorer]
                bodyparts = animal_df.columns.get_level_values(0).unique()
                logger.info("Parsing DLC 2.x format")

            num_frames = len(df)
            num_keypoints = len(bodyparts)

            keypoints = np.zeros((num_frames, num_keypoints, 3))

            for i, bp in enumerate(bodyparts):
                keypoints[:, i, 0] = animal_df[bp]["x"].values
                keypoints[:, i, 1] = animal_df[bp]["y"].values
                if "likelihood" in animal_df[bp].columns:
                    keypoints[:, i, 2] = animal_df[bp]["likelihood"].values
                else:
                    keypoints[:, i, 2] = 1.0

            logger.info(f"Parsed {num_frames} frames, {num_keypoints} keypoints")
            return keypoints
        except Exception as e:
            logger.warning(f"Error parsing keypoints: {e}")
            return df.values.reshape(-1, self.model_config["num_keypoints"], 3)

    def get_keypoint_names(self, filtered: bool = True) -> List[str]:
        """
        Get list of keypoint names for the current model.

        Args:
            filtered: If True and use_keypoints is set, return only filtered keypoints

        Returns:
            List of keypoint names
        """
        # DLC 3.0 SuperAnimal keypoint names (from actual model output)
        if self.model_type == "topviewmouse":
            all_keypoints = [
                "nose", "left_ear", "right_ear", "left_ear_tip", "right_ear_tip",
                "left_eye", "right_eye", "neck", "mid_back", "mouse_center",
                "mid_backend", "mid_backend2", "mid_backend3", "tail_base",
                "tail1", "tail2", "tail3", "tail4", "tail5",
                "left_shoulder", "left_midside", "left_hip",
                "right_shoulder", "right_midside", "right_hip",
                "tail_end", "head_midpoint",
            ]
        elif self.model_type == "quadruped":
            all_keypoints = [
                "nose", "upper_jaw", "lower_jaw", "mouth_end_right", "mouth_end_left",
                "right_eye", "right_earbase", "right_earend", "right_antler_base",
                "right_antler_end", "left_eye", "left_earbase", "left_earend",
                "left_antler_base", "left_antler_end", "neck_base", "neck_end",
                "throat_base", "throat_end", "back_base", "back_end", "back_middle",
                "tail_base", "tail_end", "tail_middle", "left_front_hoof",
                "left_front_knee", "left_front_paw", "left_front_elbow",
                "right_front_hoof", "right_front_knee", "right_front_paw",
                "right_front_elbow", "left_back_hoof", "left_back_knee",
                "left_back_paw", "left_back_elbow", "right_back_hoof",
                "right_back_knee", "right_back_paw", "right_back_elbow", "belly_bottom",
            ]
        else:
            all_keypoints = []

        # Apply filtering if requested
        if filtered and self.use_keypoints:
            return [kp for kp in self.use_keypoints if kp in all_keypoints]
        return all_keypoints

    def filter_keypoints(
        self,
        keypoints: np.ndarray,
        all_keypoint_names: List[str]
    ) -> tuple:
        """
        Filter keypoints array to only include specified keypoints.

        Args:
            keypoints: Full keypoints array (frames, all_keypoints, 3)
            all_keypoint_names: List of all keypoint names from model

        Returns:
            Tuple of (filtered_keypoints, filtered_names)
        """
        if not self.use_keypoints:
            return keypoints, all_keypoint_names

        # Find indices of keypoints to keep
        indices = []
        filtered_names = []
        for kp in self.use_keypoints:
            if kp in all_keypoint_names:
                indices.append(all_keypoint_names.index(kp))
                filtered_names.append(kp)
            else:
                logger.warning(f"Keypoint '{kp}' not found in model output, skipping")

        if not indices:
            logger.warning("No valid keypoints found, returning all keypoints")
            return keypoints, all_keypoint_names

        filtered_keypoints = keypoints[:, indices, :]
        logger.info(f"Filtered keypoints: {len(all_keypoint_names)} -> {len(filtered_names)}")
        return filtered_keypoints, filtered_names
