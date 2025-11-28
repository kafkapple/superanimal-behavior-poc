"""
Action Recognition Models for Behavior Classification.

Includes:
- RuleBasedClassifier: Velocity-based heuristics (baseline)
- LSTMClassifier: LSTM-based sequence model
- TransformerClassifier: Transformer-based sequence model
- MLP Classifier: Simple feedforward network

All models implement a common interface for easy comparison.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Neural network models will be disabled.")


# ============================================================
# Common Data Structures
# ============================================================

@dataclass
class ClassificationResult:
    """Result from action classification."""
    predictions: np.ndarray  # (num_frames,) predicted labels
    probabilities: np.ndarray  # (num_frames, num_classes) class probabilities
    model_name: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModelMetrics:
    """Evaluation metrics for a model."""
    accuracy: float
    f1_macro: float
    f1_per_class: Dict[str, float]
    confusion_matrix: np.ndarray
    model_name: str


# ============================================================
# Base Classifier Interface
# ============================================================

class BaseActionClassifier(ABC):
    """Abstract base class for action classifiers."""

    def __init__(self, num_classes: int = 4, class_names: List[str] = None):
        """
        Initialize classifier.

        Args:
            num_classes: Number of action classes
            class_names: List of class names (e.g., ['stationary', 'walking', 'running', 'other'])
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> Dict:
        """
        Train the classifier.

        Args:
            X_train: Training features (num_samples, seq_len, num_features)
            y_train: Training labels (num_samples,) or (num_samples, seq_len)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Training history/metrics
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> ClassificationResult:
        """
        Predict action classes.

        Args:
            X: Input features (num_samples, seq_len, num_features)

        Returns:
            ClassificationResult with predictions and probabilities
        """
        pass

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ModelMetrics:
        """
        Evaluate classifier on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            ModelMetrics with evaluation results
        """
        result = self.predict(X_test)
        predictions = result.predictions

        # Flatten if needed
        y_true = y_test.flatten()
        y_pred = predictions.flatten()

        # Calculate accuracy
        accuracy = np.mean(y_true == y_pred)

        # Calculate F1 scores
        f1_per_class = {}
        f1_scores = []

        for i, class_name in enumerate(self.class_names):
            tp = np.sum((y_true == i) & (y_pred == i))
            fp = np.sum((y_true != i) & (y_pred == i))
            fn = np.sum((y_true == i) & (y_pred != i))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            f1_per_class[class_name] = f1
            f1_scores.append(f1)

        f1_macro = np.mean(f1_scores)

        # Confusion matrix
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
        for true_label, pred_label in zip(y_true, y_pred):
            if 0 <= true_label < self.num_classes and 0 <= pred_label < self.num_classes:
                cm[int(true_label), int(pred_label)] += 1

        return ModelMetrics(
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_per_class=f1_per_class,
            confusion_matrix=cm,
            model_name=self.__class__.__name__,
        )

    def save(self, path: Path):
        """Save model to disk."""
        pass

    def load(self, path: Path):
        """Load model from disk."""
        pass


# ============================================================
# Rule-Based Classifier (Baseline)
# ============================================================

class RuleBasedClassifier(BaseActionClassifier):
    """
    Rule-based action classifier using velocity thresholds.

    This is the baseline model that doesn't require training.
    It uses hand-crafted rules based on movement velocity.
    """

    def __init__(
        self,
        num_classes: int = 4,
        class_names: List[str] = None,
        fps: float = 30.0,
        thresholds: Dict[str, float] = None,
    ):
        """
        Initialize rule-based classifier.

        Args:
            num_classes: Number of action classes
            class_names: Class names
            fps: Video frame rate
            thresholds: Velocity thresholds for classification
        """
        super().__init__(num_classes, class_names or ["stationary", "walking", "running", "other"])
        self.fps = fps
        self.thresholds = thresholds or {
            "stationary": 0.5,  # body-lengths/sec
            "walking": 3.0,
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> Dict:
        """Rule-based classifier doesn't need training."""
        logger.info("RuleBasedClassifier: No training required (using hand-crafted rules)")
        return {"status": "no_training_required"}

    def predict(self, X: np.ndarray) -> ClassificationResult:
        """
        Predict using velocity-based rules.

        Args:
            X: Input features (num_samples, seq_len, num_features)
               Expected: features include velocity or position data

        Returns:
            ClassificationResult
        """
        # Handle different input shapes
        if X.ndim == 2:
            X = X.reshape(1, *X.shape)

        num_samples, seq_len, num_features = X.shape

        # Compute velocity from position if not directly provided
        # Assume first 2 features are x, y positions
        if num_features >= 2:
            positions = X[:, :, :2]
            velocity = np.zeros((num_samples, seq_len))

            for i in range(num_samples):
                disp = np.diff(positions[i], axis=0)
                vel = np.linalg.norm(disp, axis=1)
                velocity[i, 1:] = vel
                velocity[i, 0] = vel[0] if len(vel) > 0 else 0

            # Estimate body size (use median distance as proxy)
            body_size = 50.0  # Default
            if num_features >= 14:  # Multiple keypoints
                # Estimate from keypoint spread
                kp_spread = np.median(np.std(X[:, :, :14].reshape(-1, 7, 2), axis=1))
                body_size = max(kp_spread * 4, 20.0)

            # Normalize velocity to body-lengths per second
            normalized_velocity = (velocity / body_size) * self.fps
        else:
            # If only velocity is provided
            normalized_velocity = X[:, :, 0]

        # Apply rules
        predictions = np.zeros((num_samples, seq_len), dtype=np.int32)
        probabilities = np.zeros((num_samples, seq_len, self.num_classes))

        stationary_thresh = self.thresholds["stationary"]
        walking_thresh = self.thresholds["walking"]

        for i in range(num_samples):
            for j in range(seq_len):
                v = normalized_velocity[i, j]

                if v < stationary_thresh:
                    predictions[i, j] = 0  # stationary
                    probabilities[i, j] = [0.8, 0.15, 0.03, 0.02]
                elif v < walking_thresh:
                    predictions[i, j] = 1  # walking
                    probabilities[i, j] = [0.1, 0.7, 0.15, 0.05]
                else:
                    predictions[i, j] = 2  # running
                    probabilities[i, j] = [0.05, 0.15, 0.75, 0.05]

        return ClassificationResult(
            predictions=predictions.flatten(),
            probabilities=probabilities.reshape(-1, self.num_classes),
            model_name="RuleBasedClassifier",
        )


# ============================================================
# MLP Classifier
# ============================================================

class MLPClassifier(BaseActionClassifier):
    """
    Simple MLP classifier for frame-by-frame classification.

    Uses a feedforward network with temporal context via windowing.
    """

    def __init__(
        self,
        num_classes: int = 4,
        class_names: List[str] = None,
        input_dim: int = 14,  # 7 keypoints * 2 (x, y)
        hidden_dims: List[int] = None,
        window_size: int = 1,  # Use 1 for frame-by-frame, increase for temporal context
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 64,
    ):
        """
        Initialize MLP classifier.

        Args:
            num_classes: Number of action classes
            class_names: Class names
            input_dim: Input feature dimension per frame
            hidden_dims: Hidden layer dimensions
            window_size: Temporal context window
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        super().__init__(num_classes, class_names)

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MLPClassifier")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_model()

    def _build_model(self):
        """Build the MLP model."""
        layers = []
        in_features = self.input_dim * self.window_size

        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            ])
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, self.num_classes))

        self.model = nn.Sequential(*layers).to(self.device)

    def _create_windows(self, X: np.ndarray, y: np.ndarray = None) -> Tuple:
        """Create windowed samples from sequences."""
        # Handle different input shapes
        if X.ndim == 2:
            X = X.reshape(1, *X.shape)

        num_samples, seq_len, num_features = X.shape
        half_window = self.window_size // 2

        X_windows = []
        y_windows = []

        for i in range(num_samples):
            for j in range(seq_len):
                # Extract window with padding
                start = max(0, j - half_window)
                end = min(seq_len, j + half_window + 1)

                window = X[i, start:end, :]

                # Pad if needed
                if window.shape[0] < self.window_size:
                    pad_before = max(0, half_window - j)
                    pad_after = self.window_size - window.shape[0] - pad_before
                    window = np.pad(window, ((pad_before, pad_after), (0, 0)), mode='edge')

                X_windows.append(window.flatten())
                if y is not None:
                    y_windows.append(y[i, j] if y.ndim > 1 else y[j])

        X_windows = np.array(X_windows)
        y_windows = np.array(y_windows) if y is not None else None

        return X_windows, y_windows

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> Dict:
        """Train the MLP model."""
        # Prepare data
        X_train_w, y_train_w = self._create_windows(X_train, y_train)

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_w),
            torch.LongTensor(y_train_w)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_w, y_val_w = self._create_windows(X_val, y_val)
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_w),
                torch.LongTensor(y_val_w)
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(y_batch).sum().item()
                train_total += y_batch.size(0)

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)

                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_correct += predicted.eq(y_batch).sum().item()
                        val_total += y_batch.size(0)

                val_loss /= len(val_loader)
                val_acc = val_correct / val_total

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            if (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{self.epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}"
                if val_loader:
                    msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                logger.info(msg)

        return history

    def predict(self, X: np.ndarray) -> ClassificationResult:
        """Predict using the MLP model."""
        X_windows, _ = self._create_windows(X)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_windows).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            predictions = outputs.argmax(dim=1).cpu().numpy()

        return ClassificationResult(
            predictions=predictions,
            probabilities=probabilities,
            model_name="MLPClassifier",
        )


# ============================================================
# LSTM Classifier
# ============================================================

class LSTMClassifier(BaseActionClassifier):
    """
    LSTM-based sequence classifier for action recognition.

    Processes sequences of keypoints and outputs per-frame predictions.
    """

    def __init__(
        self,
        num_classes: int = 4,
        class_names: List[str] = None,
        input_dim: int = 14,  # 7 keypoints * 2 (x, y)
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        seq_len: int = 64,
    ):
        """
        Initialize LSTM classifier.

        Args:
            num_classes: Number of action classes
            class_names: Class names
            input_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            bidirectional: Use bidirectional LSTM
            dropout: Dropout rate
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Training batch size
            seq_len: Sequence length for training
        """
        super().__init__(num_classes, class_names)

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTMClassifier")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_model()

    def _build_model(self):
        """Build the LSTM model."""

        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, num_classes, bidirectional, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_dim, hidden_dim, num_layers,
                    batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0
                )
                lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
                self.fc = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(lstm_output_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes),
                )

            def forward(self, x):
                # x: (batch, seq_len, input_dim)
                lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim * 2)
                out = self.fc(lstm_out)  # (batch, seq_len, num_classes)
                return out

        self.model = LSTMModel(
            self.input_dim, self.hidden_dim, self.num_layers,
            self.num_classes, self.bidirectional, self.dropout
        ).to(self.device)

    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple:
        """Prepare fixed-length sequences for training."""
        # Handle different input shapes
        if X.ndim == 2:
            X = X.reshape(1, *X.shape)

        if y is not None and y.ndim == 1:
            y = y.reshape(1, -1)

        sequences_X = []
        sequences_y = []

        for i in range(len(X)):
            total_len = len(X[i])
            for start in range(0, total_len - self.seq_len + 1, self.seq_len // 2):
                end = start + self.seq_len
                sequences_X.append(X[i, start:end])
                if y is not None:
                    sequences_y.append(y[i, start:end])

        return np.array(sequences_X), np.array(sequences_y) if y is not None else None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> Dict:
        """Train the LSTM model."""
        # Prepare sequences
        X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train)

        if len(X_train_seq) == 0:
            logger.warning("No training sequences created. Data may be too short.")
            return {"status": "no_data"}

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_seq),
            torch.LongTensor(y_train_seq)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)
            if len(X_val_seq) > 0:
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val_seq),
                    torch.LongTensor(y_val_seq)
                )
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)  # (batch, seq_len, num_classes)
                outputs = outputs.view(-1, self.num_classes)
                y_batch = y_batch.view(-1)

                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(y_batch).sum().item()
                train_total += y_batch.size(0)

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        outputs = self.model(X_batch)
                        outputs = outputs.view(-1, self.num_classes)
                        y_batch = y_batch.view(-1)

                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_correct += predicted.eq(y_batch).sum().item()
                        val_total += y_batch.size(0)

                val_loss /= len(val_loader)
                val_acc = val_correct / val_total

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            if (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{self.epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}"
                if val_loader:
                    msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                logger.info(msg)

        return history

    def predict(self, X: np.ndarray) -> ClassificationResult:
        """Predict using the LSTM model."""
        # Handle different input shapes
        if X.ndim == 2:
            X = X.reshape(1, *X.shape)

        self.model.eval()
        all_predictions = []
        all_probabilities = []

        for seq in X:
            # Process in chunks if sequence is long
            seq_len = len(seq)
            predictions = np.zeros(seq_len, dtype=np.int32)
            probabilities = np.zeros((seq_len, self.num_classes))

            with torch.no_grad():
                for start in range(0, seq_len, self.seq_len):
                    end = min(start + self.seq_len, seq_len)
                    chunk = seq[start:end]

                    # Pad if needed
                    if len(chunk) < self.seq_len:
                        chunk = np.pad(chunk, ((0, self.seq_len - len(chunk)), (0, 0)), mode='edge')

                    X_tensor = torch.FloatTensor(chunk).unsqueeze(0).to(self.device)
                    outputs = self.model(X_tensor)  # (1, seq_len, num_classes)
                    probs = F.softmax(outputs, dim=2).squeeze(0).cpu().numpy()
                    preds = outputs.argmax(dim=2).squeeze(0).cpu().numpy()

                    actual_len = end - start
                    predictions[start:end] = preds[:actual_len]
                    probabilities[start:end] = probs[:actual_len]

            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)

        return ClassificationResult(
            predictions=np.array(all_predictions),
            probabilities=np.array(all_probabilities),
            model_name="LSTMClassifier",
        )


# ============================================================
# Transformer Classifier
# ============================================================

class TransformerClassifier(BaseActionClassifier):
    """
    Transformer-based sequence classifier for action recognition.

    Uses self-attention to model temporal dependencies in keypoint sequences.
    """

    def __init__(
        self,
        num_classes: int = 4,
        class_names: List[str] = None,
        input_dim: int = 14,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 0.0001,
        epochs: int = 50,
        batch_size: int = 32,
        seq_len: int = 64,
    ):
        """
        Initialize Transformer classifier.

        Args:
            num_classes: Number of action classes
            class_names: Class names
            input_dim: Input feature dimension
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Training batch size
            seq_len: Sequence length for training
        """
        super().__init__(num_classes, class_names)

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TransformerClassifier")

        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_model()

    def _build_model(self):
        """Build the Transformer model."""

        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)

            def forward(self, x):
                return x + self.pe[:, :x.size(1)]

        class TransformerModel(nn.Module):
            def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, d_model)
                self.pos_encoder = PositionalEncoding(d_model)

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

                self.fc = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, num_classes),
                )

            def forward(self, x):
                # x: (batch, seq_len, input_dim)
                x = self.input_proj(x)  # (batch, seq_len, d_model)
                x = self.pos_encoder(x)
                x = self.transformer(x)  # (batch, seq_len, d_model)
                out = self.fc(x)  # (batch, seq_len, num_classes)
                return out

        self.model = TransformerModel(
            self.input_dim, self.d_model, self.nhead, self.num_layers,
            self.dim_feedforward, self.num_classes, self.dropout
        ).to(self.device)

    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple:
        """Prepare fixed-length sequences for training."""
        # Handle different input shapes
        if X.ndim == 2:
            X = X.reshape(1, *X.shape)

        if y is not None and y.ndim == 1:
            y = y.reshape(1, -1)

        sequences_X = []
        sequences_y = []

        for i in range(len(X)):
            total_len = len(X[i])
            for start in range(0, total_len - self.seq_len + 1, self.seq_len // 2):
                end = start + self.seq_len
                sequences_X.append(X[i, start:end])
                if y is not None:
                    sequences_y.append(y[i, start:end])

        return np.array(sequences_X), np.array(sequences_y) if y is not None else None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> Dict:
        """Train the Transformer model."""
        # Prepare sequences
        X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train)

        if len(X_train_seq) == 0:
            logger.warning("No training sequences created. Data may be too short.")
            return {"status": "no_data"}

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_seq),
            torch.LongTensor(y_train_seq)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)
            if len(X_val_seq) > 0:
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val_seq),
                    torch.LongTensor(y_val_seq)
                )
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Training
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss()

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)  # (batch, seq_len, num_classes)
                outputs = outputs.view(-1, self.num_classes)
                y_batch = y_batch.view(-1)

                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(y_batch).sum().item()
                train_total += y_batch.size(0)

            scheduler.step()
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        outputs = self.model(X_batch)
                        outputs = outputs.view(-1, self.num_classes)
                        y_batch = y_batch.view(-1)

                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_correct += predicted.eq(y_batch).sum().item()
                        val_total += y_batch.size(0)

                val_loss /= len(val_loader)
                val_acc = val_correct / val_total

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            if (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{self.epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}"
                if val_loader:
                    msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                logger.info(msg)

        return history

    def predict(self, X: np.ndarray) -> ClassificationResult:
        """Predict using the Transformer model."""
        # Handle different input shapes
        if X.ndim == 2:
            X = X.reshape(1, *X.shape)

        self.model.eval()
        all_predictions = []
        all_probabilities = []

        for seq in X:
            seq_len = len(seq)
            predictions = np.zeros(seq_len, dtype=np.int32)
            probabilities = np.zeros((seq_len, self.num_classes))

            with torch.no_grad():
                for start in range(0, seq_len, self.seq_len):
                    end = min(start + self.seq_len, seq_len)
                    chunk = seq[start:end]

                    # Pad if needed
                    if len(chunk) < self.seq_len:
                        chunk = np.pad(chunk, ((0, self.seq_len - len(chunk)), (0, 0)), mode='edge')

                    X_tensor = torch.FloatTensor(chunk).unsqueeze(0).to(self.device)
                    outputs = self.model(X_tensor)
                    probs = F.softmax(outputs, dim=2).squeeze(0).cpu().numpy()
                    preds = outputs.argmax(dim=2).squeeze(0).cpu().numpy()

                    actual_len = end - start
                    predictions[start:end] = preds[:actual_len]
                    probabilities[start:end] = probs[:actual_len]

            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)

        return ClassificationResult(
            predictions=np.array(all_predictions),
            probabilities=np.array(all_probabilities),
            model_name="TransformerClassifier",
        )


# ============================================================
# Model Factory
# ============================================================

def get_action_classifier(
    model_name: str,
    num_classes: int = 4,
    class_names: List[str] = None,
    **kwargs
) -> BaseActionClassifier:
    """
    Get action classifier by name.

    Args:
        model_name: 'rule_based', 'mlp', 'lstm', or 'transformer'
        num_classes: Number of classes
        class_names: Class names
        **kwargs: Additional model arguments

    Returns:
        Classifier instance
    """
    model_name = model_name.lower().replace("-", "_")

    if model_name in ("rule_based", "rule", "baseline"):
        return RuleBasedClassifier(num_classes, class_names, **kwargs)
    elif model_name == "mlp":
        return MLPClassifier(num_classes, class_names, **kwargs)
    elif model_name == "lstm":
        return LSTMClassifier(num_classes, class_names, **kwargs)
    elif model_name in ("transformer", "attention"):
        return TransformerClassifier(num_classes, class_names, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: rule_based, mlp, lstm, transformer")


# ============================================================
# Comparison Utilities
# ============================================================

def compare_models(
    models: List[BaseActionClassifier],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, ModelMetrics]:
    """
    Train and compare multiple action recognition models.

    Args:
        models: List of classifier instances
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data

    Returns:
        Dictionary mapping model name to evaluation metrics
    """
    results = {}

    for model in models:
        model_name = model.__class__.__name__
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'='*50}")

        # Train
        history = model.fit(X_train, y_train, X_val, y_val)

        # Evaluate
        metrics = model.evaluate(X_test, y_test)

        results[model_name] = metrics

        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Accuracy: {metrics.accuracy:.4f}")
        logger.info(f"  F1 (macro): {metrics.f1_macro:.4f}")
        logger.info(f"  F1 per class: {metrics.f1_per_class}")

    return results
