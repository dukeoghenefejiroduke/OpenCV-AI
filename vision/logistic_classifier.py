"""Numpy-only Logistic Regression Classifier with Group-Aware CV and Evidence Gating (vision/logistic_classifier.py).

Stdlib + numpy only. Challenge pack and lockbox are never read during training.
"""

import os
import json
import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class NumpyLogisticRegression:
    def __init__(self, lr: float = 0.01, epochs: int = 500, l2: float = 0.001):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.calib_scale: float = 1.0
        self.calib_bias: float = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # Gradients with L2 regularization
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.l2 * self.weights)
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return np.zeros(X.shape[0])
        linear_model = np.dot(X, self.weights) + self.bias
        probs = self._sigmoid(linear_model)
        # Calibration transform
        calib_logits = self.calib_scale * np.log(np.clip(probs, 1e-7, 1 - 1e-7) / (1 - np.clip(probs, 1e-7, 1 - 1e-7))) + self.calib_bias
        return self._sigmoid(calib_logits)

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray):
        if self.weights is None:
            return
        raw_probs = self.predict_proba(X_val)
        # Simple Platt scaling fit using numpy
        eps = 1e-7
        logits = np.log(np.clip(raw_probs, eps, 1 - eps) / (1 - np.clip(raw_probs, eps, 1 - eps)))
        # Fit scale and bias on logits via simple gradient descent
        scale, bias = 1.0, 0.0
        lr = 0.05
        for _ in range(200):
            p = self._sigmoid(scale * logits + bias)
            ds = np.mean((p - y_val) * logits)
            db = np.mean(p - y_val)
            scale -= lr * ds
            bias -= lr * db
        self.calib_scale = float(scale)
        self.calib_bias = float(bias)


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = (z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)) / denominator
    return max(0.0, center - margin)


def check_evidence_gating(targets_path: str = "data/validation_targets.json", lockbox_results: Optional[Dict[str, Any]] = None) -> bool:
    """Refuses to enable classifier unless targets file is signed and lockbox lower bound meets it."""
    if not os.path.exists(targets_path):
        return False
    try:
        with open(targets_path, "r", encoding="utf-8") as f:
            targets = json.load(f)
    except Exception:
        return False

    if targets.get("status") != "sealed" or not targets.get("signature"):
        return False

    if not lockbox_results:
        return False

    required_lb = float(targets.get("min_wilson_lower_bound", 0.0))
    actual_lb = float(lockbox_results.get("wilson_lower_bound", 0.0))
    return actual_lb >= required_lb
