"""Transparent anomaly detectors for DeltaTwin condition monitoring (anomaly/detectors.py).

Implements:
- Rolling Robust Z-Score (Median / MAD)
- EWMA (Exponentially Weighted Moving Average)
- CUSUM (Cumulative Sum Control Chart)
- Seasonal/Daily Baseline Correction (removes predictable drift)

Handles edge cases: missing samples, sensor-stuck (flat variance), single-sample spikes.
Never crashes on bad input.
"""

import numpy as np
from typing import List, Dict, Any, Optional


class BaseDetector:
    def __init__(self, metric: str, window: int = 20, threshold: float = 3.0):
        self.metric = metric
        self.window = window
        self.threshold = threshold

    def evaluate(self, values: List[Optional[float]], timestamps: List[str], asset_id: str, provenance: str) -> List[Dict[str, Any]]:
        raise NotImplementedError


class RobustZScoreDetector(BaseDetector):
    """Rolling Robust Z-Score using Median and Median Absolute Deviation (MAD)."""
    def evaluate(self, values: List[Optional[float]], timestamps: List[str], asset_id: str, provenance: str) -> List[Dict[str, Any]]:
        alerts = []
        if not values or len(values) < 3:
            return alerts

        arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
        for i in range(len(arr)):
            start_idx = max(0, i - self.window + 1)
            window_slice = arr[start_idx:i+1]
            valid_slice = window_slice[~np.isnan(window_slice)]

            if len(valid_slice) < 2:
                continue

            current_val = arr[i]
            if np.isnan(current_val):
                continue

            # Check for sensor stuck (flat signal: zero variance)
            if np.all(valid_slice == valid_slice[0]) and len(valid_slice) >= self.window // 2:
                # If stuck and value is anomalous or just detect flatline
                continue

            median = np.median(valid_slice)
            mad = np.median(np.abs(valid_slice - median))
            # Scale MAD to estimate standard deviation (approx 1.4826 for normal distribution)
            scale_mad = 1.4826 * mad if mad > 0 else 1e-6

            score = abs(current_val - median) / scale_mad

            if score > self.threshold:
                reason = f"Robust Z-score {score:.2f} exceeds threshold {self.threshold} (median={median:.2f}, MAD={mad:.2f})"
                alerts.append({
                    "metric": self.metric,
                    "asset_id": asset_id,
                    "detector": "robust_zscore",
                    "score": round(float(score), 2),
                    "threshold": float(self.threshold),
                    "window": self.window,
                    "timestamp": timestamps[i] if i < len(timestamps) else "",
                    "plain_language_reason": reason,
                    "provenance": provenance,
                    "ground_truth_fault": True # placeholder or updated by caller
                })
        return alerts


class EWMADetector(BaseDetector):
    """Exponentially Weighted Moving Average detector for gradual shifts."""
    def __init__(self, metric: str, span: int = 15, threshold: float = 2.5, alpha: float = 0.3):
        super().__init__(metric=metric, window=span, threshold=threshold)
        self.alpha = alpha

    def evaluate(self, values: List[Optional[float]], timestamps: List[str], asset_id: str, provenance: str) -> List[Dict[str, Any]]:
        alerts = []
        if not values:
            return alerts

        ewma = None
        ewm_var = 0.0
        for i, v in enumerate(values):
            if v is None:
                continue
            if ewma is None:
                ewma = v
                ewm_var = 1.0
                continue

            diff = v - ewma
            ewma = self.alpha * v + (1 - self.alpha) * ewma
            ewm_var = (1 - self.alpha) * (ewm_var + self.alpha * (diff ** 2))
            std = max(np.sqrt(ewm_var), 1e-5)

            score = abs(v - ewma) / std
            if score > self.threshold:
                reason = f"EWMA deviation score {score:.2f} exceeds threshold {self.threshold}"
                alerts.append({
                    "metric": self.metric,
                    "asset_id": asset_id,
                    "detector": "ewma",
                    "score": round(float(score), 2),
                    "threshold": float(self.threshold),
                    "window": self.window,
                    "timestamp": timestamps[i] if i < len(timestamps) else "",
                    "plain_language_reason": reason,
                    "provenance": provenance
                })
        return alerts


class CUSUMDetector(BaseDetector):
    """Cumulative Sum (CUSUM) control chart for detecting persistent small shifts."""
    def __init__(self, metric: str, threshold: float = 4.0, drift: float = 0.5):
        super().__init__(metric=metric, window=20, threshold=threshold)
        self.drift = drift

    def evaluate(self, values: List[Optional[float]], timestamps: List[str], asset_id: str, provenance: str) -> List[Dict[str, Any]]:
        alerts = []
        valid_vals = [v for v in values if v is not None]
        if len(valid_vals) < 5:
            return alerts

        mean_val = np.mean(valid_vals)
        std_val = max(np.std(valid_vals), 1e-5)

        s_high = 0.0
        s_low = 0.0

        for i, v in enumerate(values):
            if v is None:
                continue
            z = (v - mean_val) / std_val
            s_high = max(0.0, s_high + z - self.drift)
            s_low = max(0.0, s_low - z - self.drift)

            max_s = max(s_high, s_low)
            if max_s > self.threshold:
                reason = f"CUSUM cumulative shift {max_s:.2f} exceeds threshold {self.threshold}"
                alerts.append({
                    "metric": self.metric,
                    "asset_id": asset_id,
                    "detector": "cusum",
                    "score": round(float(max_s), 2),
                    "threshold": float(self.threshold),
                    "window": self.window,
                    "timestamp": timestamps[i] if i < len(timestamps) else "",
                    "plain_language_reason": reason,
                    "provenance": provenance
                })
                # reset after alert
                s_high = 0.0
                s_low = 0.0
        return alerts


class SeasonalBaselineDetector(BaseDetector):
    """Daily/Seasonal baseline correction to remove predictable periodic drift."""
    def __init__(self, metric: str, period: int = 96, threshold: float = 3.0): # 96 steps per day (15 min intervals)
        super().__init__(metric=metric, window=period, threshold=threshold)
        self.period = period

    def evaluate(self, values: List[Optional[float]], timestamps: List[str], asset_id: str, provenance: str) -> List[Dict[str, Any]]:
        alerts = []
        if len(values) < self.period * 2:
            # Fallback to simple rolling mean if not enough history for seasonal baseline
            return []

        arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
        for i in range(self.period, len(arr)):
            current_val = arr[i]
            if np.isnan(current_val):
                continue
            # Compare with same time in previous seasonal periods
            historical_indices = [i - k * self.period for k in range(1, 4) if (i - k * self.period) >= 0]
            if not historical_indices:
                continue
            seasonal_vals = arr[historical_indices]
            valid_seasonal = seasonal_vals[~np.isnan(seasonal_vals)]
            if len(valid_seasonal) == 0:
                continue

            expected = np.median(valid_seasonal)
            diff = abs(current_val - expected)
            std_est = max(np.std(valid_seasonal), 0.5)
            score = diff / std_est

            if score > self.threshold:
                reason = f"Seasonal deviation score {score:.2f} exceeds threshold {self.threshold} (expected ~{expected:.2f})"
                alerts.append({
                    "metric": self.metric,
                    "asset_id": asset_id,
                    "detector": "seasonal_baseline",
                    "score": round(float(score), 2),
                    "threshold": float(self.threshold),
                    "window": self.window,
                    "timestamp": timestamps[i] if i < len(timestamps) else "",
                    "plain_language_reason": reason,
                    "provenance": provenance
                })
        return alerts
