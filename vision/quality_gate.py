"""Image Quality Gate for CementSight AI (vision/quality_gate.py).

Performs pre-inspection image quality assessment including blur (Laplacian variance),
exposure (mean intensity), glare saturation, and contrast. Protects analytical detectors
by flagging unsuitable or degraded imagery before inspection.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import cv2


class ImageQualityGate:
    """Evaluates image quality metrics prior to hazard screening."""

    # Constants with engineering rationale
    BLUR_LAPLACIAN_THRESHOLD: float = 100.0  # Below 100 indicates excessive blur/loss of high-frequency detail
    EXPOSURE_MIN_MEAN: float = 25.0          # Below 25 indicates severe under-exposure (too dark)
    EXPOSURE_MAX_MEAN: float = 230.0         # Above 230 indicates severe over-exposure (washed out)
    GLARE_VALUE_THRESHOLD: int = 240         # High value indicating specular reflection / glare
    GLARE_SAT_THRESHOLD: int = 30            # Low saturation for pure white specular glare
    GLARE_MAX_FRACTION: float = 0.05         # Max 5% of pixels can be glare before flagging
    CONTRAST_MIN_STD: float = 20.0           # Grayscale standard deviation below 20 indicates low contrast

    def evaluate(self, image: np.ndarray) -> Dict[str, Any]:
        """Evaluates image quality across blur, exposure, glare, and contrast.

        Args:
            image: BGR numpy array.

        Returns:
            Dictionary containing verdict ('OK' or 'LOW_QUALITY'), reasons list, recommendation, and metrics.
        """
        if image is None or image.size == 0:
            return {
                "verdict": "LOW_QUALITY",
                "reasons": ["Image is empty or unreadable."],
                "recommendation": "retake image recommended",
                "metrics": {}
            }

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]

        # 1. Blur Check (Laplacian Variance)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = laplacian_var < self.BLUR_LAPLACIAN_THRESHOLD

        # 2. Exposure Check (Mean Intensity)
        mean_intensity = float(np.mean(gray))
        is_under_exposed = mean_intensity < self.EXPOSURE_MIN_MEAN
        is_over_exposed = mean_intensity > self.EXPOSURE_MAX_MEAN

        # 3. Glare / Specular Reflection Check
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        glare_mask = (val > self.GLARE_VALUE_THRESHOLD) & (sat < self.GLARE_SAT_THRESHOLD)
        glare_fraction = float(np.sum(glare_mask)) / float(height * width)
        has_glare = glare_fraction > self.GLARE_MAX_FRACTION

        # 4. Contrast Check (Standard Deviation)
        contrast_std = float(np.std(gray))
        has_low_contrast = contrast_std < self.CONTRAST_MIN_STD

        reasons = []
        if is_blurry:
            reasons.append(f"Image is blurry (Laplacian variance {laplacian_var:.1f} < {self.BLUR_LAPLACIAN_THRESHOLD}).")
        if is_under_exposed:
            reasons.append(f"Image is under-exposed (mean intensity {mean_intensity:.1f} < {self.EXPOSURE_MIN_MEAN}).")
        if is_over_exposed:
            reasons.append(f"Image is over-exposed (mean intensity {mean_intensity:.1f} > {self.EXPOSURE_MAX_MEAN}).")
        if has_glare:
            reasons.append(f"Excessive specular glare detected ({glare_fraction * 100:.1f}% > {self.GLARE_MAX_FRACTION * 100}% of pixels).")
        if has_low_contrast:
            reasons.append(f"Low image contrast (grayscale std {contrast_std:.1f} < {self.CONTRAST_MIN_STD}).")

        verdict = "LOW_QUALITY" if reasons else "OK"
        recommendation = "retake image recommended" if verdict == "LOW_QUALITY" else "none"

        return {
            "verdict": verdict,
            "reasons": reasons,
            "recommendation": recommendation,
            "metrics": {
                "laplacian_variance": round(laplacian_var, 2),
                "mean_intensity": round(mean_intensity, 2),
                "glare_fraction": round(glare_fraction, 4),
                "contrast_std": round(contrast_std, 2)
            }
        }
