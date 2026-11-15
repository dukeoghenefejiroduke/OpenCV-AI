"""Image Quality Gate for CementSight AI (vision/quality_gate.py).

Performs pre-inspection image quality assessment including blur (Laplacian variance),
exposure (mean intensity & dark pixel share), blown highlights (compact bright blobs),
and contrast on a standardized downscaled working image (longest side 640).
Protects analytical detectors by flagging unsuitable or degraded imagery before inspection.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import cv2


class ImageQualityGate:
    """Evaluates image quality metrics prior to hazard screening on standardized 640p scale."""

    # Constants with engineering rationale
    TARGET_LONG_SIDE: int = 640              # Standardized resolution for invariant metric computation
    BLUR_LAPLACIAN_THRESHOLD: float = 80.0   # Below 80 indicates excessive blur/loss of high-frequency detail at 640p
    EXPOSURE_MIN_MEAN: float = 25.0          # Below 25 indicates severe under-exposure (too dark)
    EXPOSURE_MAX_MEAN: float = 248.0         # Above 248 indicates severe over-exposure (washed out)
    DARK_PIXEL_THRESHOLD: int = 30           # Value threshold for very dark pixels
    DARK_PIXEL_MAX_SHARE: float = 0.35       # Max 35% very dark pixels before flagging extreme darkness
    BLOWN_VALUE_THRESHOLD: int = 250         # High value (V >= 250) for blown highlights
    BLOWN_SAT_THRESHOLD: int = 30            # Low saturation (S <= 30) for white specular reflection
    BLOWN_MIN_BLOB_AREA: int = 40            # Minimum contiguous blob area to qualify as blown highlight
    BLOWN_MAX_FRACTION: float = 0.05         # Max 5% of area in blown blob regions before flagging
    CONTRAST_MIN_STD: float = 18.0           # Grayscale standard deviation below 18 indicates low contrast

    def evaluate(self, image: np.ndarray) -> Dict[str, Any]:
        """Evaluates image quality across blur, exposure, blown highlights, and contrast.

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

        # Downscale to fixed size (longest side 640) for consistent metric evaluation
        h, w = image.shape[:2]
        long_side = max(h, w)
        if long_side > self.TARGET_LONG_SIDE:
            scale = self.TARGET_LONG_SIDE / float(long_side)
            new_w = max(64, int(w * scale))
            new_h = max(64, int(h * scale))
            working_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            working_img = image.copy()

        gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
        wh, ww = gray.shape[:2]
        total_pixels = wh * ww

        # 1. Blur Check (Laplacian Variance on 640p grayscale)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = laplacian_var < self.BLUR_LAPLACIAN_THRESHOLD

        # 2. Exposure & Darkness Check
        mean_intensity = float(np.mean(gray))
        is_under_exposed = mean_intensity < self.EXPOSURE_MIN_MEAN
        is_over_exposed = mean_intensity > self.EXPOSURE_MAX_MEAN

        dark_pixel_count = int(np.sum(gray < self.DARK_PIXEL_THRESHOLD))
        dark_pixel_share = float(dark_pixel_count) / float(total_pixels)
        is_extremely_dark = (mean_intensity < 45.0) and (dark_pixel_share > self.DARK_PIXEL_MAX_SHARE)

        # 3. Blown Highlights (Specular Glare in compact blobs)
        hsv = cv2.cvtColor(working_img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        blown_mask = ((val >= self.BLOWN_VALUE_THRESHOLD) & (sat <= self.BLOWN_SAT_THRESHOLD)).astype(np.uint8) * 255

        # Filter blown mask for compact blobs via connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blown_mask, connectivity=8)
        valid_blown_pixels = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.BLOWN_MIN_BLOB_AREA:
                valid_blown_pixels += area

        blown_fraction = float(valid_blown_pixels) / float(total_pixels)
        has_blown_highlights = blown_fraction > self.BLOWN_MAX_FRACTION

        # 4. Contrast Check (Standard Deviation)
        contrast_std = float(np.std(gray))
        has_low_contrast = contrast_std < self.CONTRAST_MIN_STD

        reasons = []
        advisory_notes = []
        if is_blurry:
            reasons.append(f"Image is blurry (Laplacian variance {laplacian_var:.1f} < {self.BLUR_LAPLACIAN_THRESHOLD}).")
        if is_under_exposed or is_extremely_dark:
            reasons.append(f"Image is severely under-exposed / too dark (mean intensity {mean_intensity:.1f}, dark pixel share {dark_pixel_share*100:.1f}%).")
        if is_over_exposed:
            reasons.append(f"Image is over-exposed (mean intensity {mean_intensity:.1f} > {self.EXPOSURE_MAX_MEAN}).")
        if has_blown_highlights:
            if blown_fraction > 0.25 and (contrast_std < 12.0 or laplacian_var < 40.0):
                reasons.append(f"Excessive blown highlights and loss of texture ({blown_fraction * 100:.1f}% clipped area, contrast std {contrast_std:.1f}).")
            else:
                advisory_notes.append(f"Advisory: Blown highlights detected ({blown_fraction * 100:.1f}% in compact blobs).")
        if has_low_contrast:
            reasons.append(f"Low image contrast (grayscale std {contrast_std:.1f} < {self.CONTRAST_MIN_STD}).")

        verdict = "LOW_QUALITY" if reasons else "OK"
        recommendation = "retake image recommended" if verdict == "LOW_QUALITY" else "none"

        return {
            "verdict": verdict,
            "reasons": reasons,
            "advisory_notes": advisory_notes,
            "recommendation": recommendation,
            "metrics": {
                "laplacian_variance": round(laplacian_var, 2),
                "mean_intensity": round(mean_intensity, 2),
                "dark_pixel_share": round(dark_pixel_share, 4),
                "blown_fraction": round(blown_fraction, 4),
                "contrast_std": round(contrast_std, 2)
            }
        }
