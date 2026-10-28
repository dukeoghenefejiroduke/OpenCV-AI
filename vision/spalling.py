"""Multi-Defect Spalling & Edge Degradation Detection (vision/spalling.py).

Detects structural cracking, concrete spalling, and edge fractures on cement plant equipment
using Canny edge detection and contour geometric analysis.
"""

import cv2
import numpy as np
from typing import Dict, Any


def detect_spalling_and_cracks(image: np.ndarray) -> Dict[str, Any]:
    """Analyzes an image for structural cracks and spalling defects using edge gradient analysis.

    Args:
        image: BGR NumPy array of the industrial asset.

    Returns:
        Dictionary containing crack count, spalling severity score, and defect bounding metrics.
    """
    if image is None or image.size == 0:
        return {"crack_count": 0, "spalling_score": 0.0, "status": "INVALID_IMAGE"}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection for structural cracks and fracture edges
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours of edge fragments
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crack_count = 0
    total_edge_pixels = int(np.sum(edges > 0))
    total_pixels = gray.shape[0] * gray.shape[1]
    spalling_score = float(total_edge_pixels / total_pixels) * 100.0

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 20:  # significant linear or jagged edge
            crack_count += 1

    return {
        "crack_count": crack_count,
        "spalling_score": round(spalling_score, 2),
        "status": "DETECTED" if crack_count > 5 else "NORMAL"
    }
