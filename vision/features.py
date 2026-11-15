"""Feature extraction module for CementSight AI (vision/features.py).

Extracts colour statistics, Lab/HSV mean/std, local roughness, edge density,
region shape, and specular share using stdlib + numpy + OpenCV only.
"""

import cv2
import numpy as np
from typing import Dict, Any, Union


def extract_features(image_or_path: Union[str, np.ndarray]) -> Dict[str, float]:
    """Extracts numerical image features for evidence-gated classification."""
    if isinstance(image_or_path, str):
        img = cv2.imread(image_or_path)
    else:
        img = image_or_path

    if img is None:
        raise ValueError("Invalid image input for feature extraction")

    h, w = img.shape[:2]
    total_pixels = h * w
    if total_pixels == 0:
        total_pixels = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h_mean, s_mean, v_mean = float(np.mean(hsv[:,:,0])), float(np.mean(hsv[:,:,1])), float(np.mean(hsv[:,:,2]))
    h_std, s_std, v_std = float(np.std(hsv[:,:,0])), float(np.std(hsv[:,:,1])), float(np.std(hsv[:,:,2]))

    l_mean, a_mean, b_mean = float(np.mean(lab[:,:,0])), float(np.mean(lab[:,:,1])), float(np.mean(lab[:,:,2]))
    l_std, a_std, b_std = float(np.std(lab[:,:,0])), float(np.std(lab[:,:,1])), float(np.std(lab[:,:,2]))

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    roughness = float(np.var(laplacian))

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / float(total_pixels)

    specular_share = float(np.sum(gray > 240)) / float(total_pixels)

    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mean_area = 0.0
    mean_circularity = 0.0
    mean_aspect_ratio = 1.0
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        mean_area = float(np.mean(areas)) / float(total_pixels)
        circularities = []
        aspect_ratios = []
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            area = cv2.contourArea(c)
            if perimeter > 0:
                circ = (4.0 * np.pi * area) / (perimeter ** 2)
                circularities.append(circ)
            x, y, cw, ch = cv2.boundingRect(c)
            if ch > 0:
                aspect_ratios.append(float(cw) / float(ch))
        if circularities:
            mean_circularity = float(np.mean(circularities))
        if aspect_ratios:
            mean_aspect_ratio = float(np.mean(aspect_ratios))

    return {
        "h_mean": h_mean, "h_std": h_std,
        "s_mean": s_mean, "s_std": s_std,
        "v_mean": v_mean, "v_std": v_std,
        "l_mean": l_mean, "l_std": l_std,
        "a_mean": a_mean, "a_std": a_std,
        "b_mean": b_mean, "b_std": b_std,
        "roughness": roughness,
        "edge_density": edge_density,
        "specular_share": specular_share,
        "mean_area": mean_area,
        "mean_circularity": mean_circularity,
        "mean_aspect_ratio": mean_aspect_ratio
    }
