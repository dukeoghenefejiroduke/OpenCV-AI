"""Optional Diagnostic Visual Processing Explorer Helper (vision/diagnostics.py).

Reuses the exact algorithm and parameters from vision/inspection.py to generate and save
intermediate visual processing stage outputs (blurred image, raw HSV mask, ROI mask,
restricted mask, post-morphology mask, and final annotated image) without altering or
modifying the frozen production inspection pipeline.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Union

from vision.inspection import inspect_corrosion, compute_foreground_roi_mask


def generate_diagnostic_stages(image_path_or_array: Union[str, np.ndarray], output_dir: str = "data/outputs/diagnostics") -> List[Dict[str, Any]]:
    """Generates and saves intermediate stage images for the Visual Processing Explorer.

    Replicates the exact pipeline steps from vision/inspection.py to produce:
    1. Original Input Image
    2. Gaussian-Blurred Image
    3. Raw HSV Corrosion Mask
    4. Structural Foreground ROI Mask
    5. ROI-Restricted Corrosion Mask
    6. Post-Morphology Cleaned Mask
    7. Final Annotated Image with Bounding Boxes

    Args:
        image_path_or_array: Path string or BGR numpy ndarray.
        output_dir: Dedicated safe output directory for saving diagnostic images.

    Returns:
        List of dicts containing stage ID, step, label, description, file path / URL, and result type.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load image safely
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
        if img is None:
            raise FileNotFoundError(f"Could not load image from {image_path_or_array}")
    else:
        img = image_path_or_array.copy()

    height, width = img.shape[:2]

    # 1. Original Image
    orig_path = os.path.join(output_dir, "stage_01_original.jpg")
    cv2.imwrite(orig_path, img)

    # 2. Gaussian Blurred Image
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    blurred_path = os.path.join(output_dir, "stage_02_gaussian_blurred.jpg")
    cv2.imwrite(blurred_path, blurred)

    # 3. Raw HSV Corrosion Mask
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_rust = np.array([5, 50, 50], dtype=np.uint8)
    upper_rust = np.array([25, 255, 255], dtype=np.uint8)
    raw_hsv_mask = cv2.inRange(hsv, lower_rust, upper_rust)
    raw_mask_path = os.path.join(output_dir, "stage_03_raw_hsv_mask.jpg")
    cv2.imwrite(raw_mask_path, raw_hsv_mask)

    # 4. Structural Foreground ROI Mask
    roi_mask, _, _ = compute_foreground_roi_mask(img)
    roi_path = os.path.join(output_dir, "stage_04_foreground_roi_mask.jpg")
    cv2.imwrite(roi_path, roi_mask)

    # 5. ROI-Restricted Corrosion Mask
    restricted_mask = cv2.bitwise_and(raw_hsv_mask, roi_mask)
    restricted_path = os.path.join(output_dir, "stage_05_restricted_mask.jpg")
    cv2.imwrite(restricted_path, restricted_mask)

    # 6. Post-Morphology Cleaned Mask
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    morph_mask = cv2.morphologyEx(restricted_mask, cv2.MORPH_OPEN, open_kernel)
    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, close_kernel)
    morph_path = os.path.join(output_dir, "stage_06_post_morph_mask.jpg")
    cv2.imwrite(morph_path, morph_mask)

    # 7. Final Annotated Image (Run standard inspect_corrosion to ensure exact parity)
    annotated_path = os.path.join(output_dir, "stage_07_final_annotated.jpg")
    report = inspect_corrosion(img, output_annotated_path=annotated_path, use_roi=True)

    stages = [
        {
            "stage_id": "original",
            "step": 1,
            "label": "Original Input Image",
            "description": "Raw ingested BGR equipment image prior to processing.",
            "image_url": orig_path,
            "result_type": "color_image"
        },
        {
            "stage_id": "blurred",
            "step": 2,
            "label": "Gaussian-Blurred Image",
            "description": "Smoothed image using 5x5 Gaussian blur to suppress high-frequency texture noise.",
            "image_url": blurred_path,
            "result_type": "color_image"
        },
        {
            "stage_id": "raw_hsv_mask",
            "step": 3,
            "label": "Raw HSV Corrosion Mask",
            "description": "Initial color segmentation mask identifying rust and orange-brown tones in HSV space.",
            "image_url": raw_mask_path,
            "result_type": "binary_mask"
        },
        {
            "stage_id": "roi_mask",
            "step": 4,
            "label": "Structural Foreground ROI Mask",
            "description": "Classical Canny edge & morphological grouping mask isolating structural equipment boundaries.",
            "image_url": roi_path,
            "result_type": "binary_mask"
        },
        {
            "stage_id": "restricted_mask",
            "step": 5,
            "label": "ROI-Restricted Corrosion Mask",
            "description": "Intersection of raw HSV rust mask and structural foreground ROI mask.",
            "image_url": restricted_path,
            "result_type": "binary_mask"
        },
        {
            "stage_id": "post_morph_mask",
            "step": 6,
            "label": "Post-Morphology Cleaned Mask",
            "description": "Mask cleaned via morphological opening and closing to bridge fragmented corrosion spots.",
            "image_url": morph_path,
            "result_type": "binary_mask"
        },
        {
            "stage_id": "annotated",
            "step": 7,
            "label": "Final Annotated Output",
            "description": "Final inspection output with detected corrosion bounding boxes and severity rating.",
            "image_url": annotated_path,
            "result_type": "annotated_image"
        }
    ]

    return stages
