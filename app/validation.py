"""Input Validation Safety Layer for CementSight AI.

Validates industrial equipment images at the application boundary prior to invoking
the OpenCV vision inspection pipeline (vision/inspection.py).

Minimum Dimensions Rationale:
- Real industrial equipment images are 1920x1080.
- Unit and robustness test images are typically 200x200 to 300x300 pixels.
- Empirical testing showed that tiny images (e.g., 32x32) lack sufficient spatial resolution
  and lead to unstable high-corrosion false positives (e.g., 34.86% spurious CRITICAL).
- Therefore, a defensible minimum size threshold of 128x128 pixels is enforced to ensure
  reliable feature extraction and ROI mask segmentation without rejecting valid test inputs.
- Note: This threshold ensures input sanity but does not guarantee inspection accuracy.
"""

import os
import cv2
import numpy as np
from typing import Tuple, Union, Optional

# Minimum dimensions required for reliable industrial equipment inspection
MIN_WIDTH: int = 128
MIN_HEIGHT: int = 128


def validate_image_input(image_path_or_array: Union[str, np.ndarray, None]) -> Tuple[bool, Optional[np.ndarray], str]:
    """Validates an image path or numpy array before inspection.

    Checks:
    1. Input is not None / empty.
    2. If string path, file exists and can be read by cv2.imread.
    3. If numpy array, it is valid and non-empty.
    4. Image dimensions meet or exceed MIN_WIDTH x MIN_HEIGHT (128x128).

    Returns:
        (is_valid, decoded_image_array, error_message)
    """
    if image_path_or_array is None:
        return False, None, "Image input is None or missing."

    img = None
    if isinstance(image_path_or_array, str):
        if not os.path.exists(image_path_or_array):
            return False, None, f"Image path '{image_path_or_array}' does not exist."
        try:
            img = cv2.imread(image_path_or_array)
        except Exception as e:
            return False, None, f"Failed to read image file '{image_path_or_array}': {str(e)}"
        if img is None:
            return False, None, f"Failed to decode image from path '{image_path_or_array}'. File may be corrupt or not a valid image format."
    elif isinstance(image_path_or_array, np.ndarray):
        if image_path_or_array.size == 0:
            return False, None, "Image numpy array is empty."
        img = image_path_or_array.copy()
    else:
        return False, None, f"Invalid image input type: {type(image_path_or_array)}. Expected file path string or numpy array."

    # Check dimensions
    if img.ndim < 2 or img.shape[0] == 0 or img.shape[1] == 0:
        return False, None, "Image has invalid shape or zero dimensions."

    height, width = img.shape[:2]
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return False, None, (
            f"Image dimensions ({width}x{height}) are below the minimum required "
            f"inspection threshold ({MIN_WIDTH}x{MIN_HEIGHT} pixels). Unsuitable for reliable inspection."
        )

    return True, img, ""
