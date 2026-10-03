import cv2
import numpy as np


def compute_foreground_roi_mask(img, output_roi_path=None):
    """Computes a classical OpenCV foreground ROI mask based on structural edges,

    gradients, and morphological grouping of equipment/pipe boundaries.
    Includes safeguards against full-frame, tiny contours, and low-extent hollow blobs.
    """
    height, width = img.shape[:2]
    total_pixels = height * width

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection for structural equipment/pipe edges
    edges = cv2.Canny(blurred, 50, 150)

    # Morphological operations to group structural edges into solid foreground equipment regions
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)
    dilated = cv2.dilate(closed, dilate_kernel, iterations=1)

    # Find external contours of structural equipment regions
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_mask = np.zeros((height, width), dtype=np.uint8)
    min_roi_area = int(total_pixels * 0.001)  # Reject tiny components (<0.1% of image)

    roi_components_total = 0
    roi_rejection_audit = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_roi_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi_components_total += 1

        # Safeguard: Reject contours that span essentially the entire frame (giant frame rejection)
        is_frame_spanning = (
            (w >= 0.85 * width and h >= 0.85 * height) or
            (area > 0.70 * total_pixels)
        )
        if is_frame_spanning:
            continue

        # Extent / fill-ratio filter: extent = contour_area / bounding_box_area
        bb_area = float(w * h)
        extent = float(area) / bb_area if bb_area > 0 else 0.0

        if extent < 0.15:
            roi_rejection_audit.append({
                "box": (int(x), int(y), int(w), int(h)),
                "extent": round(extent, 3),
                "area": area
            })
            continue

        # Draw valid equipment ROI component onto mask
        cv2.drawContours(roi_mask, [cnt], -1, 255, -1)

    # Fallback for synthetic plain images where edge density might be low
    if np.sum(roi_mask > 0) == 0:
        roi_mask.fill(255)

    if output_roi_path is not None:
        cv2.imwrite(output_roi_path, roi_mask)

    return roi_mask, roi_components_total, roi_rejection_audit


def inspect_corrosion(image_path_or_array, output_annotated_path=None, output_roi_path=None, use_roi=True):
    """Inspects an industrial image for corrosion (rust/staining) using classical OpenCV,

    with optional classical foreground pipe/equipment ROI segmentation and extent filtering.

    Thresholds for recommendation:
    - PASS: corrosion_percentage < 2.0%
    - WARNING: 2.0% <= corrosion_percentage < 10.0%
    - CRITICAL: corrosion_percentage >= 10.0%

    Returns:
        dict containing image dimensions, pixel count, percentage, region count,
        bounding boxes, and recommendation, along with diagnostic mask metrics and ROI audit.
    """
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
        if img is None:
            raise FileNotFoundError(f"Could not load image from {image_path_or_array}")
    else:
        img = image_path_or_array.copy()

    height, width = img.shape[:2]
    total_pixels = height * width

    # Step 1: Compute foreground ROI mask if enabled
    roi_components_total = 0
    roi_rejection_audit = []
    if use_roi:
        roi_mask, roi_components_total, roi_rejection_audit = compute_foreground_roi_mask(img, output_roi_path=output_roi_path)
    else:
        roi_mask = np.full((height, width), 255, dtype=np.uint8)

    foreground_roi_pixels = int(np.sum(roi_mask > 0))
    foreground_roi_coverage_percentage = (foreground_roi_pixels / total_pixels) * 100.0

    # Step 2: Apply gentle Gaussian blur to reduce high-frequency texture noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Convert to HSV color space for robust color segmentation of rust/orange-brown tones
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Define refined color range for rust / corrosion in HSV
    lower_rust = np.array([5, 50, 50], dtype=np.uint8)
    upper_rust = np.array([25, 255, 255], dtype=np.uint8)

    raw_hsv_mask = cv2.inRange(hsv, lower_rust, upper_rust)
    raw_hsv_mask_pixels = int(np.sum(raw_hsv_mask > 0))
    raw_hsv_mask_percentage = (raw_hsv_mask_pixels / total_pixels) * 100.0

    # Step 3: Restrict corrosion detection using foreground ROI mask
    restricted_hsv_mask = cv2.bitwise_and(raw_hsv_mask, roi_mask)
    restricted_mask_pixels = int(np.sum(restricted_hsv_mask > 0))
    restricted_mask_percentage = (restricted_mask_pixels / total_pixels) * 100.0

    # Step 4: Morphological operations on restricted mask to clean noise and bridge fragmented corrosion spots
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    mask = cv2.morphologyEx(restricted_hsv_mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    post_morph_pixels = int(np.sum(mask > 0))
    post_morph_mask_percentage = (post_morph_pixels / total_pixels) * 100.0

    # Step 5: Contour analysis to find distinct corrosion regions and filter tiny false-positive noise
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dynamic minimum contour area based on image dimensions to suppress tiny background artifacts
    min_contour_area = max(100, int(total_pixels * 0.0002))

    valid_contours = []
    filtered_mask = np.zeros_like(mask)
    rejection_audit = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            rejection_audit.append({"reason": "Below min_contour_area", "area": area})
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # REJECTION RULE: Reject giant frame-spanning or near-frame-spanning background contours.
        is_frame_spanning = (
            (w >= 0.85 * width and h >= 0.85 * height) or
            (area > 0.70 * total_pixels)
        )
        if is_frame_spanning:
            rejection_audit.append({"reason": "Frame-spanning giant contour", "area": area, "box": (x, y, w, h)})
            continue

        valid_contours.append(cnt)
        cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

    # Recalculate corrosion pixels and percentage based on valid filtered regions
    corrosion_pixels = int(np.sum(filtered_mask > 0))
    corrosion_percentage = (
        (corrosion_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    )

    bounding_boxes = []
    annotated_img = img.copy()

    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bounding_boxes.append((int(x), int(y), int(w), int(h)))
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    region_count = len(bounding_boxes)

    # Determine recommendation based on documented thresholds
    if corrosion_percentage < 2.0:
        recommendation = "PASS"
    elif corrosion_percentage < 10.0:
        recommendation = "WARNING"
    else:
        recommendation = "CRITICAL"

    if output_annotated_path is not None:
        cv2.imwrite(output_annotated_path, annotated_img)

    return {
        "dimensions": {"width": width, "height": height},
        "total_pixels": total_pixels,
        "raw_hsv_mask_percentage": round(raw_hsv_mask_percentage, 2),
        "foreground_roi_coverage_percentage": round(foreground_roi_coverage_percentage, 2),
        "restricted_mask_percentage": round(restricted_mask_percentage, 2),
        "post_morph_mask_percentage": round(post_morph_mask_percentage, 2),
        "corrosion_pixel_count": corrosion_pixels,
        "corrosion_percentage": round(corrosion_percentage, 2),
        "region_count": region_count,
        "bounding_boxes": bounding_boxes,
        "recommendation": recommendation,
        "roi_components_total": roi_components_total,
        "roi_rejection_audit": roi_rejection_audit,
        "rejection_audit": rejection_audit,
    }
