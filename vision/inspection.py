import cv2
import numpy as np


def merge_overlapping_boxes(boxes, iou_thresh=0.05, proximity_thresh=30):
    """Iteratively merges overlapping or close bounding boxes [x, y, w, h] using RETR_EXTERNAL and IoU/proximity."""
    if not boxes:
        return []
    
    def compute_box_iou(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
        bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = (w1 * h1) + (w2 * h2) - inter
        return inter / union if union > 0 else 0.0

    changed = True
    while changed:
        changed = False
        new_boxes = []
        merged_flags = [False] * len(boxes)
        for i in range(len(boxes)):
            if merged_flags[i]:
                continue
            curr = boxes[i]
            x1, y1, w1, h1 = curr
            for j in range(i + 1, len(boxes)):
                if merged_flags[j]:
                    continue
                other = boxes[j]
                x2, y2, w2, h2 = other
                iou = compute_box_iou(curr, other)
                dist_x = abs((x1 + w1/2) - (x2 + w2/2))
                dist_y = abs((y1 + h1/2) - (y2 + h2/2))
                if iou > iou_thresh or (dist_x < (w1 + w2)/2 + proximity_thresh and dist_y < (h1 + h2)/2 + proximity_thresh):
                    nx = min(x1, x2)
                    ny = min(y1, y2)
                    nw = max(x1 + w1, x2 + w2) - nx
                    nh = max(y1 + h1, y2 + h2) - ny
                    curr = [nx, ny, nw, nh]
                    x1, y1, w1, h1 = curr
                    merged_flags[j] = True
                    changed = True
            new_boxes.append(curr)
        boxes = new_boxes
    return [tuple(b) for b in boxes]


def compute_foreground_roi_mask(img, output_roi_path=None):
    """Computes a classical OpenCV foreground ROI mask based on structural edges,

    texture variance (high Laplacian variance outside Canny contours to capture valid rust on lower pipe shafts),
    and morphological grouping. Safeguards against ROI wipeout (ensures adequate coverage).
    """
    height, width = img.shape[:2]
    total_pixels = height * width

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection for structural equipment/pipe edges
    edges = cv2.Canny(blurred, 40, 140)

    # Local texture variance / Laplacian gradient magnitude to capture textured rust/equipment outside Canny edges
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    grad_mag = cv2.convertScaleAbs(lap)
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)
    texture_roi = (grad_norm > 8).astype(np.uint8) * 255

    combined_edges = cv2.bitwise_or(edges, texture_roi)

    # Morphological operations to group structural edges into solid foreground equipment regions
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    closed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, close_kernel)
    dilated = cv2.dilate(closed, dilate_kernel, iterations=2)

    # Find external contours using cv2.RETR_EXTERNAL
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_mask = np.zeros((height, width), dtype=np.uint8)
    min_roi_area = int(total_pixels * 0.0005)  

    roi_components_total = 0
    roi_rejection_audit = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_roi_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi_components_total += 1

        is_frame_spanning = (
            (w >= 0.90 * width and h >= 0.90 * height) or
            (area > 0.80 * total_pixels)
        )
        if is_frame_spanning:
            continue

        cv2.drawContours(roi_mask, [cnt], -1, 255, -1)

    # Safeguard against ROI wipeout: if ROI coverage < 20%, fallback to full frame or dilated mask
    if np.sum(roi_mask > 0) < int(total_pixels * 0.20):
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

    # Expand rust mask to include specular glare / dark brown ochre rough rust / corroded hex heads:
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    grad_mag = cv2.convertScaleAbs(lap)
    
    # Dark brown / ochre rough rust (V between 30 and 130, a* > 126, b* > 130) across all 3 pipes and hex heads
    dark_brown_rust_mask = ((grad_mag > 3) & (a_channel > 126) & (b_channel > 130) & (hsv[:, :, 2] >= 30) & (hsv[:, :, 2] <= 130)).astype(np.uint8) * 255
    specular_glare_rust_mask = ((grad_mag > 6) & ((a_channel > 130) | (b_channel > 135)) & (hsv[:, :, 2] > 180)).astype(np.uint8) * 255
    raw_hsv_mask = cv2.bitwise_or(raw_hsv_mask, dark_brown_rust_mask)
    raw_hsv_mask = cv2.bitwise_or(raw_hsv_mask, specular_glare_rust_mask)

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

    raw_bounding_boxes = []
    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        raw_bounding_boxes.append((int(x), int(y), int(w), int(h)))

    bounding_boxes = merge_overlapping_boxes(raw_bounding_boxes)
    annotated_img = img.copy()

    for (x, y, w, h) in bounding_boxes:
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
