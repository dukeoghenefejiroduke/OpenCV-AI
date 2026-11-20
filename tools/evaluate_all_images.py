import os
import cv2
import numpy as np
from vision.corrosion_detector import CorrosionDetector
from vision.inspection import compute_foreground_roi_mask

test_images = [
    "data/challenge_pack/Thin Corrosion2.jpg",
    "data/challenge_pack/joint.jpeg",
    "data/challenge_pack/Orange1.jpg",
    "data/challenge_pack/Thin Corrosion1.jpg"
]

detector = CorrosionDetector()

for img_path in test_images:
    if not os.path.exists(img_path):
        print(f"Skipping {img_path} (not found)")
        continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_mask = f"mask_{base_name}_v6.jpg"

    res = detector.detect(img, output_annotated_path=f"fixed_{base_name}_v6.jpg", output_mask_path=out_mask)
    
    # Compute true mask coverage
    roi_mask, _, _ = compute_foreground_roi_mask(img)
    roi_pixels = int(np.sum(roi_mask > 0))
    if roi_pixels == 0:
        roi_pixels = h * w

    # Re-compute mask pixels using v6 dual HSV + CLAHE logic
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_c, a_c, b_c = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l_c)
    norm_img = cv2.cvtColor(cv2.merge((cl, a_c, b_c)), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(norm_img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask_brown = cv2.inRange(hsv, np.array([5, 30, 30], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8))
    mask_dark = cv2.inRange(hsv, np.array([0, 5, 5], dtype=np.uint8), np.array([25, 140, 130], dtype=np.uint8))
    raw_mask = cv2.bitwise_or(mask_brown, mask_dark)
    restricted = cv2.bitwise_and(raw_mask, roi_mask)
    cleaned = cv2.morphologyEx(restricted, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    mask_pixels = int(np.sum(cleaned > 0))
    coverage = (mask_pixels / roi_pixels) * 100.0

    # Save visualization overlay with red mask and thin boxes
    overlay = img.copy()
    overlay[cleaned > 0] = (0, 0, 255)
    mask_vis = cv2.addWeighted(overlay, 0.40, img, 0.60, 0)

    for f in res.get("findings", []):
        bx, by, bw, bh = f["bbox"]
        rid = f["region_id"]
        cv2.rectangle(mask_vis, (bx, by), (bx + bw, by + bh), (0, 255, 255), 1)
        cv2.putText(mask_vis, rid, (bx, max(12, by - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imwrite(out_mask, mask_vis)
    file_size = os.path.getsize(out_mask)

    print(f"Image: {img_path}")
    print(f"  - Status: {res.get('execution_status')}")
    print(f"  - Findings: {len(res.get('findings', []))}")
    print(f"  - Mask Pixels: {mask_pixels}")
    print(f"  - ROI Pixels: {roi_pixels}")
    print(f"  - True Mask Coverage: {coverage:.2f}%")
    print(f"  - Saved Mask: {out_mask} ({file_size} bytes)\n")
