import os
import time
import cv2
import numpy as np
from vision.corrosion_detector import CorrosionDetector

image_path = "data/challenge_pack/joint.jpeg"
if not os.path.exists(image_path):
    image_path = "data/challenge_pack/Orange1.jpg"

print(f"Testing v6 inspection on test image: {image_path}")
orig_size = os.path.getsize(image_path)

img = cv2.imread(image_path)
h, w = img.shape[:2]

# Sample 20 pixels around center to inspect HSV values
hsv_full = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
center_patch = hsv_full[h//3:2*h//3, w//3:2*w//3]
pixels_flat = center_patch.reshape(-1, 3)
sampled_hsv = pixels_flat[::max(1, len(pixels_flat)//20)]
print("Sampled HSV values from center region:")
for s in sampled_hsv[:10]:
    print(f"  H: {s[0]}, S: {s[1]}, V: {s[2]}")

# Run CorrosionDetector v6 and save mask overlay
out_mask_path = "mask_flange_v6.jpg"
detector = CorrosionDetector()
res = detector.detect(img, output_annotated_path="fixed_flange_v6.jpg", output_mask_path=out_mask_path)

# Generate explicit mask visualization with red overlay
from vision.inspection import compute_foreground_roi_mask
roi_mask, _, _ = compute_foreground_roi_mask(img)
roi_pixels = int(np.sum(roi_mask > 0))

# Dual HSV + CLAHE processing for mask overlay
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_c, a_c, b_c = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
cl = clahe.apply(l_c)
norm_img = cv2.cvtColor(cv2.merge((cl, a_c, b_c)), cv2.COLOR_LAB2BGR)
blurred = cv2.GaussianBlur(norm_img, (5, 5), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

lower_brown = np.array([5, 30, 30], dtype=np.uint8)
upper_brown = np.array([25, 255, 255], dtype=np.uint8)
mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

lower_dark = np.array([0, 5, 5], dtype=np.uint8)
upper_dark = np.array([25, 140, 130], dtype=np.uint8)
mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

raw_mask = cv2.bitwise_or(mask_brown, mask_dark)
restricted = cv2.bitwise_and(raw_mask, roi_mask if roi_pixels > 0 else np.full((h, w), 255, dtype=np.uint8))
cleaned = cv2.morphologyEx(restricted, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

mask_pixels = int(np.sum(cleaned > 0))
coverage = (mask_pixels / roi_pixels * 100.0) if roi_pixels > 0 else (mask_pixels / (h*w) * 100.0)

# Save mask visualization with red overlay (40% transparent red)
overlay = img.copy()
overlay[cleaned > 0] = (0, 0, 255) # Pure Red
mask_vis = cv2.addWeighted(overlay, 0.40, img, 0.60, 0)

findings = res.get("findings", [])
for f in findings:
    bx, by, bw, bh = f["bbox"]
    rid = f["region_id"]
    cv2.rectangle(mask_vis, (bx, by), (bx + bw, by + bh), (0, 255, 255), 1)
    cv2.putText(mask_vis, rid, (bx, max(12, by - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

cv2.imwrite(out_mask_path, mask_vis)

# Verify file size and timestamp
if os.path.exists(out_mask_path):
    mask_size = os.path.getsize(out_mask_path)
    mask_mtime = time.ctime(os.path.getmtime(out_mask_path))
    print(f"\n--- Output File Verification ---")
    print(f"Original image size: {orig_size} bytes")
    print(f"Generated {out_mask_path} size: {mask_size} bytes")
    print(f"Timestamp: {mask_mtime}")
    print(f"Execution Status: {res.get('execution_status')}")
    print(f"Findings Count: {len(findings)}")
    print(f"Mask Pixels: {mask_pixels}")
    print(f"ROI Pixels: {roi_pixels}")
    print(f"True Mask Coverage = {coverage:.2f}%")
else:
    print(f"Error: {out_mask_path} was not created.")
