import os
import cv2
import numpy as np
from vision.inspection import inspect_corrosion, compute_foreground_roi_mask

path = "data/challenge_pack/Thin Corrosion2.jpg"
img = cv2.imread(path)
height, width = img.shape[:2]

report = inspect_corrosion(img)
roi_mask, _, _ = compute_foreground_roi_mask(img)
roi_pixels = int(np.sum(roi_mask > 0))

# Get regions / valid contours
blurred = cv2.GaussianBlur(img, (5, 5), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
lower_rust = np.array([5, 50, 50], dtype=np.uint8)
upper_rust = np.array([25, 255, 255], dtype=np.uint8)
raw_hsv = cv2.inRange(hsv, lower_rust, upper_rust)
restricted = cv2.bitwise_and(raw_hsv, roi_mask)
open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
mask = cv2.morphologyEx(restricted, cv2.MORPH_OPEN, open_kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_contour_area = max(100, int((height * width) * 0.0002))

region_data = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < min_contour_area:
        continue
    x, y, w, h = cv2.boundingRect(cnt)
    if (w >= 0.85 * width and h >= 0.85 * height) or (area > 0.70 * (height * width)):
        continue
    # Exact mask pixels for this region
    rmask = np.zeros_like(mask)
    cv2.drawContours(rmask, [cnt], -1, 255, -1)
    mpixels = int(np.sum(rmask > 0))
    region_data.append({"bbox": [x, y, w, h], "mask_pixels": mpixels, "cnt": cnt})

sorted_regions = sorted(region_data, key=lambda r: r["mask_pixels"], reverse=True)[:8]
top8_mask_pixels = sum(r["mask_pixels"] for r in sorted_regions)
top8_coverage = (top8_mask_pixels / roi_pixels) * 100.0

print(f"Top 8 Regions Mask Pixels Sum: {top8_mask_pixels}")
print(f"Pipe ROI Pixels: {roi_pixels}")
print(f"Top 8 True Mask Coverage = {top8_coverage:.2f}%")

# Generate mask_Thin_Corrosion2.jpg with red 40% transparent overlay
filtered_mask = np.zeros_like(mask)
for r in sorted_regions:
    cv2.drawContours(filtered_mask, [r["cnt"]], -1, 255, -1)

overlay = img.copy()
overlay[filtered_mask > 0] = (0, 0, 255) # Red overlay
mask_vis = cv2.addWeighted(overlay, 0.40, img, 0.60, 0)

for idx, r in enumerate(sorted_regions):
    x, y, w, h = r["bbox"]
    reg_id = f"COR-{idx+1:03d}"
    cv2.rectangle(mask_vis, (x, y), (x + w, y + h), (0, 255, 255), 1)
    cv2.putText(mask_vis, reg_id, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

cv2.imwrite("mask_Thin_Corrosion2.jpg", mask_vis)
print("Saved mask_Thin_Corrosion2.jpg successfully.")
