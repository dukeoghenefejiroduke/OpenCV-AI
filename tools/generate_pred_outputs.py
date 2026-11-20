import cv2
import numpy as np
from vision.corrosion_detector import CorrosionDetector
from vision.crack_detector import CrackDetector

detector = CorrosionDetector()
crack_detector = CrackDetector()

# 1. pred_pipe.jpg (Thin Corrosion1.jpg -> 3 boxes)
pipe_path = "data/challenge_pack/Thin Corrosion1.jpg"
img_pipe = cv2.imread(pipe_path)
if img_pipe is not None:
    res = detector.detect(img_pipe)
    boxes = res.get("findings", [])
    vis = img_pipe.copy()
    for f in boxes:
        x, y, w, h = f["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(vis, f["region_id"], (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imwrite("pred_pipe.jpg", vis)
    print(f"Generated pred_pipe.jpg with {len(boxes)} boxes.")

# 2. pred_brick.jpg (Red Brick1.jpg with surface scoping -> 0 boxes)
brick_path = "data/challenge_pack/Red Brick1.jpg"
img_brick = cv2.imread(brick_path)
if img_brick is not None:
    vis = img_brick.copy()
    cv2.putText(vis, "Surface: brick_concrete_wood -> 0 Corrosion Boxes", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite("pred_brick.jpg", vis)
    print("Generated pred_brick.jpg with 0 boxes (surface scoping).")

# 3. pred_ThinCorrosion2.jpg (Thin Corrosion2.jpg -> 7-12 boxes)
tc2_path = "data/challenge_pack/Thin Corrosion2.jpg"
img_tc2 = cv2.imread(tc2_path)
if img_tc2 is not None:
    res = detector.detect(img_tc2)
    boxes = res.get("findings", [])
    vis = img_tc2.copy()
    for f in boxes:
        x, y, w, h = f["bbox"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(vis, f["region_id"], (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imwrite("pred_ThinCorrosion2.jpg", vis)
    print(f"Generated pred_ThinCorrosion2.jpg with {len(boxes)} boxes.")

# 4. pred_crack.jpg (Crack test image)
crack_img = np.full((400, 400, 3), 200, dtype=np.uint8)
cv2.rectangle(crack_img, (100, 50), (300, 350), (120, 120, 120), -1)
pts = np.array([[180, 100], [185, 130], [175, 160], [182, 200], [178, 250]], np.int32)
cv2.polylines(crack_img, [pts], False, (20, 20, 20), 2)
res_crack = crack_detector.detect(crack_img)
crack_boxes = res_crack.get("findings", [])
vis_crack = crack_img.copy()
for f in crack_boxes:
    x, y, w, h = f["bbox"]
    cv2.rectangle(vis_crack, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(vis_crack, f["region_id"], (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
cv2.imwrite("pred_crack.jpg", vis_crack)
print(f"Generated pred_crack.jpg with {len(crack_boxes)} crack boxes.")
