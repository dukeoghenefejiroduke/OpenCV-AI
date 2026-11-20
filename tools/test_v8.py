import cv2
import numpy as np
from vision.corrosion_detector import CorrosionDetector

detector = CorrosionDetector()

# Test Image 1: Rusty Pipe (Thin Corrosion2.jpg)
pipe_path = "data/challenge_pack/Thin Corrosion2.jpg"
img_pipe = cv2.imread(pipe_path)
if img_pipe is not None:
    res_pipe = detector.detect(img_pipe)
    boxes_pipe = res_pipe.get("findings", [])
    print(f"=== TEST A: Rusty Pipe ({pipe_path}) ===")
    print(f"  Status: {res_pipe.get('execution_status')}")
    print(f"  Boxes Count: {len(boxes_pipe)}")
    for f in boxes_pipe:
        print(f"    - {f['region_id']} bbox: {f['bbox']} confidence: {f['confidence']}")
    
    # Save annotated v8_pipe.jpg
    vis_pipe = img_pipe.copy()
    for f in boxes_pipe:
        x, y, w, h = f["bbox"]
        cv2.rectangle(vis_pipe, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(vis_pipe, f["region_id"], (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imwrite("v8_pipe.jpg", vis_pipe)
    print("  Saved v8_pipe.jpg\n")
else:
    print(f"Could not load {pipe_path}")

# Test Image 2: Brick Wall (Red Brick1.jpg)
brick_path = "data/challenge_pack/Red Brick1.jpg"
img_brick = cv2.imread(brick_path)
if img_brick is not None:
    res_brick = detector.detect(img_brick)
    boxes_brick = res_brick.get("findings", [])
    print(f"=== TEST B: Brick Wall ({brick_path}) ===")
    print(f"  Status: {res_brick.get('execution_status')}")
    print(f"  Boxes Count: {len(boxes_brick)}")
    for f in boxes_brick:
        print(f"    - {f['region_id']} bbox: {f['bbox']} confidence: {f['confidence']}")
    
    # Save annotated v8_brick.jpg
    vis_brick = img_brick.copy()
    for f in boxes_brick:
        x, y, w, h = f["bbox"]
        cv2.rectangle(vis_brick, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(vis_brick, f["region_id"], (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imwrite("v8_brick.jpg", vis_brick)
    print("  Saved v8_brick.jpg\n")
else:
    print(f"Could not load {brick_path}")
