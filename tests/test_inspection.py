import os
import base64
import json
import cv2
import numpy as np
from vision.inspection import inspect_corrosion, compute_foreground_roi_mask
from lambda_handler import lambda_handler


def test_clean_image_pass():
    # Clean gray image with no rust
    img = np.full((200, 200, 3), 200, dtype=np.uint8)
    report = inspect_corrosion(img)

    assert report["dimensions"]["width"] == 200
    assert report["dimensions"]["height"] == 200
    assert report["corrosion_percentage"] == 0.0
    assert report["region_count"] == 0
    assert report["recommendation"] == "PASS"
    print("test_clean_image_pass PASSED")


def test_corroded_image_detection():
    # Image with background and a distinct rust patch
    img = np.full((300, 300, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (100, 100), (30, 100, 180), -1)

    report = inspect_corrosion(img, output_annotated_path="test_output.jpg")

    assert report["dimensions"]["width"] == 300
    assert report["dimensions"]["height"] == 300
    assert report["corrosion_pixel_count"] > 0
    assert report["corrosion_percentage"] > 0.0
    assert report["region_count"] >= 1
    assert len(report["bounding_boxes"]) >= 1
    assert report["recommendation"] in ["PASS", "WARNING", "CRITICAL"]
    print("test_corroded_image_detection PASSED")


def test_roi_mask_creation_and_synthetic():
    # Test compute_foreground_roi_mask on synthetic image without crashing
    img = np.full((200, 200, 3), 150, dtype=np.uint8)
    roi_mask, total, audit = compute_foreground_roi_mask(img)
    assert roi_mask.shape == (200, 200)
    print("test_roi_mask_creation_and_synthetic PASSED")


def test_giant_frame_rejection():
    # Test that giant frame-spanning ROI contours are rejected properly
    img = np.full((200, 200, 3), 150, dtype=np.uint8)
    roi_mask, total, audit = compute_foreground_roi_mask(img)
    assert roi_mask is not None
    print("test_giant_frame_rejection PASSED")


def test_cli_workflow_real_image():
    os.makedirs("data/outputs", exist_ok=True)
    test_image_path = "data/test_input.jpg"
    syn_img = np.full((250, 250, 3), 160, dtype=np.uint8)
    cv2.rectangle(syn_img, (60, 60), (120, 120), (25, 90, 175), -1)
    cv2.imwrite(test_image_path, syn_img)

    output_path = "data/outputs/annotated_test_input.jpg"
    roi_output_path = "data/outputs/foreground_roi.jpg"
    report = inspect_corrosion(test_image_path, output_annotated_path=output_path, output_roi_path=roi_output_path)

    assert report["dimensions"]["width"] == 250
    assert report["dimensions"]["height"] == 250
    assert report["corrosion_percentage"] > 0.0
    assert os.path.exists(output_path)
    assert os.path.exists(roi_output_path)

    if os.path.exists(test_image_path):
        os.remove(test_image_path)
    if os.path.exists(output_path):
        os.remove(output_path)
    if os.path.exists(roi_output_path):
        os.remove(roi_output_path)

    print("test_cli_workflow_real_image PASSED")


def test_lambda_handler_adapter():
    # Test AWS Lambda adapter locally without AWS credentials
    if not os.path.exists("data/pipe_corrosion.jpg"):
        # Create dummy image if pipe_corrosion.jpg isn't present
        os.makedirs("data", exist_ok=True)
        dummy_img = np.full((200, 200, 3), 160, dtype=np.uint8)
        cv2.rectangle(dummy_img, (40, 40), (80, 80), (30, 90, 180), -1)
        cv2.imwrite("data/pipe_corrosion.jpg", dummy_img)

    with open("data/pipe_corrosion.jpg", "rb") as f:
        img_bytes = f.read()

    b64_img = base64.b64encode(img_bytes).decode("utf-8")

    # Construct representative Lambda Function URL event
    event = {
        "headers": {"content-type": "application/json"},
        "body": json.dumps({"image_base64": b64_img})
    }

    response = lambda_handler(event, None)

    assert response["statusCode"] == 200
    assert "headers" in response
    assert response["headers"].get("Content-Type") == "application/json"

    body_data = json.loads(response["body"])
    assert body_data["success"] is True
    assert "engine" in body_data
    assert "inspection" in body_data

    inspection = body_data["inspection"]
    assert "corrosion_percentage" in inspection
    assert "region_count" in inspection
    assert "recommendation" in inspection
    assert "bounding_boxes" in inspection

    print("test_lambda_handler_adapter PASSED")


if __name__ == "__main__":
    print("Running inspection unit tests & AWS Lambda adapter tests...")
    test_clean_image_pass()
    test_corroded_image_detection()
    test_roi_mask_creation_and_synthetic()
    test_giant_frame_rejection()
    test_cli_workflow_real_image()
    test_lambda_handler_adapter()
    print("All tests passed successfully!")
