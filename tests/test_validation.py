"""Unit and Integration Tests for CementSight Input Validation Safety Layer (app/validation.py)."""

import os
import base64
import json
import cv2
import numpy as np

from app.validation import validate_image_input, MIN_WIDTH, MIN_HEIGHT
from vision.inspection import inspect_corrosion
from agent.decision import evaluate_inspection
from lambda_handler import lambda_handler


def test_valid_real_image():
    # A. Valid real image: data/pipe_corrosion.jpg
    image_path = "data/pipe_corrosion.jpg"
    if not os.path.exists(image_path):
        os.makedirs("data", exist_ok=True)
        dummy = np.full((300, 300, 3), 160, dtype=np.uint8)
        cv2.imwrite(image_path, dummy)

    is_valid, img, err = validate_image_input(image_path)
    assert is_valid is True
    assert img is not None
    assert err == ""

    # Run inspection and agent decision
    report = inspect_corrosion(img)
    decision = evaluate_inspection(report)
    assert report["recommendation"] == "WARNING"
    assert decision.decision == "WARNING"
    print("test_valid_real_image PASSED")


def test_tiny_image_rejected():
    # B. Tiny image (32x32): rejected before inspection
    tiny_img = np.full((32, 32, 3), 150, dtype=np.uint8)
    is_valid, img, err = validate_image_input(tiny_img)
    assert is_valid is False
    assert img is None
    assert "below the minimum required inspection threshold" in err
    print("test_tiny_image_rejected PASSED")


def test_acceptable_medium_image():
    # C. Acceptable small-but-adequate image around threshold (e.g. 200x200 >= 128x128)
    med_img = np.full((200, 200, 3), 160, dtype=np.uint8)
    is_valid, img, err = validate_image_input(med_img)
    assert is_valid is True
    assert img is not None
    assert img.shape[:2] == (200, 200)
    print("test_acceptable_medium_image PASSED")


def test_invalid_corrupt_image():
    # D. Invalid/corrupt image data
    invalid_path = "data/corrupt_test_image.jpg"
    with open(invalid_path, "wb") as f:
        f.write(b"not an image file bytes")

    is_valid, img, err = validate_image_input(invalid_path)
    assert is_valid is False
    assert img is None
    assert "Failed to decode image" in err

    if os.path.exists(invalid_path):
        os.remove(invalid_path)
    print("test_invalid_corrupt_image PASSED")


def test_missing_image_path():
    # E. Missing image path
    missing_path = "data/non_existent_image_xyz_12345.jpg"
    is_valid, img, err = validate_image_input(missing_path)
    assert is_valid is False
    assert img is None
    assert "does not exist" in err
    print("test_missing_image_path PASSED")


def test_lambda_invalid_image():
    # F. Lambda invalid image -> controlled 400 response
    tiny_img = np.full((32, 32, 3), 150, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", tiny_img)
    b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

    event = {
        "headers": {"content-type": "application/json"},
        "body": json.dumps({"image_base64": b64})
    }
    response = lambda_handler(event, None)
    assert response["statusCode"] == 400
    body = json.loads(response["body"])
    assert body["success"] is False
    assert "Image validation failed" in body["error"]
    print("test_lambda_invalid_image PASSED")


def test_lambda_valid_real_image():
    # G. Lambda valid real image -> successful inspection (200 OK)
    image_path = "data/pipe_corrosion.jpg"
    if not os.path.exists(image_path):
        os.makedirs("data", exist_ok=True)
        dummy = np.full((300, 300, 3), 160, dtype=np.uint8)
        cv2.imwrite(image_path, dummy)

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    event = {
        "headers": {"content-type": "application/json"},
        "body": json.dumps({"image_base64": b64})
    }
    response = lambda_handler(event, None)
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["success"] is True
    assert "inspection" in body
    print("test_lambda_valid_real_image PASSED")


if __name__ == "__main__":
    print("Running Input Validation Safety Layer Tests...")
    test_valid_real_image()
    test_tiny_image_rejected()
    test_acceptable_medium_image()
    test_invalid_corrupt_image()
    test_missing_image_path()
    test_lambda_invalid_image()
    test_lambda_valid_real_image()
    print("All input validation tests passed successfully!")
