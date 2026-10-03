"""AWS Lambda Adapter for CementSight AI Vision Inspection Engine.

ENVIRONMENT NOTE:
- Local development runs on OpenCV 4.14 because of Android/Termux wheel limitations.
- AWS Lambda deployment runs on OpenCV 5.0.0.93 (via opencv-python-headless).
- The vision algorithm (vision/inspection.py) is 100% shared and frozen between both environments.
"""

import base64
import json
import traceback
import cv2
import numpy as np
from vision.inspection import inspect_corrosion


def lambda_handler(event, context):
    """AWS Lambda entry point for CementSight AI image inspection.

    Supports Lambda Function URL and API Gateway event structures.
    Expects JSON payload with base64-encoded image data under 'image_base64' or 'body'.
    """
    try:
        # 1. Parse request body
        body = event.get("body")
        if body is None:
            # If event itself is the payload dict
            body_data = event
        else:
            if event.get("isBase64Encoded", False):
                try:
                    body_decoded = base64.b64decode(body).decode("utf-8")
                    body_data = json.loads(body_decoded)
                except Exception:
                    body_data = json.loads(body)
            else:
                if isinstance(body, str):
                    try:
                        body_data = json.loads(body)
                    except json.JSONDecodeError:
                        body_data = {"image_base64": body}
                else:
                    body_data = body

        # 2. Extract base64 image string
        image_b64 = None
        if isinstance(body_data, dict):
            image_b64 = body_data.get("image_base64") or body_data.get("image") or body_data.get("data")

        if not image_b64:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "success": False,
                    "error": "Missing image data. Provide base64 image under 'image_base64' key."
                })
            }

        # Handle data URL prefix if present (e.g. data:image/jpeg;base64,...)
        if isinstance(image_b64, str) and "," in image_b64 and "base64" in image_b64[:30]:
            image_b64 = image_b64.split(",")[1]

        # 3. Decode base64 image safely
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as e:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "success": False,
                    "error": f"Invalid base64 encoding: {str(e)}"
                })
            }

        # 4. Convert to numpy array and decode with cv2.imdecode
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None or img.size == 0:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "success": False,
                    "error": "Failed to decode image from bytes. Ensure image is a valid JPEG/PNG."
                })
            }

        # 5. Call existing vision inspection function
        try:
            inspection_report = inspect_corrosion(img, output_annotated_path=None, output_roi_path=None, use_roi=True)
        except Exception as e:
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({
                    "success": False,
                    "error": f"Vision inspection execution error: {str(e)}",
                    "traceback": traceback.format_exc()
                })
            }

        # 6. Construct judge-friendly JSON response
        response_payload = {
            "success": True,
            "engine": {
                "name": "CementSight Vision Engine",
                "opencv_required": "5.0.0",
                "opencv_actual": cv2.__version__
            },
            "inspection": inspection_report
        }

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(response_payload)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "success": False,
                "error": f"Internal server error: {str(e)}",
                "traceback": traceback.format_exc()
            })
        }
