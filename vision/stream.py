"""Edge IoT & RTSP / Video Stream Processing Utility (vision/stream.py).

Provides frame extraction helpers for processing live RTSP streams from autonomous plant
inspection drones or fixed-mount security cameras, with explicit simulation fallback marking.
"""

import cv2
import numpy as np
from typing import Generator, Optional, Tuple


class StreamProcessor:
    """Handles frame extraction from video files or RTSP streams for edge inspection."""

    def __init__(self, source_uri: str):
        self.source_uri = source_uri
        self.is_simulated = False

    def extract_frames(self, max_frames: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Generator yielding frame index and BGR NumPy frame from the stream source.

        Args:
            max_frames: Optional limit on number of frames to yield.

        Yields:
            Tuple of (frame_index, frame_ndarray).
        """
        cap = cv2.VideoCapture(self.source_uri)
        if not cap.isOpened():
            # If stream cannot be opened directly, mark as simulated fallback
            self.is_simulated = True
            yield 0, np.full((300, 300, 3), 160, dtype=np.uint8)
            return

        self.is_simulated = False
        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame_idx, frame
                frame_idx += 1
                if max_frames and frame_idx >= max_frames:
                    break
        finally:
            cap.release()
