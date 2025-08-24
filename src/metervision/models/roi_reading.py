"""
Reading ROI model wrapper.

This module wraps a YOLO model (Ultralytics) configured for axis-aligned detection
(task="detect") to locate the meter reading within the display ROI. It extracts and
resizes the reading ROI for downstream OCR/recognition.
"""

import sys
import time

import numpy as np
from ultralytics import YOLO

from metervision.exception.custom_exception import CustomException
from metervision.logger.logs import logging
from metervision.utils.roi_postprocessing import extract_roi


class ReadingDetector:
    """
    Wrapper for the YOLO model that locates the meter reading region.

    Parameters
    ----------
    model_path : str
        Path to the YOLO model weights / file to load.
    params : object
        Parameter object (from config) containing keys like reading_roi_resized.width/height.
    """

    def __init__(self, model_path, params):
        self.params = params
        try:
            start_time = time.perf_counter()
            logging.info("Trying to Load the Model")

            # Use simple detect task for axis-aligned bounding boxes for readings
            self.model = YOLO(model=model_path, task="detect", verbose=True)

            elapsed = round(time.perf_counter() - start_time, 3)
            logging.info(f"Reading ROI Model Loaded Successfully (time {elapsed}s)")
        except Exception as e:
            raise CustomException(str(e), e)

    def detect_reading(self, image: np.ndarray) -> np.ndarray:
        """
        Run detection on the given image and return a polygon/box for the reading.

        Parameters
        ----------
        img_arr : ndarray
            Input image array.

        Returns
        -------
        np.ndarray
            box coordinates (int32) suitable for extract_roi.
        """

        try:
            start_time = time.perf_counter()
            logging.info("Finding Reading Bounding Box")
            results = self.model(image)
            elapsed = round(time.perf_counter() - start_time, 3)
            logging.info(f"Reading Bounding Box Found Successfully (time {elapsed}s)")

            # Results[0].boxes yields boxes for first (and often only) image passed.
            # We iterate and reshape to the format expected by extract_roi.
            for result in results[0].boxes:
                polygon_box = result.xyxy.numpy()  # axis-aligned box coords
                polygon = polygon_box.reshape((-1, 1, 2)).astype(np.int32)

            return polygon
        except Exception as exc:
            raise CustomException(str(exc), sys)

    def extract_reading_roi(self, img: np.ndarray) -> np.ndarray:
        """
        Detect and extract the reading ROI.

        Parameters
        ----------
        img : ndarray
           Input image array (typically the display ROI).

        Returns
        -------
        reading_image: ndarray
            Resized reading ROI image array.
        """

        target_w = self.params.resized_width
        target_h = self.params.resized_height

        polygon = self.detect_reading(image=img.copy())
        reading_image = extract_roi(img, polygon, (target_w, target_h), "Reading")

        return reading_image
