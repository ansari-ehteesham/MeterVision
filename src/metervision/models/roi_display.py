"""
Display ROI model wrapper.

This module wraps a YOLO model (Ultralytics) configured for oriented bounding boxes
(task="obb") to detect the meter display. It exposes methods to detect the display
polygon and to extract a resized display image using postprocessing utilities.
"""

import sys
import time

import numpy as np
from ultralytics import YOLO

from metervision.exception.custom_exception import CustomException
from metervision.logger.logs import logging
from metervision.utils.roi_postprocessing import extract_roi


class DisplayDetector:
    """
    Wrapper for the YOLO model that locates the meter display (oriented bounding box).

    Parameters
    ----------
    model_path : str
        Path to the YOLO model weights / file to load.
    params : object
        Parameter object (from config) containing keys like display_roi_resized.width/height.
    """

    def __init__(self, model_path, params):
        self.params = params
        try:
            start_time = time.perf_counter()
            logging.info("Trying to Load the Model")

            # Use YOLO with 'obb' task for oriented bounding boxes
            self.model = YOLO(model=model_path, task="obb", verbose=True)

            elapsed = round(time.perf_counter() - start_time, 3)
            logging.info(f"Display ROI Model Loaded Successfully (time {elapsed}s)")
        except Exception as e:
            raise CustomException(str(e), e)

    def detect_display(self, image: np.ndarray) -> np.ndarray:
        """
        Run the detector and return a polygon representing the display OBB.

         Parameters
         ----------
         image : ndarray
             Input image array.

         Returns
         -------
         polygon: np.ndarray
             Polygon coordinates (int32) suitable for extract_roi.
        """

        try:
            start_time = time.perf_counter()
            logging.info("Finding Display Bounding Box")
            results = self.model(image)
            elapsed = round(time.perf_counter() - start_time, 3)
            logging.info(f"Display Bounding Box Found Successfully (time {elapsed}s)")

            for result in results:
                if result:
                    # getting highest confident boxes
                    conf_lst = result.obb.conf.tolist()
                    idx_max_conf = conf_lst.index(max(conf_lst))

                    # result.obb.xyxyxyxy likely contains 8 numbers (x1,y1,...,x4,y4)
                    polygon_obb = result.obb.xyxyxyxy[idx_max_conf].numpy()
                    polygon = polygon_obb.reshape((-1, 1, 2)).astype(np.int32)
                else:
                    logging.warning("Display is not Detected")
                    polygon = None

            return polygon
        except Exception as e:
            raise CustomException(str(e), sys)

    def extract_display_roi(self, img: np.ndarray) -> np.ndarray:
        """
        Full flow for display ROI extraction:
        - Detect the display polygon.
        - Extract and resize the ROI using `extract_roi`.

        Parameters
        ----------
        img : ndarray
            Input image array.

        Returns
        -------
        display_image: ndarray
            Resized display ROI image array.
        """
        target_w = self.params.resized_width
        target_h = self.params.resized_height

        polygon = self.detect_display(image=img.copy())
        if polygon.any():
            display_image = extract_roi(img, polygon, (target_w, target_h), "Display")
            return display_image
        else:
            return img
