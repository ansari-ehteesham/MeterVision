import sys
import time

import numpy as np
from ultralytics import YOLO

from metervision.exception.custom_exception import CustomException
from metervision.logger.logs import logging
from metervision.utils.roi_postprocessing import extract_roi


class DisplayROIModel:
    def __init__(self, model_path, params):
        self.params = params
        try:
            start_time = time.perf_counter()
            logging.info("Trying to Load the Model")
            self.model = YOLO(model=model_path, task="obb", verbose=True)
            elapsed = round(time.perf_counter() - start_time, 3)
            logging.info(f"Display ROI Model Loaded Successfully (time {elapsed}s)")
        except Exception as e:
            raise CustomException(str(e), e)

    def detect_display(self, org_img_arr):
        try:
            start_time = time.perf_counter()
            logging.info("Finding Display Bounding Box")
            results = self.model(org_img_arr)
            elapsed = round(time.perf_counter() - start_time, 3)
            logging.info(f"Display Bounding Box Found Successfully (time {elapsed}s)")

            for result in results:
                polygon_obb = result.obb.xyxyxyxy.numpy()
                reshaped_ploygon = polygon_obb.reshape((-1, 1, 2)).astype(np.int32)

            return reshaped_ploygon
        except Exception as e:
            raise CustomException(str(e), sys)

    def display_roi_prediction(self, img_arr):
        width = self.params.display_roi_resized.width
        height = self.params.display_roi_resized.height

        bb_cord = self.detect_display(org_img_arr=img_arr.copy())
        display_arr = extract_roi(img_arr, bb_cord, (width, height), "Display")

        return display_arr
