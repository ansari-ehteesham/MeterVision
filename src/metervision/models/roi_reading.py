import sys
import time

import numpy as np
from ultralytics import YOLO

from metervision.exception.custom_exception import CustomException
from metervision.logger.logs import logging
from metervision.utils.roi_postprocessing import extract_roi


class ReadingROIModel:
    def __init__(self, model_path, params):
        self.params = params
        try:
            start_time = time.perf_counter()
            logging.info("Trying to Load the Model")
            self.model = YOLO(model=model_path, task="detect", verbose=True)
            elapsed = round(time.perf_counter() - start_time, 3)
            logging.info(f"Reading ROI Model Loaded Successfully (time {elapsed}s)")
        except Exception as e:
            raise CustomException(str(e), e)

    def detect_reading(self, img_arr):
        try:
            start_time = time.perf_counter()
            logging.info("Finding Reading Bounding Box")
            results = self.model(img_arr)
            elapsed = round(time.perf_counter() - start_time, 3)
            logging.info(f"Reading Bounding Box Found Successfully (time {elapsed}s)")

            for result in results[0].boxes:
                polygon_obb = result.xyxy.numpy()
                reshaped_ploygon = polygon_obb.reshape((-1, 1, 2)).astype(np.int32)

            return reshaped_ploygon
        except Exception as e:
            raise CustomException(str(e), sys)

    def reading_roi_prediction(self, img_arr):
        width = self.params.reading_roi_resized.width
        height = self.params.reading_roi_resized.height

        bb_cord = self.detect_reading(img_arr=img_arr.copy())
        reading_arr = extract_roi(img_arr, bb_cord, (width, height), "Reading")

        return reading_arr
