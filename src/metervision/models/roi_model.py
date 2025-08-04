from ultralytics import YOLO
import cv2
import numpy as np
import sys
import time

from metervision.logger.logs import logging
from metervision.exception.custom_exception import CustomException

class ROIModel:
    def __init__(self, model_path, params):
        self.params = params
        try:
            start_time = time.perf_counter()
            logging.info("Trying to Load the Model")
            self.model = YOLO(model=model_path, task="obb", verbose=True)
            logging.info(f"ROI Model Loaded Successfully (time {round(time.perf_counter()-start_time, 3)}s)")
        except Exception as e:
            raise CustomException(str(e), e)
        
    def detect_display(self, org_img_arr):
        try:
            start_time = time.perf_counter()
            logging.info("Finding Display Bounding Box")
            results = self.model(org_img_arr)
            logging.info(f"Display Bounding Box Found Successfully (time {round(time.perf_counter()-start_time, 3)}s)")

            for result in results:
                polygon_obb = result.obb.xyxyxyxy.numpy()
                reshaped_ploygon = polygon_obb.reshape((-1, 1, 2)).astype(np.int32)
                bb_img_arr = cv2.polylines(img=org_img_arr, pts=[reshaped_ploygon], isClosed=True, color=(255, 0, 0), thickness=1)
            logging.info("Drawn Rectangle Bounding Box on the Image")

            return bb_img_arr, reshaped_ploygon
        except Exception as e:
            raise CustomException(str(e), sys)

    def extract_display(self, bb_img_arr, bb_cord):
        try:
            logging.info("Trying to Extract Display ROI")
            x_coords = bb_cord[:, 0, 0]
            y_coords = bb_cord[:, 0, 1]
            x1, x2 = x_coords.min(), x_coords.max()
            y1, y2 = y_coords.min(), y_coords.max()

            display_only_img = bb_img_arr[y1:y2, x1:x2]
            logging.info("ROI Extraction Completed Successfully")

            width = self.params.resized_width
            height = self.params.resized_height
            resized_display_img = cv2.resize(display_only_img, (width, height),interpolation=cv2.INTER_LINEAR)
            logging.info("Extracted ROI Resized Completed")

            return resized_display_img
        except Exception as e:
            raise CustomException(str(e), sys)

    def roi_prediction(self, img_arr):
        bb_img_arr, bb_cord = self.detect_display(org_img_arr=img_arr)
        display_arr = self.extract_display(bb_img_arr, bb_cord)

        return bb_img_arr, display_arr
