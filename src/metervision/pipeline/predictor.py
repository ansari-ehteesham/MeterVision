from metervision.constants import DIR_CONFIG_FILE, PARAMS_CONFIG_FILE
from metervision.models.roi_display import DisplayROIModel
from metervision.models.roi_reading import ReadingROIModel


class PredictionPipeline:
    def __init__(self):
        self.dir_config = DIR_CONFIG_FILE
        self.params_config = PARAMS_CONFIG_FILE

        self.display_roi = DisplayROIModel(
            model_path=self.dir_config.display_roi_model,
            params=self.params_config.display_roi_resized,
        )

        self.reading_roi = ReadingROIModel(
            model_path=self.dir_config.reading_roi_model,
            params=self.params_config.reading_roi_resized,
        )

    def prediction(self, img_array):
        # ROI Prediction
        display_arr = self.display_roi.display_roi_prediction(img_arr=img_array)
        reading_arr = self.reading_roi.reading_roi_prediction(img_arr=display_arr)

        return display_arr, reading_arr
