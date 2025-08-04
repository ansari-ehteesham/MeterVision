import streamlit as st
import sys
import time
# Own Module
from metervision.logger.logs import logging
from metervision.exception.custom_exception import CustomException
from metervision.constant import (DIR_CONFIG_FILE)
from metervision.utils.common import (read_img)
from metervision.pipeline.prediction import PredictionPipeline

# loading the environement variable
from dotenv import load_dotenv

try:
    if "env_loaded" not in st.session_state:
        load_dotenv()
        st.session_state["env_loaded"] = True
        logging.info("Environment Variables are Loaded Successfully")
    else:
        pass
except Exception as e:
    raise CustomException(str(e), sys)


# Initializing the Session State

# ------------------------------------------------------------------------------------------------------------------

# setting the page logo and title
st.set_page_config(
    page_title="MeterVision",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Loading the CSS File
try:
    with open(DIR_CONFIG_FILE.sidebar_css) as f:
        css = f.read()
    st.markdown(f"<style> {css} </style>", unsafe_allow_html=True)
except Exception as e:
    raise CustomException(str(e), sys)

# Loading all model once
@st.cache_resource
def models_loading():
    pred_pipeline = PredictionPipeline()
    return pred_pipeline
prediction_pipeline =  models_loading()
# ------------------------------------------------------------------------------------------------------------------


# Home Page
def home_page() -> None:
    with st.container():
        st.markdown("<div class='main-header'>⚡ MeterVision Dashboard</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>A Vision to Power the Electric Meter</div>", unsafe_allow_html=True)
    return None
# ------------------------------------------------------------------------------------------------------------------


# Bulk Data Uplaoder
def bulk_data_page() -> None:
    st.title("Bulk Data")
    return None
# ------------------------------------------------------------------------------------------------------------------

# Model Prediction
def prediction_page() -> None:
    st.title("Prediction Page")
    pred_upload = st.file_uploader( 
        label="Select the Image",
        type=['jpg', 'jpeg', 'png']
    )
    if pred_upload:
        logging.info(f"Meter Image has been Uploaded")

        if st.button("Start Prediction", type="primary"):
            logging.info("Prediction has been Started....")
            with st.spinner("Wait For It...", show_time=True):
                img_arr = read_img(pred_upload)
                bb_img_arr, display_arr = prediction_pipeline.prediction(img_array=img_arr)
            with st.container():
                img_col, roi_cols = st.columns(spec=2, gap="small", border=True)

                if bb_img_arr.any() and display_arr.any():
                    with img_col:
                        st.image(image=bb_img_arr, caption="Meter Image")
                    with roi_cols:
                        st.image(image=display_arr, caption="ROI Image", width=300)
            with st.container():
                st.number_input(label="Meter Reading", value=1,disabled=True)

    return None
# ------------------------------------------------------------------------------------------------------------------

# Model Training
def model_training_page() -> None:
    st.title("Model Training")
    return None
# ------------------------------------------------------------------------------------------------------------------


# Navigation Pages Setup
pg = st.navigation([st.Page(home_page, title="🏠 Home Page", default=True),
                    st.Page(bulk_data_page, title="📂 Bulk Data"), 
                    st.Page(prediction_page, title="🔍 Prediction"), 
                    st.Page(model_training_page, title="⚙️ Model Training")])
pg.run()