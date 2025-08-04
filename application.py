import streamlit as st
from PIL import Image
import numpy as np

# Own Module
from metervision.logger.logs import logging

# loading the environement variable
from dotenv import load_dotenv

if "env_loaded" not in st.session_state:
    load_dotenv()
    st.session_state["env_loaded"] = True
    logging.info("Environment Variables are Loaded Successfully")
else:
    logging.info("Environment Variables are Already Loaded")