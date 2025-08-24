import logging
import os
from datetime import datetime

logging_format = "[%(asctime)s]: %(levelname)s: %(module)s : %(message)s"
LOG_FILE = f"{datetime.now().strftime("%m_%d_%Y")}"
log_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path, f"{LOG_FILE}.log")

logging.basicConfig(
    format=logging_format,
    level=logging.INFO,
    handlers=[logging.StreamHandler(), logging.FileHandler(filename=LOG_FILE_PATH)],
)
