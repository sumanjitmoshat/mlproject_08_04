import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%Y-%m-%d')}.log"
LOG_DIR=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH=os.path.join(LOG_DIR,LOG_FILE)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='%(asctime)s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
)

if __name__ == "__main__":
    logging.info("Logging setup complete.")
    logging.info("This is a test log message.")
    logging.info("Log file path: %s", LOG_FILE_PATH)
    logging.info("Log directory: %s", LOG_DIR)