import logging
import os
from datetime import datetime

# Wriging logging file.....

LOG_DIR = 'logs'
os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR,f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    filename=LOG_FILE,
    format= '%(ascitime)s - %(levelname)s - %(message)s',
    level=logging.info()
)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger 