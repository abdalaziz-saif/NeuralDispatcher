
"""

_ logger Module 

"""

import logging
import logging.config
from urllib.request import HTTPDefaultErrorHandler 

def get_logger(name: str) -> logging.Logger :

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

