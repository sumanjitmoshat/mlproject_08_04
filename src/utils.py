
import pandas as pd
import numpy as np
import os
import sys
from src.exception import CustomException
from src.logger import logging
import dill


def save_object( file_path,obj):
    """
    Save the object to a file using pickle.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e