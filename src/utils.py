
import pandas as pd
import numpy as np
import os
import sys
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score


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

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate the models and return a report of their performance.
    """
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_square = r2_score(y_test, y_pred)
            report[model_name] = r2_square
        return report
    except Exception as e:
        raise CustomException(e, sys) from e    