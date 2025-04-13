
import pandas as pd
import numpy as np
import os
import sys
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


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

def evaluate_models(X_train, y_train, X_test, y_test, models,param):
    """
    Evaluate the models and return a report of their performance.
    """
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            if len(param[model_name]) > 0:
                grid_search = GridSearchCV(model, param[model_name], cv=3, n_jobs=-1, verbose=2)
                grid_search.fit(X_train, y_train)
                model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
            logging.info(f"{model_name} model train score: {train_model_score}")
            logging.info(f"{model_name} model test score: {test_model_score}")
            logging.info(f"{model_name} model parameters: {model.get_params()}")
        logging.info(f"Model evaluation report: {report}")
        return report
    except Exception as e:
        raise CustomException(e, sys) from e    