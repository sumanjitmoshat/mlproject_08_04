import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.model = RandomForestRegressor()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1]
            models = {
                'RandomForest': RandomForestRegressor(),
                'DecisionTree': DecisionTreeRegressor(),
                'LinearRegression': LinearRegression(),
                'SVR': SVR(),
                'XGBoost': XGBRegressor(),
                'CatBoost': CatBoostRegressor(verbose=0),
                'LightGBM': LGBMRegressor()
            }
            """ model_report: dict=evaluate_models(X_train, y_train, X_test, y_test, models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            logging.info("Tuning hyperparameters for the best model")
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}") """
            params = {
                'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
                'DecisionTree': {'max_depth': [10, 20]},
                'LinearRegression': {},
                'SVR': {'kernel': ['linear', 'rbf']},
                'XGBoost': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
                'CatBoost': {'depth': [6, 8], 'iterations': [100, 200]},
                'LightGBM': {'num_leaves': [31, 50], 'max_depth': [-1, 10]}
            }
            model_report: dict=evaluate_models(X_train, y_train, X_test, y_test, models,params)
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e, sys) from e
