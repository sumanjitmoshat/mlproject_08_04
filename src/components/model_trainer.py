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
            model_report: dict=evaluate_models(X_train, y_train, X_test, y_test, models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            logging.info("Tuning hyperparameters for the best model")
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}")
            if best_model_name == 'RandomForest':
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            elif best_model_name == 'DecisionTree':
                param_grid = {
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            elif best_model_name == 'LinearRegression':
                param_grid = {}
            elif best_model_name == 'SVR':
                param_grid = {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                }
            elif best_model_name == 'XGBoost':
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
            elif best_model_name == 'CatBoost':
                param_grid = {
                    'iterations': [100, 200],
                    'depth': [6, 8]
                }
            elif best_model_name == 'LightGBM':
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'num_leaves': [31, 63]
                }
            grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            logging.info(f"Best hyperparameters: {grid_search.best_params_}")
            logging.info("Evaluating the best model on test data")
            y_test_pred = best_model.predict(X_test)
            r2_square = r2_score(y_test, y_test_pred)
            logging.info(f"R2 score on test data: {r2_square}")
            logging.info("Saving the best model")
            """ preprocessor = load_object(preprocessor_path)
            model_with_preprocessor = {
                'model': best_model,
                'preprocessor': preprocessor
            }
            save_object(self.model_trainer_config.trained_model_file_path, model_with_preprocessor) """
            #logging.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}")
            return r2_square, best_model_name, grid_search.best_params_
        except Exception as e:
            raise CustomException(e, sys) from e
