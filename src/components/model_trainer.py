
import os
import sys
from src.utils import evaluate_models, save_object

from dataclasses import dataclass
from src.logger import logging
from src.exception import custom_exception
from catboost import CatBoostRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import(
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.model_selection import GridSearchCV

@dataclass
class ModelTrainerconfig:
    trained_model_file_path: str=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerconfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear regression": LinearRegression(),
                "KNN Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Adaboost Regressor": AdaBoostRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "XGB Regressor": XGBRegressor(),
                "Catboost regressor": CatBoostRegressor(verbose=False)
            }

            params = {
                "Decision Tree Regressor": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]
                },

                "Random Forest Regressor": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20]
                },

                "Adaboost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.5]
                },

                "Gradient Boosting Regressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.5],
                    "subsample": [0.6, 0.8, 1.0],
                    "max_depth": [3, 5, 7]
                },

                "Linear regression": {},

                "KNN Regressor": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan"],
                },

                "XGB Regressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.5]
                },

                "Catboost regressor": {
                    "iterations": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.5],
                    "depth": [3, 5, 7]
                }
            }
             
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_score_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_score_name]

            if best_model_score < 0.6:
                raise custom_exception("no best model found")
            
            logging.info(f"best model found on both training and testing dataset: {best_model_score_name} with r2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            prediction = best_model.predict(X_test)

            r2_square = r2_score(y_test, prediction)
            return r2_square
        
        except custom_exception as e:
            raise custom_exception(e, sys)