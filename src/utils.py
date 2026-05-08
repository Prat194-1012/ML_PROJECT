
import os
import pickle
import pandas as pd
import numpy as np
import sys
import src.exception as custom_exception
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import logging

logging.basicConfig(level=logging.INFO)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise custom_exception.custom_exception(e, sys)
            

def evaluate_models(x_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            para = params[model_name]

            gs = GridSearchCV(model, para, cv=5)
            gs.fit(x_train, y_train)

            best_model_score = gs.best_score_
            if best_model_score < 0.5:
                raise custom_exception.custom_exception(
                    f"Best model score {best_model_score:.4f} is below threshold", sys
                )
            best_params = gs.best_params_
            logging.info(f"best params for {model_name}: {best_params}")
            
            best_model = model.__class__(**gs.best_params_)
            best_model.fit(x_train, y_train)

            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(X_test)
            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            report[model_name] = r2_test
      
        return report
    except Exception as e:
        raise custom_exception.custom_exception(e, sys)