
import os
import pickle
import pandas as pd
import numpy as np
import sys
import src.exception as custom_exception
import dill
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise custom_exception.custom_exception(e, sys)
            

def evaluate_models(x_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(X_test)
            r2_train = r2_score(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            report[model_name] = r2_test
      
        return report
    except Exception as e:
        raise custom_exception.custom_exception(e, sys)