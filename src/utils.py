
import os
import pickle
import pandas as pd
import numpy as np
import sys
import src.exception as custom_exception
import dill

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise custom_exception.custom_exception(e, sys)
            