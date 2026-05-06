
import os
import sys
import numpy as np
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import custom_exception
from src.logger import logging
from src.utils import save_object


class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_Data_Transformer_object(self):
        """
        this funtion is responsible for data tranformation
        """
        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"categorical columns transformation completed: {categorical_columns}")
            logging.info(f"numerical columns transformation completed: {numerical_columns}")
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise custom_exception(e, sys)
        
    def initiate_data_tranformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("read train and test data completed")
            logging.info("obtaining preprocessor object")
            preprocessor = self.get_Data_Transformer_object()
            target_columns_name= 'math_score'
            numerical_columns: list[str] = ['writing_score','reading_score']
            input_feature_train_df = train_df.drop(columns=[target_columns_name],axis=1)
            input_feature_test_df = test_df.drop(columns=[target_columns_name],axis=1)
            target_feature_train_df = train_df[target_columns_name]
            target_feature_test_df = test_df[target_columns_name]
         
            logging.info(f"applying preprocessor object on traning and testing dataframe")

            input_feature_train_df = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_df = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_df, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_df,np.array(target_feature_test_df)]
            
            logging.info(f"saving preprocessor object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor
            )


            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise e