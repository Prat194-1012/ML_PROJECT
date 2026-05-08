
import os
import sys
import logging
import src.exception as customexception
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transform import DataTransformation
from src.components.data_transform import DataTransformationConfig

from src.components.model_trainer import ModelTrainerconfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion method starts")
        try:
            df = pd.read_csv(os.path.join("notebook/stud.csv"))
            logging.info("Dataset read as pandas dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("train_test_split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("train_test_split completed")
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise customexception.custom_exception(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data =obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    test_arr,train_arr,_ = data_transformation.initiate_data_tranformation(train_data,test_data)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr,test_arr)

    print(np.isnan(train_arr).sum())
    print(np.isnan(test_arr).sum())