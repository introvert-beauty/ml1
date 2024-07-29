import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


class DataIngestionConfig:
    test_file_path = os.path.join("artifacts", "test.csv")
    train_file_path = os.path.join("artifacts", "train.csv")
    raw_file_path = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv(os.path.join("notebook", "cleaneddata.csv"))
            logging.info("Data ingestion started")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_file_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_file_path, index=False)
            
            logging.info("Split the train and test data")
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=42)

            train_set.to_csv(self.ingestion_config.train_file_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_file_path, index=False, header=True)

            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_file_path,
                self.ingestion_config.test_file_path
            )
        except Exception as e:
            pass
            # logging.error(f"Error during data ingestion: {e}")
            # raise CustomException(e)






