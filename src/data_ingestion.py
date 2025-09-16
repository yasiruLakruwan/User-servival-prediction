import psycopg2
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomExeption
from config.database_config import DB_CONFIG
from config.path_config import *
import sys
import os
from sklearn.model_selection import train_test_split
import pandas as pd

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self,db_config,output_dir):
        self.db_config = db_config
        self.output_dir= output_dir

        os.makedirs(self.output_dir,exist_ok=True)

    def connect_to_db(self):
        try:
            logger.info("Connecting database ............")
            conn = psycopg2.connect(
                host = self.db_config['host'], 
                port = self.db_config['port'],
                dbname = self.db_config['dbname'],
                user = self.db_config['user'],
                password = self.db_config['password']
            )
            logger.info("Database connected.........")
            return conn
        
        except Exception as e:
            logger.error(f"Error hapened in the connecting {e}")
            raise CustomExeption("Error while connecting the database",e)
        
    def extract_data(self):
        try:
            conn = self.connect_to_db()
            query = "select * from public.titanic"
            df = pd.read_sql_query(query,conn)
            conn.close()
            logger.info("Data extracted from Database...")
            return df
        
        except Exception as e:
            logger.error(f"Error hapened extracting data {e}")
            raise CustomExeption(str(e),sys)

    def save_data(self,df):
        try:
            train_df,test_df = train_test_split(df,test_size=0.2,random_state=42)
            train_df.to_csv(TRAIN_PATH,index=False)
            test_df.to_csv(TEST_PATH,index = False)

            logger.info("Data spliting and saving done.....")

        except Exception as e:
            logger.error(f"Error hapening while data spliting {e}")
            raise CustomExeption(str(e),sys)
    
    def run(self):
        try:
            logger.info("Data ingestion Pipeline started...")
            df = self.extract_data()
            self.save_data(df)
            logger.info("End of the data ingestion pipeline")

        except Exception as e:
            logger.error(f"Error hapening while data spliting {e}")
            raise CustomExeption(str(e),sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion(DB_CONFIG,RAW_DIR)
    data_ingestion.run()