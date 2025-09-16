## in here we will write the training pipeline....

from config.database_config import *
from config.path_config import *
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.feature_store import RedisFeatures
from src.model_training import ModelTraining


if __name__ == "__main__":
    ## Data Ingestion.......
    
    data_ingestion = DataIngestion(DB_CONFIG,RAW_DIR)
    data_ingestion.run()

    ## Data processing

    feature_store = RedisFeatures()

    data_processor = DataProcessing(TRAIN_PATH,TEST_PATH,feature_store)
    data_processor.run()

    ## Model Training

    feature_store = RedisFeatures()
    model_trainer = ModelTraining(feature_store)
    model_trainer.run()