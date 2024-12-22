import os
import logging

from DataAcquisition import DataAcquisition
from ProdRun import Scoring
import pandas as pd

from ProdSnowql.PrepareData import PrepareData

# Setting up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Start:
    def __init__(self, config):
        """
        Initialize the AutoML class with configuration details.
        This will initialize all the required classes for data acquisition, 
        data preparation, feature exploration, and model training.

        Parameters:
        - config: Configuration dictionary containing all the necessary settings.
        """
        self.config = config
        self.data_acquisition = DataAcquisition(config)  # Initialize DataAcquisition
        self.scoring = None
        self.prepareData = PrepareData(self.config)
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """
        Set up the logger for the class.
        Returns:
        - Configured logger instance.
        """
        logger = logging.getLogger("Prod")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def run(self):
        """
        Run the AutoML pipeline.
        - Fetch training and OOT data.
        - Perform data preparation, feature exploration, and model building.
        """
        try:
            logger.info("Starting Prod run")
            self.prepareData.run()
            logger.info("Prediction data prepared successfully")
            # Step 1: Data Acquisition

            # if self.config['local_file']['use']:
            #     logger.info("Fetching from local: prod prediction data...")
            #     prod_data = pd.read_csv(self.config['local_file']['local_file_name'])
            # else:   
            #     logger.info("Fetching prod prediction data data...")
            #     prod_data = self.data_acquisition.fetch_prod_data()  # Fetch training data
            #     logger.info(f"Training data fetched with {len(prod_data)} rows.")
            
            
            # if self.config['predict']['enabled']=='Y':
                
            #     # Initialize Scoring
            #     self.scoring = Scoring(self.config, logger=self.logger)
            #     results = self.scoring.score(prod_data)
            #     self.logger.info(f"Prod run completed. Results: {results}")

        except Exception as e:
            logger.error(f"An error occurred during the PROD RUN process: {str(e)}")
        finally:
            # Step 5: Close the Data Acquisition connection after the process is complete
            try:
                logger.info("Closing the data acquisition connection.")
                self.data_acquisition.close_connection()
                
            except Exception as close_err:
                logger.error(f"Error while closing the data acquisition connection: {str(close_err)}")

