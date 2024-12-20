import os
import logging
from DataPreparation import DataPreparation
from DataAcquisition import DataAcquisition
from FeatureExploration import FeatureExploration
from TrainModels import TrainModels  # Importing the TrainModels class
from EDA import EDA
from Scoring import Scoring
import pandas as pd

# Setting up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutoML:
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
        self.data_preparation = None
        self.feature_exploration = None
        self.train_models = None  # Placeholder for TrainModels
        self.scoring = None
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """
        Set up the logger for the class.
        Returns:
        - Configured logger instance.
        """
        logger = logging.getLogger("AutoML")
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
            # Step 1: Data Acquisition

            if self.config['local_file']['use']:
                logger.info("Fetching from local: training and out-of-time data...")
                training_data = pd.read_csv(self.config['local_file']['local_file_name'])
            else:   
                logger.info("Fetching training and out-of-time data...")
                training_data = self.data_acquisition.fetch_training_data()  # Fetch training data
                logger.info(f"Training data fetched with {len(training_data)} rows.")
            
            # Step 2: Data Preparation
            if self.config['data_preparation']['enabled']=='Y':
                logger.info("Starting data preparation...")
                self.data_preparation = DataPreparation(training_data, self.config, self.config.get('target_column'))
                self.data_preparation.handle_missing_values()
                self.data_preparation.encode_categorical_features()
                self.data_preparation.standardize_or_normalize()
                os.makedirs(os.path.join(self.config['output_folder'],"Prep"),exist_ok=True)
                self.data_preparation.save_df(os.path.join(self.config['output_folder'],"Prep","prepared_data.csv"))
                prepared_df = self.data_preparation.df
                
                print("Prepared df after standardization: ",prepared_df['FUTURE_COST'][0])
            
            

            # Saving the prepared DataFrame (after handling missing values, encoding, etc.)
                if self.config['eda']['enabled']=='Y':
                    logger.info("Starting EDA...")
                    self.eda = EDA(self.config,prepared_df, self.config.get('target_column'), self.config.get('output_folder','output'))
                    
                    # Perform EDA: Summary statistics, target distribution, correlation heatmap, etc.
                    self.eda.summary_statistics()
                    logger.info("stats Done")
                    self.eda.target_distribution()
                    logger.info("target dist Done")
                    self.eda.correlation_heatmap()
                    logger.info("corr Done")
                    self.eda.missing_values_analysis()
                    logger.info("missing val done")
                    self.eda.outlier_analysis()
                    logger.info("Outlier done")
                    # self.eda.pairwise_relationships()
                    # logger.info("Pair wise donne")
                    self.eda.target_vs_features()
                    logger.info("target vs feature done")
                    self.eda.feature_distribution()
                    logger.info("Eda Complete")
                    
                # self.data_preparation.save_df('prepared_data.csv')
                logger.info("Prepared data saved to output/Prep/prepared_data.csv")

            # Step 3: Feature Exploration
            if self.config.get('feature_exploration', {}).get('enabled', 'N') == 'Y':
                logger.info("Starting feature exploration...")
                self.feature_exploration = FeatureExploration(None, self.config)
                self.feature_exploration.compute_feature_importance()
                
                # Saving the selected features and important columns
                # self.feature_exploration.select_features_based_on_importance(self.feature_exploration.importance_df)

                logger.info("Selected features saved to output/Explore/selected_features.csv")

            # # Step 4: Model Training
            if self.config['build']['enabled']=='Y':
                logger.info("Starting model training...")
                self.train_models = TrainModels(self.config)  # Initialize the TrainModels class
                self.train_models.run()  # Execute the model training pipeline
                logger.info("Model training completed successfully.")
                
                
            
            if self.config['score']['enabled']=='Y':
                self.logger.info("Starting scoring pipeline...")
                if self.config['local_file']['use']:
                    logger.info("Fetching from local: out-of-time data...")
                    oot_data = pd.read_csv(self.config['local_file']['oot_file_name'])
                else:
                    logger.info("Fetching out-of-time data...")
                    oot_data = self.data_acquisition.fetch_oot_data()  # Fetch training data
                    logger.info(f"Oot data fetched with {len(oot_data)} rows.")
  
    
                # Initialize Scoring
                self.scoring = Scoring(self.config, logger=self.logger)
                results = self.scoring.score(oot_data)
                self.logger.info(f"Scoring completed. Results: {results}")

        except Exception as e:
            logger.error(f"An error occurred during the AutoML process: {str(e)}")
        finally:
            # Step 5: Close the Data Acquisition connection after the process is complete
            try:
                logger.info("Closing the data acquisition connection.")
                self.data_acquisition.close_connection()
                
            except Exception as close_err:
                logger.error(f"Error while closing the data acquisition connection: {str(close_err)}")
