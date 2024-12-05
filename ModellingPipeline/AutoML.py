import os
import logging
from DataPreparation import DataPreparation
from DataAcquisition import DataAcquisition
from FeatureExploration import FeatureExploration

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

    def run(self):
        """
        Run the AutoML pipeline.
        - Fetch training and OOT data.
        - Perform data preparation, feature exploration, and model building.
        """
        try:
            # Step 1: Data Acquisition
            logger.info("Fetching training and out-of-time data...")
            training_data = self.data_acquisition.fetch_training_data()  # Fetch training data
            # oot_data = self.data_acquisition.fetch_oot_data()  # Fetch OOT data (optional)
            logger.info(f"Training data fetched with {len(training_data)} rows.")
            
            # Step 2: Data Preparation
            logger.info("Starting data preparation...")
            self.data_preparation = DataPreparation(training_data, self.config, self.config.get('target_column'))
            self.data_preparation.handle_missing_values()
            self.data_preparation.standardize_or_normalize()
            self.data_preparation.encode_categorical_features()
            # self.data_preparation.save_columns()
            # self.data_preparation.save_label_encoders(label_mappings)
            
            # Saving the prepared DataFrame (after handling missing values, encoding, etc.)
            prepared_df = self.data_preparation.df
            self.data_preparation.save_df('prepared_data.csv')
            logger.info("Prepared data saved to temp/prepared_data.csv")

            # Step 3: Feature Exploration
            if(self.config.get('feature_exploration',{}).get('enabled','N'))=='Y':
                logger.info("Starting feature exploration...")
                self.feature_exploration = FeatureExploration(prepared_df, self.config.get('target_column'), self.config)
                self.feature_exploration.compute_feature_importance()
                
                # Saving the selected features and important columns
                self.feature_exploration.select_features_based_on_importance(self.feature_exploration.importance_df)

                logger.info("Selected features saved to temp/selected_features.csv")

            # Step 4: Further steps like model training can follow here, using prepared data and selected features.

        except Exception as e:
            logger.error(f"An error occurred during the AutoML process: {str(e)}")
        finally:
            # Step 5: Close the Data Acquisition connection after the process is complete
            try:
                logger.info("Closing the data acquisition connection.")
                self.data_acquisition.close_connection()
            except Exception as close_err:
                logger.error(f"Error while closing the data acquisition connection: {str(close_err)}")
