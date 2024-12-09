import glob
import os
import logging
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from math import sqrt

from DataPreparation import DataPreparation


class Scoring:
    def __init__(self, config, logger=None):
        """
        Initialize the Scoring class.
        Parameters:
        - config: Configuration dictionary containing paths and settings for scoring.
        - logger: Logger instance for logging. If None, a logger is created.
        """
        self.config = config
        self.logger = logger or self._setup_logger()
        self.index_col = self.config['Index']
        self.target = self.config['target_column']
        self.output_folder = config.get("output_folder", "output")
        self.best_model_folder = os.path.join(self.output_folder,"build", "best_model/model")
        self.task_type = config.get("task_type", "regression").lower()
        self.score_folder = os.path.join(config.get("output_folder", "output"),"score")
        print(self.score_folder)
        # Ensure the output folder exists
        os.makedirs(self.score_folder, exist_ok=True)

    def _setup_logger(self):
        """
        Set up the logger for the class.
        Returns:
        - Configured logger instance.
        """
        logger = logging.getLogger("Scoring")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def _get_best_model_path(self):
        """
        Dynamically find and return the path of the .gz file in the best_model_folder.
        Returns:
        - Path to the best model file.
        Raises:
        - ValueError if no .gz file or multiple .gz files are found.
        """
        try:
            gz_files = glob.glob(os.path.join(self.best_model_folder, "*.gz"))
            if len(gz_files) == 0:
                raise ValueError(f"No .gz file found in {self.best_model_folder}.")
            if len(gz_files) > 1:
                raise ValueError(f"Multiple .gz files found in {self.best_model_folder}. Please ensure only one model file is present.")
            self.logger.info(f"Found model file: {gz_files[0]}")
            return gz_files[0]
        except Exception as e:
            self.logger.error(f"Error locating the best model file: {e}")
            raise

    def load_best_model(self):
        """
        Load the best model from the .gz file.
        Returns:
        - Loaded model object.
        """
        try:
            best_model_path = self._get_best_model_path()
            self.logger.info(f"Loading the best model from {best_model_path}...")
            model = joblib.load(best_model_path)
            self.logger.info("Best model successfully loaded.")
            return model
        except Exception as e:
            self.logger.error(f"Error loading the best model: {e}")
            raise
    
    def score(self, oot_data):
        """
        Score the OOT data using the trained pipeline.
        Parameters:
        - oot_data: The out-of-time dataset (already fetched).

        Returns:
        - evaluation_results: Dictionary containing scoring metrics.
        """
        try:
           
            self.logger.info("Starting data preparation for OOT data...")
            # Step 1: Load Selected Features
            selected_features = self._load_selected_features()+[self.target,self.index_col]
            self.logger.info(f"Selected features loaded: {len(selected_features)} features.")
            oot_data = oot_data[selected_features]
            
            # Step 2: Data Preparation
            if self.config['data_preparation']['enabled']=='Y':
                self.logger.info("Starting data preparation...")
                self.data_preparation = DataPreparation(oot_data, self.config, self.config.get('target_column'))
                self.data_preparation.handle_missing_values()
                self.data_preparation.encode_categorical_features()
                self.data_preparation.standardize_or_normalize()
                oot_prepared_df = self.data_preparation.df
                self.data_preparation.save_df('oot_prepared_data.csv')
                self.logger.info("Prepared data saved to temp/oot_prepared_data.csv")

            # Step 3: Load the Best Model
            

            best_model = self.load_best_model()
            self.logger.info("Best model successfully loaded.")
            
            print(oot_data.columns)

            # Step 4: Predict and Evaluate
            y_true = oot_prepared_df[self.target]
            drop_columns = [self.target,self.index_col] if self.index_col not in [None,""] else [self.target]
            X_oot = oot_data.drop(columns=drop_columns)
            predictions = best_model.predict(X_oot)

            pred_vs_actual = pd.DataFrame({
                "Actual": y_true,
                "Predicted": predictions
                })

            # Save Predictions vs Actuals DataFrame
            pred_vs_actual_file = os.path.join(self.score_folder, "pred_vs_actual.csv")
            pred_vs_actual.to_csv(pred_vs_actual_file, index=False)
            self.logger.info(f"Predictions vs Actuals saved to {pred_vs_actual_file}.")

            if self.task_type == "regression":
                mse = mean_squared_error(y_true, predictions)
                rmse = sqrt(mse)
                r2 = r2_score(y_true, predictions)
                evaluation_results = {"MSE": mse, "RMSE": rmse, "R2_Score": r2}
            else:  # Classification
                accuracy = accuracy_score(y_true, predictions)
                evaluation_results = {"Accuracy": accuracy}

                # Save probabilities if available
                if hasattr(best_model, "predict_proba"):
                    probabilities = best_model.predict_proba(oot_data)
                    prob_file = os.path.join(self.score_folder, "oot_probabilities.csv")
                    pd.DataFrame(probabilities).to_csv(prob_file, index=False)
                    self.logger.info("Class probabilities saved.")

            # Step 5: Save Predictions
            predictions_file = os.path.join(self.score_folder, "oot_predictions.csv")
            pd.DataFrame(predictions, columns=["Predictions"]).to_csv(predictions_file, index=False)
            self.logger.info("OOT predictions saved.")

            return evaluation_results
        except Exception as e:
            self.logger.error(f"Error during OOT scoring: {e}")
            raise
        
    def _load_selected_features(self):
        """
        Load the selected features from a CSV file.
        Returns:
        - List of selected features.
        """
        # Determine the best model name from the path
        selected_features_path = os.path.join(self.output_folder, "build", "best_model", "params",
                                              f"selected_features.csv")

        if not os.path.exists(selected_features_path):
            raise FileNotFoundError(f"Selected features file not found: {selected_features_path}")

        selected_features = pd.read_csv(selected_features_path)["feature"].tolist()
        self.logger.info(f"Loaded {len(selected_features)} selected features from {selected_features_path}.")
        return selected_features
