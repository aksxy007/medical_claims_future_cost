import os
import logging
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_squared_error, accuracy_score,r2_score
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from math import sqrt
import shutil

class TrainModels:
    def __init__(self, config):
        """
        Initialize the TrainModels class.
        Parameters:
        - config: Configuration dictionary containing task type, output folder, and train/test split ratio.
        """
        self.config = config
        # self.output_folder = os.path.join(config.get("output_folder", "output"),"explore")
        self.output_folder = config.get("output_folder", "output")
        self.temp_folder = "temp"
        self.build_folder = os.path.join(self.output_folder, "build","artifacts")
        self.best_model_folder = os.path.join(self.output_folder,"build", "best_model")
        self.train_test_ratio = config.get("train_test_ratio", 0.8)
        self.task_type = config.get("task_type", "regression").lower()
        self.target_column = config.get("target_column", "")
        self.models_config = config.get("build",{}).get("models", {})
        self.cv_enabled = config.get("build",{}).get("cv_enabled", False)
        self.cv_folds = config.get("build",{}).get("cv_folds", 5)
        self.cumulative_importance_threshold = config.get("train",{}).get("cumulative_importance_threshold", 0.95)
        self.logger = self._setup_logger()
        self.evaluation_results = {}

        # Ensure necessary folders exist
        os.makedirs(self.build_folder, exist_ok=True)
        os.makedirs(self.best_model_folder, exist_ok=True)

    def _setup_logger(self):
        """
        Set up the logger for the class.
        Returns:
        - Configured logger instance.
        """
        logger = logging.getLogger("TrainModels")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def _initialize_models(self):
        """
        Initialize models based on the configuration file.
        Returns:
        - A dictionary of models and their hyperparameters.
        """
        models = {}
        try:
            for model_name, allParams in self.models_config.items():
                model_class = None

                if model_name.lower() == "linear_regression" and self.task_type == "regression":
                    model_class = LinearRegression
                elif model_name.lower() == "logistic_regression" and self.task_type == "classification":
                    model_class = LogisticRegression
                elif model_name.lower() == "random_forest":
                    model_class = RandomForestRegressor if self.task_type == "regression" else RandomForestClassifier
                elif model_name.lower() == "decision_tree":
                    model_class = DecisionTreeRegressor if self.task_type == "regression" else DecisionTreeClassifier
                elif model_name.lower() == "svm":
                    model_class = SVR if self.task_type == "regression" else SVC
                elif model_name.lower() == "xgboost":
                    model_class = XGBRegressor if self.task_type == "regression" else XGBClassifier
                elif model_name.lower() == "lightgbm":
                    model_class = LGBMRegressor if self.task_type == "regression" else LGBMClassifier

                if model_class:
                    model = model_class()
                    models[model_name] = {
                        "model": model,
                        "default_params": allParams.get("default", {}),
                        "hyper_params": allParams.get("params", {}),
                    }

        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise
        return models

    def load_data(self):
        """
        Load the data with selected features and split it into train and test sets.
        """
        try:
            selected_features=[]
            if self.config.get('feature_exploration').get('enabled')=='Y' or os.path.join(
                    self.output_folder,
                    "selected_features_top_{}.csv".format(self.config["feature_exploration"]["feature_selection"]["top_n"]),
                ):
                selected_features_file = os.path.join(
                    self.output_folder,
                    "selected_features_top_{}.csv".format(self.config["feature_exploration"]["feature_selection"]["top_n"]),
                )
                selected_features_df = pd.read_csv(selected_features_file)
                selected_features = selected_features_df["Feature"].tolist()

            prepared_data_file = os.path.join(self.temp_folder, "prepared_data.csv")
            prepared_data = pd.read_csv(prepared_data_file)
            if self.target_column not in prepared_data.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in the prepared dataset.")

            selected_columns = selected_features + [self.target_column]
            data = prepared_data[selected_columns]
            # print(f"Loaded Data has {len(data.columns)} columns")

            X = data.drop(columns=[self.target_column])
            y = data[self.target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1 - self.train_test_ratio, random_state=42
            )

            # Save train/test datasets
            # X_train.to_csv(os.path.join(self.build_folder,"train_features.csv"), index=False)
            # X_test.to_csv(os.path.join(self.build_folder,"test_features.csv"), index=False)
            y_train.to_csv(os.path.join(self.build_folder, "train_target.csv"), index=False)
            y_test.to_csv(os.path.join(self.build_folder,"test_target.csv"), index=False)

            self.logger.info("Data successfully loaded, processed, and split into train/test sets.")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def train_model(self, model_name, model, X_train, y_train):
        """
        Train a single model.
        """
        try:
            # Get the model's parameters
            mymodel = model["model"]
            if self.cv_enabled:
                hpparams = model["hyper_params"]
                self.logger.info(f"Performing cross-validation for {model_name}...")
                grid = GridSearchCV(
                    mymodel, hpparams, cv=self.cv_folds, n_jobs=-1,
                    scoring="neg_mean_squared_error" if self.task_type == "regression" else "accuracy"
                )
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                params = model["default_params"]
                mymodel.set_params(**params)
                self.logger.info(f"Training {model_name} without cross-validation...")
                mymodel.fit(X_train, y_train)
                best_model = mymodel
                best_params = params

            # Save feature importance (if applicable)
            feature_importance = None
            if hasattr(best_model, "feature_importances_"):
                feature_importance = pd.DataFrame({
                    "Feature": X_train.columns,
                    "Importance": best_model.feature_importances_
                }).sort_values(by="Importance", ascending=False)
            
            return best_model, best_params, feature_importance
        except Exception as e:
            self.logger.error(f"Error training model {model_name}: {e}")
            raise

    def run(self):
        """
        Run the entire process: train models, select features based on importance, retrain, and evaluate.
        """
        try:
            self.logger.info("Starting model training process...")

            # Load data
            X_train, X_test, y_train, y_test = self.load_data()
            models = self._initialize_models()

            # Initial training to get feature importance (parallelized)
            feature_importance_per_model = {}
            
            # Parallelize the training process
            results = Parallel(n_jobs=cpu_count())(  # Using all available cores
                delayed(self.train_model)(model_name, model, X_train, y_train)
                for model_name, model in models.items()
            )

            # Process feature importance and filter features
            cumulative_threshold = self.cumulative_importance_threshold
            for model_name, (trained_model, best_params, feature_importance) in zip(models.keys(), results):
                if feature_importance is not None:
                    feature_importance["Cumulative_Importance"] = feature_importance["Importance"].cumsum()
                    selected_features = feature_importance[
                        feature_importance["Cumulative_Importance"] <= cumulative_threshold
                    ]["Feature"].tolist()
                    feature_importance_per_model[model_name] = selected_features
            
            self.logger.info("Intial model training finished")      
            self.logger.info("Training model on their selected features")
            # Retrain models with filtered features (parallelized)
            retrained_results = {}
            retrained_models = Parallel(n_jobs=cpu_count())(  # Using all available cores
                delayed(self.retrain_and_evaluate_model)(
                    model_name, model, X_train, X_test, y_train, y_test, feature_importance_per_model
                )
                for model_name, model in models.items()
            )

            # Collect results
            self.logger.info("collecting results")
            for model_name, (metric,rmse, r2,best_params) in zip(models.keys(), retrained_models):
                retrained_results[model_name] = {"MSE": metric, "RMSE":rmse, "R2_Score":r2, "Params": best_params}

            self.logger.info("Retraining models finished")
            # Choose the best model
            # Now find the best model based on the MSE (or other metric)
            # Handle None or invalid MSE values directly within the min() function
        
            minn =  float("inf")
            try:
                for model_name,rresults in retrained_results.items():
                    if rresults['MSE'] !=None and minn>rresults['MSE']:
                        best_model_name = model_name
                        minn=rresults['MSE']
                        # Extract model_name from the dictionary
                os.makedirs(os.path.join(self.best_model_folder,"model"),exist_ok=True)
                os.makedirs(os.path.join(self.best_model_folder,"params"),exist_ok=True)
            
                
                src =  os.path.join(self.build_folder,f"{best_model_name}_model.pkl")
                dest = os.path.join(self.best_model_folder,"model",f"{best_model_name}.gz")

                # Ensure destination directory exists
                os.makedirs(os.path.dirname(dest), exist_ok=True)

                # Copy the file
                try:
                    shutil.copy(src, dest)
                    print(f"File copied successfully from {src} to {dest}.")
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                except PermissionError as e:
                    print(f"Permission Error: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")

                self.logger.info("Saving best model")
                # Save evaluation results
                pd.DataFrame(retrained_results).to_csv(os.path.join(self.best_model_folder, "params", "evaluation_results.csv"), index=False)
                pd.DataFrame(feature_importance_per_model[best_model_name], columns=["feature"]).to_csv(
                    os.path.join(self.best_model_folder, "params", f"selected_features.csv"),
                    index=False
                )
                self.logger.info("Model training process completed successfully.")
            except ValueError:
                # In case retrained_results is empty or no valid MSE is found
                best_model_name = None
                self.logger.warning("No valid models found with MSE.")

            
            

        except Exception as e:
            self.logger.error(f"Error during the run process: {e}")
            raise

    def retrain_and_evaluate_model(self, model_name, model, X_train, X_test, y_train, y_test, feature_importance_per_model):
        """
        Retrain a model with selected features, evaluate it, and save it.
        Handles both regression and classification tasks.
        """
        try:
            if model_name in feature_importance_per_model:
                # Filter features
                filtered_features = feature_importance_per_model[model_name]
                X_train_filtered = X_train[filtered_features]
                X_test_filtered = X_test[filtered_features]

                # Save filtered train/test datasets
                X_train_filtered.to_csv(os.path.join(self.build_folder, "train_features.csv"), index=False)
                X_test_filtered.to_csv(os.path.join(self.build_folder, "test_features.csv"), index=False)

                # Retrain the model
                retrained_model, best_params, _ = self.train_model(model_name, model, X_train_filtered, y_train)

                # Make predictions
                predictions = retrained_model.predict(X_test_filtered)

                # Save predictions
                predictions_file = os.path.join(self.build_folder, f"{model_name}_predictions.csv")
                pd.DataFrame(predictions, columns=["Predictions"]).to_csv(predictions_file, index=False)

                # Calculate metrics
                if self.task_type == "regression":
                    metric = mean_squared_error(y_test, predictions)
                    rmse = sqrt(metric)
                    r2 = r2_score(y_test, predictions)
                else:  # Classification
                    metric = accuracy_score(y_test, predictions)
                    rmse = None  # Not applicable for classification
                    r2 = None    # Not applicable for classification
                    print(f"{model_name}_accuracy: {metric}")
                    # Optionally save probabilities if the model supports `predict_proba`
                    if hasattr(retrained_model, "predict_proba"):
                        probabilities = retrained_model.predict_proba(X_test_filtered)
                        prob_file = os.path.join(self.build_folder, f"{model_name}_probabilities.csv")
                        pd.DataFrame(probabilities).to_csv(prob_file, index=False)

                # Save the retrained model
                model_file = os.path.join(self.build_folder, f"{model_name}_model.pkl")
                joblib.dump(retrained_model, model_file)

                return metric, rmse, r2, best_params

            return None, None, None, None
        except Exception as e:
            self.logger.error(f"Error retraining and evaluating model {model_name}: {e}")
            raise
