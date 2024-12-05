import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import os
class FeatureExploration:
    def __init__(self, df, target, config):
        
        if df is None:
            temp_df_path = os.path.join(os.getcwd(), 'temp', 'prepared_data.csv')  # Load the processed data
            self.df = pd.read_csv(temp_df_path)
        else:
            self.df = df
        
        self.target = target
        self.config = config
        self.model_for_selection = self.config.get('feature_exploration', {}).get('feature_selection', {}).get('model_for_selection', 'random_forest')
        self.model = self._initialize_model()
        self.output_folder = self.config.get('output_folder', 'output')
        self.importance_df=None
        
        os.makedirs(self.output_folder,exist_ok=True)

    def _initialize_model(self):
        """Initialize model based on configuration."""
        model_type = self.model_for_selection.lower()
        print("Model for feature selection: ",model_type)
        if model_type == 'random_forest':
            if self.config.get('feature_exploration', {}).get('task_type', 'classification') == 'classification':
                return RandomForestClassifier()
            else:
                return RandomForestRegressor()
        elif model_type == 'xgboost':
            if self.config.get('feature_exploration', {}).get('task_type', 'classification') == 'classification':
                return XGBClassifier()
            else:
                return XGBRegressor()
        else:
            raise ValueError(f"Unsupported model: {model_type}")

    def compute_feature_importance(self):
        """Compute and save feature importance."""
        # Separate target variable from features
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        # Fit the model
        self.model.fit(X, y)

        # Get feature importances
        feature_importances = self.model.feature_importances_

        # Create a DataFrame for feature importance
        self.importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importances
        })

        # Sort the features by importance
        self.importance_df = self.importance_df.sort_values(by='Importance', ascending=False)
        
        self.importance_df["Cummulative Importance"] = self.importance_df["Importance"].cumsum();

        # Save feature importance CSV
        
        model_name = self.model_for_selection.lower()
        self.importance_df.to_csv(f'{self.output_folder}/feature_importance_{model_name}.csv', index=False)

        # Plot feature importance
        if self.config.get('feature_exploration', {}).get('feature_imp_plot', 'Y') == 'Y':
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Importance', y='Feature', data=self.importance_df)
            plt.title(f'Feature Importance ({model_name})')
            plt.savefig(f'{self.output_folder}/feature_importance_{model_name}.png')
            plt.close()

        # Feature selection based on configuration
        self.select_features_based_on_importance(self.importance_df)

    def select_features_based_on_importance(self, importance_df):
        """Select top N features or features based on cumulative importance."""
        feature_selection_config = self.config.get('feature_exploration', {}).get('feature_selection', {})
        top_n = feature_selection_config.get('top_n',None)
        cum_importance_threshold = feature_selection_config.get('cum_importance_threshold',None)

        if top_n:
            # Select top N features based on importance
            selected_features = importance_df.head(int(top_n))
            selected_features.to_csv(f"{self.config.get('output_folder', 'output')}/selected_features_top_{top_n}.csv", index=False)
            print(f"Top {top_n} features selected and saved.")
        elif cum_importance_threshold:
            # Select features where cumulative importance exceeds the threshold
            importance_df['Cumulative_Importance'] = importance_df['Importance'].cumsum()
            selected_features = importance_df[importance_df['Cumulative_Importance'] <= float(cum_importance_threshold)]
            selected_features.to_csv(f"{self.output_folder}/selected_features_cumulative_{cum_importance_threshold}.csv", index=False)
            print(f"Features selected based on cumulative importance threshold ({cum_importance_threshold}) and saved.")
        else:
            print("No valid feature selection method provided in configuration.")

