import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class EDA:
    def __init__(self,config, df, target, output_folder):
        """
        Parameters:
        - df: DataFrame containing the dataset.
        - target: Target column for analysis.
        - output_folder: Root folder where the analysis artifacts will be saved.
        """
        self.config = config
        self.df = df
        self.target = target
        self.output_folder = output_folder
        self.index_col = self.config.get("Index",None)

        # Create the root output folder
        os.makedirs(self.output_folder, exist_ok=True)

    def save_plot(self, folder_name, file_name):
        """Helper function to save a plot to the appropriate folder."""
        folder_path = os.path.join(self.output_folder, 'analysis', folder_name)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, file_name)
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

    def summary_statistics(self):
        """Save summary statistics as a CSV file."""
        stats_folder = os.path.join(self.output_folder, 'analysis', 'summary_statistics')
        os.makedirs(stats_folder, exist_ok=True)

        # Save numerical statistics
        try:
            self.df.describe().to_csv(os.path.join(stats_folder, 'numerical_summary.csv'))
            
            # Save categorical statistics
            self.df.describe(include=['object']).to_csv(os.path.join(stats_folder, 'categorical_summary.csv'))
        
        except Exception as e:
            print(f"Error: {e}")
            print("Continuing task")

    def correlation_heatmap(self):
        """Save the correlation heatmap as a plot."""
        drop_columns = [self.index_col] if self.index_col not in [None,''] else []
        corr = self.df.drop(columns=drop_columns).corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title("Correlation Heatmap")
        self.save_plot('correlation_heatmap', 'correlation_heatmap.png')

    def target_distribution(self):
        """Save the target distribution plot."""
        
        plt.figure(figsize=(8, 6))
        if self.df[self.target].dtype in ['int64', 'float64']:
            sns.histplot(self.df[self.target], kde=True, bins=30, color='blue')
        else:
            sns.countplot(x=self.df[self.target], palette='Set2')
        plt.title(f"Distribution of Target Variable: {self.target}")
        self.save_plot('target_distribution', f'{self.target}_distribution.png')

    def missing_values_analysis(self):
        """Save the missing values analysis plot."""
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        missing_data = pd.DataFrame({'Missing Values': missing, 'Percentage': missing_percent})
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_data.index, y=missing_data['Percentage'], palette='Set2')
        plt.title("Missing Values Analysis")
        plt.xticks(rotation=90)
        plt.ylabel("Percentage (%)")
        self.save_plot('missing_values', 'missing_values_analysis.png')

        # Save missing data summary as CSV
        missing_folder = os.path.join(self.output_folder, 'analysis', 'missing_values')
        os.makedirs(missing_folder, exist_ok=True)
        missing_data.to_csv(os.path.join(missing_folder, 'missing_values_summary.csv'))

    def outlier_analysis(self, features=None):
        """
        Save outlier detection plots using boxplots.
        Parameters:
        - features: List of features to analyze for outliers. If None, analyze all numerical features.
        """
        if features is None:
            features = self.df.select_dtypes(include=['int64', 'float64']).columns

        for feature in features:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=self.df[feature], color='orange')
            plt.title(f"Outlier Analysis for {feature}")
            self.save_plot('outlier_analysis', f'outlier_{feature}.png')

    def pairwise_relationships(self, features=None):
        """
        Save pairwise relationship plots using a pairplot.
        Parameters:
        - features: List of features to include. If None, uses all features.
        """
        if features is None:
            features = self.df.columns
        features = [feature for feature in features if feature not in [self.target,self.index_col]]
        pairplot = sns.pairplot(self.df[features], diag_kind='kde', corner=True)
        pairplot.fig.suptitle("Pairwise Relationships Between Features", y=1.02)
        self.save_plot('pairwise_relationships', 'pairwise_relationships.png')

    def target_vs_features(self, features=None):
        """
        Save plots showing the relationship between the target and individual features.
        Parameters:
        - features: List of features to analyze. If None, analyze all features.
        """
        if features is None:
            features = [col for col in self.df.columns if col not in [self.target,self.index_col]]

        for feature in features:
            plt.figure(figsize=(8, 6))
            if self.df[feature].dtype in ['int64', 'float64']:
                sns.scatterplot(x=self.df[feature], y=self.df[self.target])
            else:
                sns.boxplot(x=self.df[feature], y=self.df[self.target])
            plt.title(f"{feature} vs {self.target}")
            self.save_plot('target_vs_features', f'{feature}_vs_{self.target}.png')

    def feature_distribution(self, features=None):
        """
        Save plots of the distribution of numerical features.
        Parameters:
        - features: List of features to analyze. If None, analyze all numerical features.
        """
        if features is None:
            features = self.df.select_dtypes(include=['int64', 'float64']).columns

        for feature in features:
            plt.figure(figsize=(8, 6))
            sns.histplot(self.df[feature], kde=True, bins=30, color='purple')
            plt.title(f"Distribution of {feature}")
            self.save_plot('feature_distribution', f'{feature}_distribution.png')

    def generate_report(self):
        """Save a summary report of the dataset."""
        report_folder = os.path.join(self.output_folder, 'analysis', 'report')
        os.makedirs(report_folder, exist_ok=True)

        # Save dataset info as a text file
        with open(os.path.join(report_folder, 'dataset_info.txt'), 'w') as f:
            f.write(str(self.df.info()))

        # Save summary statistics as a CSV
        self.df.describe(include='all').to_csv(os.path.join(report_folder, 'summary_statistics.csv'))
