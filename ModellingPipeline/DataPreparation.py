import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import json

class DataPreparation:
    def __init__(self, df, config, target):
        self.df = df
        self.config = config
        self.target = target
        self.index_col = self.config.get("Index","")
        self.temp_folder = os.path.join(os.getcwd(), 'temp')  # Save to temp folder
        self.input_folder = self.config.get('input_folder', 'input')  # Folder to save output files
        self.data_preparation_config = self.config.get("data_preparation")
        # Ensure the temp and output folder exist
        os.makedirs(self.temp_folder, exist_ok=True)
        os.makedirs(self.input_folder, exist_ok=True)
        self.save_columns()
        self.convert_columns_to_float()
    
    def convert_columns_to_float(self):
        """Convert columns containing numeric values stored as strings to float."""
        # Loop through each column
        print(self.config.get("Index"))
        for col in self.df.columns:
            
            # Check if the column is of type object (i.e., potentially a string)
            if self.df[col].dtype == 'object' and col not in [self.target,self.index_col]:
                # Attempt to convert the column to float, handling errors gracefully
                try:
                    # Convert to float, errors='coerce' will turn non-convertible values into NaN
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    print(f"Column '{col}' converted to float.")
                except Exception as e:
                    print(f"Error converting column '{col}' to float: {e}")
            else:
                print(f"Column '{col}' is already numeric.")
        
    def save_df(self,filename):
        self.df.to_csv(os.path.join(self.temp_folder, filename))

    def save_label_encoders(self, label_encoders, filename):
        """Save the label encoders (categorical column mapping) to a CSV file."""
        label_encoders_df = pd.DataFrame(label_encoders)
        label_encoders_df.to_csv(os.path.join(self.input_folder, filename), index=False)
        print(f"Saved label encoders to {os.path.join(self.input_folder, filename)}")

    
    def save_columns(self):
        """Save the column names and their data types to a txt file."""
         
        columns_info = [(column, self.df[column].dtype) for column in self.df.columns]
        columns_file_path = os.path.join(self.input_folder, 'columns.txt')

        with open(columns_file_path, 'w') as file:
            for column, dtype in columns_info:
                file.write(f"{column},{dtype}\n")
        print(f"Saved columns and data types to {columns_file_path}")

    def handle_missing_values(self):
        """Handle missing values according to configuration."""
        if self.data_preparation_config.get('fillna', 'N') == 'Y':
            # fill_value = self.config.get('fill_value', 'mean')
            # imputer = SimpleImputer(strategy=fill_value)
            # self.df = pd.DataFrame(imputer.fit_transform(self.df), columns=self.df.columns)
            self.df.fillna(0, inplace=True)
        elif self.config.get('dropna', 'N') == 'Y':
            self.df.dropna(inplace=True)

    def standardize_or_normalize(self):
        """Standardize or normalize the numerical columns excluding the index and target column."""
        # Get the list of numeric columns excluding the Index and target column
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns

        # Exclude Index and target columns
        numeric_cols = [col for col in numeric_cols if col not in [self.config.get("Index", ""), self.target]]
        
        print(f"Numeric Columns for Standardization: {numeric_cols}")
        
        if self.data_preparation_config.get('standardize', 'N') == 'Y':
            # Standardize the selected numeric columns
            scaler = StandardScaler()
            self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
            print("Standardization completed.")


        
    def encode_categorical_features(self):
        """Encode categorical features using label encoding or one-hot encoding."""
        encoding_type = self.data_preparation_config.get('label_encoding', 'label')
        cat_columns = self.data_preparation_config.get('categorical_columns', None)  # Get the list of categorical columns from config
        label_encoders = {}

        if encoding_type == 'label' and cat_columns:
            for col in cat_columns:
                if col in self.df.columns:
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col])
                    label_encoders[col] = dict(zip(le.classes_, range(len(le.classes_))))
                    print(f"Encoded column '{col}' with label encoding.")
            # Save label encoders to a CSV
            self.save_label_encoders(label_encoders, 'label_encoders.csv')

        elif encoding_type == 'one_hot' and cat_columns:
            self.df = pd.get_dummies(self.df, columns=cat_columns)
            print(f"Performed one-hot encoding for columns: {cat_columns}")
        
        # Save columns info to a text file
        # self.save_columns()
