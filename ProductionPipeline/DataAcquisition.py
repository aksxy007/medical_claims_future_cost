from SnowflakeConnector import SFConnector

class DataAcquisition:
    def __init__(self, config):
        """
        Initialize DataAcquisition class with config details.
        """
        self.config = config
        self.sf_connector = SFConnector(
            account=self.config['snowflake_connection']['account'],
            user=self.config['snowflake_connection']['user'],
            password=self.config['snowflake_connection']['password'],
            warehouse=self.config['snowflake_connection']['warehouse'],
            database=self.config['snowflake_connection']['database'],
            schema=self.config['snowflake_connection']['schema']
        )

    def fetch_prod_data(self):
        """
        Fetch the training data using the input_table_name from the config.
        """
        input_table = self.config.get('prediction_table_name', None)
        if input_table is None:
            raise ValueError("input_table_name is not provided in the config.")
        
        query = f"SELECT * FROM {input_table}"
        print(f"Fetching prod prediction data from table: {input_table}")
        
        # Fetch training data as Pandas DataFrame
        prod_data = self.sf_connector.fetch_as_pandas(query)
        print(f"Prod Prediction data fetched with {len(prod_data)} rows.")
        return prod_data

    def close_connection(self):
        """
        Close the Snowflake connection.
        """
        self.sf_connector.close_connection()
