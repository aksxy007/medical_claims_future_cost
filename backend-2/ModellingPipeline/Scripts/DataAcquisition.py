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

    def fetch_training_data(self):
        """
        Fetch the training data using the input_table_name from the config.
        """
        input_table = self.config.get('input_table_name', None)
        if input_table is None:
            raise ValueError("input_table_name is not provided in the config.")
        
        query = f"SELECT * FROM {input_table}"
        print(f"Fetching training data from table: {input_table}")
        
        # Fetch training data as Pandas DataFrame
        training_data = self.sf_connector.fetch_as_pandas(query)
        print(f"Training data fetched with {len(training_data)} rows.")
        return training_data

    def fetch_oot_data(self, oot_table_name=None):
        """
        Fetch out-of-time (OOT) data from the specified table.
        :param oot_table_name: Optional parameter, if not provided, it will be fetched from config.
        """
        oot_table = oot_table_name or self.config.get('oot_table_name', None)
        
        if oot_table is None:
            raise ValueError("oot_table_name is not provided in the config or as an argument.")
        
        query = f"SELECT * FROM {oot_table}"
        print(f"Fetching out-of-time (OOT) data from table: {oot_table}")
        
        # Fetch OOT data as Pandas DataFrame
        oot_data = self.sf_connector.fetch_as_pandas(query)
        print(f"OOT data fetched with {len(oot_data)} rows.")
        return oot_data

    def close_connection(self):
        """
        Close the Snowflake connection.
        """
        self.sf_connector.close_connection()
