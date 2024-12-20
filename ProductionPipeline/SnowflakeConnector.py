import snowflake.connector as sfconnector
from snowflake.connector import ProgrammingError
import pandas as pd
from pyspark.sql import SparkSession

from pyspark.sql import SparkSession
import sys
import os

os.environ['HADOOP_HOME'] = r"D:\Programs\hadoop\hadoop-3.4.4"
sys.path.append(r"D:\Programs\hadoop\hadoop-3.4.4\bin")
jdbc_path = r"C:\Users\atul1\Downloads\snowflake-jdbc-3.13.28.jar"
spark_sf_connector_path = r"C:\Users\atul1\Downloads\spark-snowflake_2.12-2.14.0-spark_3.2.jar"

class SFConnector:
    def __init__(self, account, user, password, warehouse, database, schema):
        """
        Initialize the SFConnector with required Snowflake connection details.
        """
        self.account = account
        self.user = user
        self.password = password
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.conn = None
        self.cursor = None
        self.spark_session = None
        self.connect()

    def connect(self):
        """
        Establish a connection to the Snowflake database.
        """
        try:
            self.conn = sfconnector.connect(
                account=self.account,
                user=self.user,
                password=self.password,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema
            )
            self.cursor = self.conn.cursor()
            print("Connected to Snowflake successfully.")
        except ProgrammingError as e:
            print(f"Error connecting to Snowflake: {e}")
            raise

    def close_connection(self):
        """
        Close the connection and cursor if open.
        """
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("Connection to Snowflake closed.")

    def execute_query(self, qry, fetch_one=False, fetch_all=False):
        """
        Execute a query and fetch results.
        :param qry: The query to execute.
        :param fetch_one: If True, fetches one result.
        :param fetch_all: If True, fetches all results.
        :return: Query results or None if no fetch is requested.
        """
        if not self.conn:
            self.connect()
        
        try:
            self.cursor.execute(qry)
            if fetch_one:
                return self.cursor.fetchone()
            if fetch_all:
                return self.cursor.fetchall()
            # Return None if no fetch is specified
        except ProgrammingError as e:
            print(f"Error executing query: {e}")
            raise

    def fetch_as_pandas(self, qry):
        """
        Fetch query results as a Pandas DataFrame.
        :param qry: The query to execute.
        :return: Pandas DataFrame with query results.
        """
        if not self.conn:
            self.connect()
        try:
            df = pd.read_sql(qry, self.conn)
            return df
        except Exception as e:
            print(f"Error fetching data as Pandas DataFrame: {e}")
            raise

    def fetch_as_spark(self, qry):
        """
        Fetch query results as a Spark DataFrame using the Snowflake Spark Connector.
        :param qry: The query to execute.
        :return: Spark DataFrame with query results.
        """
        if not self.spark_session:
            self.spark_session = SparkSession.builder \
                .appName("SnowflakeIntegration") \
                .config("spark.jars", f"{jdbc_path},{spark_sf_connector_path}") \
                .config("spark.executor.extraClassPath", f"{jdbc_path}:{spark_sf_connector_path}") \
                .config("spark.driver.extraClassPath", f"{jdbc_path}:{spark_sf_connector_path}") \
                .getOrCreate()
        
        print("Created Spark Session Successfully")

        snowflake_options = {
            "sfURL": f"{self.account}.snowflakecomputing.com",
            "sfDatabase": self.database,
            "sfSchema": self.schema,
            "sfWarehouse": self.warehouse,
            "sfUser": self.user,
            "sfPassword": self.password,
        }

        try:
            df = self.spark_session.read \
                .format("snowflake") \
                .options(**snowflake_options) \
                .option("query", qry) \
                .load()
            print("Fetched the data successfully")
            return df
        except Exception as e:
            print(f"Error fetching data as Spark DataFrame: {e}")
            raise
    
    def create_table_from_spark(self, query, table_name, mode="overwrite"):
        """
        Creates a new table in Snowflake using a query to fetch data into a Spark DataFrame,
        and then writes it to a Snowflake table.
        :param query: The query to execute and fetch data.
        :param table_name: The name of the Snowflake table to create.
        :param mode: The mode to use, either 'overwrite' or 'append'.
        :return: None
        """
        if not self.spark_session:
            self.spark_session = SparkSession.builder \
                .appName("SnowflakeIntegration") \
                .config("spark.jars", f"{jdbc_path},{spark_sf_connector_path}") \
                .config("spark.executor.extraClassPath", f"{jdbc_path}:{spark_sf_connector_path}") \
                .config("spark.driver.extraClassPath", f"{jdbc_path}:{spark_sf_connector_path}") \
                .getOrCreate()

        # Define Snowflake options
        snowflake_options = {
            "sfURL": f"{self.account}.snowflakecomputing.com",
            "sfDatabase": self.database,
            "sfSchema": self.schema,
            "sfWarehouse": self.warehouse,
            "sfUser": self.user,
            "sfPassword": self.password,
        }

        try:
            # Fetch the data into a Spark DataFrame using the provided query
            df_spark = self.fetch_as_spark(query)
            df_spark.show(3)

            # Write the Spark DataFrame to Snowflake
            df_spark.write \
                .format("snowflake") \
                .options(**snowflake_options) \
                .option("dbtable", table_name) \
                .mode(mode) \
                .save()

            print(f"Table {table_name} created successfully in Snowflake.")
        except Exception as e:
            print(f"Error creating table in Snowflake: {e}")
            raise
