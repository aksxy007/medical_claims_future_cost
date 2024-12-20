from datetime import datetime,date

from SnowflakeConnector import SFConnector
class Queries:
    
    def __init__(self,config):
        self.config=config
        self.queries = self.config['snowflake_queries']['queries']
        self.isSequential = self.config['snowflake_queries']['sequential']  
        self.sf_connector = SFConnector(
            account=self.config['snowflake_connection']['account'],
            user=self.config['snowflake_connection']['user'],
            password=self.config['snowflake_connection']['password'],
            warehouse=self.config['snowflake_connection']['warehouse'],
            database=self.config['snowflake_connection']['database'],
            schema=self.config['snowflake_connection']['schema']
        )
        
    def run_queries(self):
        if self.isSequential:
            final_output_table=None
            for i in range(0,len(self.queries)):
                current_query = self.queries[i][f"{i+1}"]
                current_day = datetime.now().day
                current_month = datetime.now().month
                current_date = str(2010)+str(current_month)+str(current_day)
                print("Current Date",current_date)
                input_table = current_query['input_table']
                output_table =current_query['output_table']
                query = current_query['query']
                query=query.format(current_date=current_date,input_table=input_table)
                # query= "CREATE OR REPLACE TABLE {output_tale} AS SELECT *,{CURRENT_DATE} AS CUTOFF_DATE FROM {PROD_TABLE_NAME} WHERE COUNT(CLM_ID)>=1 AND CLM_FROM_DT >= {CURRENT_DATE} AND CLM_FROM_DT <= TO_NUMBER(TO_CHAR(DATEADD(DAY, 29, TO_DATE({CURRENT_DATE}, 'YYYYMMDD')),'YYYYMMDD'));".format(CURRENT_DATE=current_date,PROD_TABLE_NAME=prod_table_name,output_table=output_table)
                self.run_query(qry=query,table_name=output_table)
                print(f"{i+1} query processed succesfully")
                final_output_table = output_table
                # return query,output_table
            if final_output_table is not None:
                final_query = self.final_query(cutoff_table=final_output_table)
                prediction_table_name = self.config['prediction_table_name']
                self.run_query(qry=final_query,table_name=prediction_table_name)
            
    
    
    def final_query(self,cutoff_table):
        icd9_columns = [f"ICD9_DGNS_CD_{i}" for i in range(1, 11)]  # ICD9_DGNS_CD_1 to ICD9_DGNS_CD_10
        icd9_proc_columns = [f"ICD9_PRCDR_CD_{i}" for i in range(1, 7)]  # ICD9_PRCDR_CD_1 to ICD9_PRCDR_CD_6
        hcpcs_columns = [f"HCPCS_CD_{i}" for i in range(1, 46)]  # HCPCS_CD_1 to HCPCS_CD_45
        other_columns = ["NCH_BENE_PTB_DDCTBL_AMT", "NCH_BENE_PTB_COINSRNC_AMT"]  
        all_columns = icd9_columns + icd9_proc_columns + hcpcs_columns + other_columns

        # Create the dynamic SQL for COALESCE(MAX()) for all the columns
        coalesce_columns = [
            f"MAX(COALESCE({col}, '0')) AS {col}" for col in all_columns
        ]

        # Create the dynamic SQL for the SELECT part of the query
        select_clause = ",\n    ".join(coalesce_columns)


        query = f"""
        SELECT 
            DESYNPUF_ID, 
            CUTOFF_DATE, 
            {select_clause}  -- Dynamically generated columns with COALESCE
            -- FUTURE_COST
        FROM 
            {cutoff_table}

        GROUP BY DESYNPUF_ID, CUTOFF_DATE;
       

        """

        print(query)
        
    
    def run_query(self,qry,table_name):
        self.sf_connector.create_table_from_spark(qry,table_name=table_name)

        
        
        
