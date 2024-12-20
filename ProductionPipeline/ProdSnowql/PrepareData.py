
from ProdSnowql.Queries import Queries


class PrepareData:
    
    def __init__(self,config):
        self.config = config
        self.queries = Queries(self.config)
        
        
    def run(self):
        # run intital query
        self.queries.run_queries()

        