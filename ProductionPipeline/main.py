from Start import Start
from Utils import ConfigHandler

def main():
    configHandler = ConfigHandler(config_path=r"D:\Projects\Healthcare claim predictions\ProductionPipeline\config_prod.json")
    config = configHandler.fetch_config()
    # print(config)
    autoML =Start(config=config)
    autoML.run()


main()
    