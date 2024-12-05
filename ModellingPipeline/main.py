from AutoML import AutoML
from Utils import ConfigHandler

def main():
    configHandler = ConfigHandler(config_path=r"D:\Projects\Healthcare claim predictions\config.json")
    config = configHandler.fetch_config()
    # print(config)
    autoML =AutoML(config=config)
    autoML.run()


main()
    