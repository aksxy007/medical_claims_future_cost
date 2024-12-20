from AutoML import AutoML
from Utils import ConfigHandler

def main():
    configHandler = ConfigHandler(config_path=r"D:\Projects\Healthcare claim predictions\ModellingPipeline\config\config.json")
    config = configHandler.fetch_config()
    # print(config)
    autoML =AutoML(config=config)
    autoML.run()

main()
    
    
#  "lightgbm": {
#             "params": {
#                 "n_estimators": [100, 200],
#                 "max_depth": [-1, 10],
#                 "learning_rate": [0.01, 0.05]
#             },
#             "default":{
#             "n_estimators": 100,
#             "max_depth": -1,
#             "learning_rate": 0.05
#             }
#         }

# "categorical_columns": ["class","cap-shape","cap-surface","cap-color","bruises","odor","gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","veil-type","veil-color","ring-number","ring-type","spore-print-color","habitat","population"]