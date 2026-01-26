"""
Project ：quali_effects_eit_measurements
Directory: data/measurement_configs
File : train_configs.py
Author ：Patricia Fuchs
Date ：14.01.2026 10:19
"""
from src.data.loading_data import MeaMeta


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

def return_data_configs(name):

    if name == "2no8":
        from data.measurement_configs.config2_no8 import input_folders, probetypes, positions, vents, mealist, testcond, \
            valicond

    else:
        from data.measurement_configs.config2_no8 import input_folders, probetypes, positions, vents, mealist, testcond, \
            valicond

    return input_folders, probetypes, positions, vents, mealist, testcond, valicond
