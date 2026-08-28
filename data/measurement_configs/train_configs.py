"""
Project ：quali_effects_eit_measurements
Directory: data/measurement_configs
File : train_configs.py
Author ：Patricia Fuchs
Date ：14.01.2026 10:19
"""


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

def return_data_configs(name):
    if name == "none" or name == "standard" or name == "config1":
        from data.measurement_configs.config1 import input_folders, probetypes, positions, vents, mealist, testcond, \
            valicond

    elif name == "2no8":
        from data.measurement_configs.config2_no8 import input_folders, probetypes, positions, vents, mealist, testcond, \
            valicond

    elif name == "3no0":
        from data.measurement_configs.config3_no0 import input_folders, probetypes, positions, vents, mealist, testcond, \
            valicond

    elif name == "4noballon":
        from data.measurement_configs.config4_noballon import input_folders, probetypes, positions, vents, mealist, \
            testcond, valicond
    elif name == "5no8sim":
        from data.measurement_configs.config5_no8sim import input_folders, probetypes, positions, vents, mealist, \
            testcond, valicond

    elif name == "6no8sim2":
        from data.measurement_configs.config6_no8sim2 import input_folders, probetypes, positions, vents, mealist, \
            testcond, valicond
    elif name == "7simfull":
        from data.measurement_configs.config7_simfull import input_folders, probetypes, positions, vents, mealist, \
            testcond, valicond
    elif name == "7simfullpr":
        from data.measurement_configs.config7_simfull_pr import input_folders, probetypes, positions, vents, mealist, \
            testcond, valicond
    elif name == "8meafull":
        from data.measurement_configs.config8_meafull import input_folders, probetypes, positions, vents, mealist, \
            testcond, valicond
    elif name == "8meafull20":
        from data.measurement_configs.config8_meafull20 import input_folders, probetypes, positions, vents, mealist, \
            testcond, valicond    
    elif name == "9meafullcyl":
        from data.measurement_configs.config9_meafull_cyl import input_folders, probetypes, positions, vents, mealist, \
            testcond, valicond
    elif name == "9meafull1pr":
        from data.measurement_configs.config9_meafull_1pr import input_folders, probetypes, positions, vents, mealist, \
            testcond, valicond
    else:
        from data.measurement_configs.config1 import input_folders, probetypes, positions, vents, mealist, testcond, \
            valicond

    return input_folders, probetypes, positions, vents, mealist, testcond, valicond
