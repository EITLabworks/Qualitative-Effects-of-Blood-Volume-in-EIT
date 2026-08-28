"""
Project ：Qualitative-Effects-of-Blood-Volume-in-EIT
Directory:
File : plot_feasibility_triple.py
Author ：Patricia Fuchs
Date ：11.08.2026 11:27
"""

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
from numpy.random import seed
#seed(1)
from tensorflow.random import set_seed
#set_seed(2)
from tensorflow.keras.layers import Input, Dense, Dropout, Normalization, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
import psutil
process=psutil.Process()
warnings.filterwarnings('ignore')

from src.data.load_data_nn import load_mea_data_static, norm_area_inverse, split_train_test2
from src.nn.help_nn_fcts import *
from src.nn.visu_nn_results import *
from data.measurement_configs.train_configs import return_data_configs
from src.nn.training_paras import TrainingParas, parse_arguments
set_global_determinism(26)
bSaveModel = True
bParseArgs = True


# Loading config ------------------------------------------------------------------------------------------ #
config_path= "../results/nn/configs/"
data_path = "../data/"

mprefix = '../results/nn/models/'
savepath = "../results/nn/saving/"


# Parsing of arguments ------------------------------------------------------------------------------------------ #
TP, parser = parse_arguments(bParseArgs)
TP.sDataConfig= "7simfullpr"
secdataconfig = "8meafull"
data_path2="../data2/"
modelpath= "/20260811-161643/"
modelpath2 = "/20260422-010501/"
modelpath3 = "/20260416-150452/"
dataconfig3= "9meafull1pr"

# Import Data Configurations
input_folders, probetypes, positions, vents, mealist, testcond, valicond = return_data_configs(TP.sDataConfig)

input_folders2, probetypes2, positions2, vents2, mealist2, testcond2, valicond2 = return_data_configs(secdataconfig)


input_folders3, probetypes3, positions3, vents3, mealist3, testcond3, valicond3 = return_data_configs(dataconfig3)

#config_file = "/config_modelTEST.json"
with open(config_path+TP.sConfigFile, "r") as file:
    config = json.load(file)
print("Starting with config:\n"+str(config))


data_path=config["data_path"]
# Data Loading
TP.iNel = config["iNel"]
TP.bAxisdep=False
TP.sNormtype="zscore"
TP.bReshape=True
#TP.bAxisdep=True



os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print("Started loading data.")
ppfXTrain, pfYTrain,pMetaTrain, ppfXTest, pfYTest, pMetaTest = load_mea_data_static(
    data_path,
    input_folders,
    testcond,
    probetypes,
    positions,
    mealist,
    vents,
    iNel = TP.iNel,
    normtype= TP.sNormtype,
    axisdep=TP.bAxisdep,
    bReshape = TP.bReshape,
    bShuffle = True,
    bAddDisturbances= TP.bAddDisturbances,
    sprescale=TP.sPrescale,
    fNoiseSNR=TP.fNoiseSNR)
print("Finished loading data.")
del ppfXTrain, pMetaTrain, pfYTrain

#Secondary Dataset
ppfXTrain2, pfYTrain2,pMetaTrain2, ppfXTest2, pfYTest2, pMetaTest2 = load_mea_data_static(
    data_path2,
    input_folders2,
    testcond2,
    probetypes2,
    positions2,
    mealist2,
    vents2,
    iNel = TP.iNel,
    normtype= TP.sNormtype,
    axisdep=TP.bAxisdep,
    bReshape = TP.bReshape,
    bShuffle =True,
    bAddDisturbances= TP.bAddDisturbances,
    sprescale=TP.sPrescale)

print("Finished loading dataset 2.")

del ppfXTrain2, pfYTrain2, pMetaTrain2



#Third Dataset
ppfXTrain3, pfYTrain3,pMetaTrain3, ppfXTest3, pfYTest3, pMetaTest3= load_mea_data_static(
    data_path2,
    input_folders3,
    testcond3,
    probetypes3,
    positions3,
    mealist3,
    vents3,
    iNel = TP.iNel,
    normtype= TP.sNormtype,
    axisdep=True,
    bReshape = TP.bReshape,
    bShuffle =True,
    bAddDisturbances= TP.bAddDisturbances,
    sprescale=TP.sPrescale)

print("Finished loading dataset 3.")

del ppfXTrain3, pfYTrain3, pMetaTrain3


# Start model initialization ----------------------------------------------------------------------------------------- #
model = tf.keras.models.load_model(mprefix+modelpath+"model.keras")
model2 = tf.keras.models.load_model(mprefix+modelpath2+"model.keras")

model3 = tf.keras.models.load_model(mprefix+modelpath3+"model.keras")

# Create results ----------------------------------------------------------------------------------------------------- #
pfYTest_preds = model(ppfXTest)
del ppfXTest
gc.collect()
pfYTest2_preds= model2(ppfXTest2)
del ppfXTest2
gc.collect()

pfYTest3_preds= model3(ppfXTest3)
del ppfXTest3
gc.collect()
# Visualization ------------------------------------------------------------------------------------------------------ #
bPlotGraphics = True

pfYTest2_preds= pfYTest2_preds.numpy().flatten()
pfYTest_preds = pfYTest_preds.numpy().flatten()
pfYTest3_preds= pfYTest3_preds.numpy().flatten()

pfYTest_preds = norm_area_inverse(pfYTest_preds)
pfYTest2_preds = norm_area_inverse(pfYTest2_preds)
pfYTest3_preds = norm_area_inverse(pfYTest3_preds)

pfYTest = norm_area_inverse(pfYTest)
pfYTest2 = norm_area_inverse(pfYTest2)
pfYTest3 = norm_area_inverse(pfYTest3)

visu_results(pfYTest, pfYTest_preds, "Simulation Data", True, savepath, bShow=True)

visu_results(pfYTest2, pfYTest2_preds, "Tank Data", True, savepath, bShow=True)

visu_results(pfYTest3, pfYTest3_preds, "Tank Data Omitted Probe", True, savepath, bShow=True)

visu_results_double(pfYTest, pfYTest_preds,pfYTest2, pfYTest2_preds, "Simulation Data",  "Tank Data", True, savepath, bShow=True)
visu_results_triple_v(pfYTest, pfYTest_preds,pfYTest2, pfYTest2_preds, pfYTest3, pfYTest3_preds,"Simulation Data",  "Tank Data", "Tank Omitted Probe",True, savepath, bShow=True)
print("Testing finished.")