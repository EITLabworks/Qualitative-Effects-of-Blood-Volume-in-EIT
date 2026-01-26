"""
Project ：Qualitative-Effects-Blood-Volume-EIT
Directory:
File : eval_model1.py
Author ：Patricia Fuchs
Date ：22.01.2026 11:36
"""

import tensorflow as tf
import warnings
import psutil
process=psutil.Process()
warnings.filterwarnings('ignore')
import gc
import keras_tuner as kt
import keras
from src.nn.help_nn_fcts import *
from src.data.load_data_nn import load_mea_data_static, norm_area_inverse, split_train_test2, apply_dataaugmentation
from src.nn.training_paras import TrainingParas, parse_arguments
from src.nn.plotting import *
from data.measurement_configs.train_configs import return_data_configs
import json
set_global_determinism(26)
bSaveModel = True
bParseArgs = False
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# Loading config ------------------------------------------------------------------------------------------ #
config_path= '../nn/configs/'
data_path = "../data/"
mprefix = '../nn/models/'


modelpath = "20260122-024634/"
# Parsing of arguments ------------------------------------------------------------------------------------------ #

TP = parse_arguments(bParseArgs)        # TrainingParameters
TP.sDataConfig="2no8"
# Import Data Configurations
input_folders, probetypes, positions, vents, mealist, testcond, valicond = return_data_configs(TP.sDataConfig)

TP.iAugFactor=0
TP.sNormtype="minmax"
TP.bAxisdep=False
TP.bReshape=True

# Initialization
with open(config_path+TP.sConfigFile, "r") as file:
    config = json.load(file)
print("Starting with config:\n"+str(config))

# Data Loading
TP.iNel = config["iNel"]
TP.iDim = int(TP.iNel*TP.iNel)
TP.fR = config["fR"]
TP.iInjSkip = config["iInjSkip"]
TP.sMeaPattern = config["sMeaPattern"]

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
    fR = TP.fR,
    injskip=TP.iInjSkip,
    meapatt=TP.sMeaPattern,
    posfactor=TP.sPosfactor,
    normtype= TP.sNormtype,
    axisdep=TP.bAxisdep,
    bReshape = TP.bReshape,
    bShuffle = True,
    bRealPart=TP.bRealPart)

print("Finished loading data.")

ppfXTrain, pMetaTrain, pfYTrain, ppfXValid, pMetaValid, pfYValid = split_train_test2(ppfXTrain, pMetaTrain,valicond, pfYTrain)

print(f"{ppfXTrain.shape=},{ppfXValid.shape=},{pfYTrain.shape=},{pfYValid.shape=}")


if TP.iAugFactor!=0:
    ppfXTrain, pMetaTrain, pfYTrain = apply_dataaugmentation(len(ppfXTrain), TP.iAugFactor, ppfXTrain, pMetaTrain, pfYTrain, TP.bReshape)



# Start model initialization --------------------------------------------------------------------- #
model = tf.keras.models.load_model(mprefix+modelpath+"model.keras")
model.summary()
print("Loading model finished finished")


pfYTrain_preds= model(ppfXTrain)

# delete training stacks
del ppfXTrain
# load test data



pfYTest_preds = model(ppfXTest)
del ppfXTest
gc.collect()
pfYValid_preds = model(ppfXValid)
del ppfXValid
gc.collect()
print("Memory used " + str(process.memory_info().rss/1024))

# Save model ----------------------------------------------------------------------------- #
path = mprefix


bPlotGraphics=True


pfYValid_preds = pfYValid_preds.numpy().flatten()
pfYTest_preds = pfYTest_preds.numpy().flatten()
pfYTrain_preds = pfYTrain_preds.numpy().flatten()

pfYTest_preds = norm_area_inverse(pfYTest_preds)
pfYValid_preds = norm_area_inverse(pfYValid_preds)
pfYTrain_preds = norm_area_inverse(pfYTrain_preds)
pfYTrain = norm_area_inverse(pfYTrain)

pfYTest = norm_area_inverse(pfYTest)
pfYValid = norm_area_inverse(pfYValid)



# Set global font settings
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 18
})


visu_metrics_heat2(pfYTest, pfYTest_preds,pfYTrain,  pfYTrain_preds, "Training- Test Static Probes",pMetaTest, pMetaTrain, bShow=bPlotGraphics, bSave=bSave)
visu_metrics_heat(pfYTest, pfYTest_preds,pfYTrain,  pfYTrain_preds, "Training- Test Static Probes",pMetaTest, pMetaTrain, bShow=bPlotGraphics, bSave=bSave)

visu_metrics_combines(pfYTest, pfYTest_preds,pfYValid,  pfYValid_preds, "Static Probes",bShow= bPlotGraphics, bSave=bSave)
visu_metrics_bypos(pfYTest, pfYTest_preds,pfYValid,  pfYValid_preds, "Static Probes",pMetaTest, pMetaValid, bShow=bPlotGraphics, bSave=bSave)
visu_metrics_bypos(pfYTest, pfYTest_preds,pfYTrain,  pfYTrain_preds, "Training- Test Static Probes",pMetaTest, pMetaTrain, bShow=bPlotGraphics, bSave=bSave)

print("Testing finished.")


