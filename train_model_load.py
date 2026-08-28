"""
Project ：Qualitative-Effects-of-Blood-Volume-in-EIT
Directory:
File : train_model_load.py
Author ：Patricia Fuchs
Date ：22.01.2026 11:36
"""

import json
import tensorflow as tf
import warnings
import psutil
process=psutil.Process()
warnings.filterwarnings('ignore')
from src.nn.help_nn_fcts import *
from src.data.load_data_nn import load_mea_data_static, norm_area_inverse, split_train_test2, get_model_path
import os
import time
from contextlib import redirect_stdout
import pickle
import shutil
from src.nn.visu_nn_results import *
from data.measurement_configs.train_configs import return_data_configs
from src.nn.training_paras import TrainingParas, parse_arguments
set_global_determinism(42)
bSaveModel = True
bParseArgs = True
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# Loading config ------------------------------------------------------------------------------------------ #
config_path= "../results/nn/configs/"
data_path = "../data/"

mprefix = '../results/nn/models/'

# Parsing of arguments ------------------------------------------------------------------------------------------ #

TP, parser = parse_arguments(bParseArgs)        # TrainingParameters

# Import Data Configurations
input_folders, probetypes, positions, vents, mealist, testcond, valicond = return_data_configs(TP.sDataConfig)


#config_file = "/config_modelTEST.json"
with open(config_path+TP.sConfigFile, "r") as file:
    config = json.load(file)
print("Starting with config:\n"+str(config))


data_path = config["data_path"]
# Data Loading
TP.iNel = config["iNel"]


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
ppfXTrain, pMetaTrain, pfYTrain, ppfXValid, pMetaValid, pfYValid = split_train_test2(ppfXTrain, pMetaTrain,valicond, pfYTrain)
print(f"{ppfXTrain.shape=},{ppfXValid.shape=},{pfYTrain.shape=},{pfYValid.shape=}")



# Start model initialization --------------------------------------------------------------------- #
nwpath = get_model_path(TP.sNormtype, TP.bAxisdep)
model_architecture = tf.keras.models.load_model(mprefix+nwpath+"model.keras")
# Create a new model with the same architecture but random weights
model = tf.keras.models.clone_model(model_architecture)



# Model Training----------------------------------------------------------------------------- #
history = model.fit(
    ppfXTrain,
    pfYTrain,
    validation_data=(ppfXValid, pfYValid),
    epochs=TP.iEpochs,
    batch_size=TP.iBatchsize,

)
print("Training finished")

pfYTrain_preds= model(ppfXTrain)
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
if bSaveModel:

    timestr = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(mprefix, timestr)
    os.mkdir(path)

    # save architecture
    with open(f'{path}/model_config.json', "w") as text_file:
        text_file.write(model.to_json())

    with open(f'{path}/model_summary.txt', "w") as text_file:
        TP.write_configs(text_file)
        text_file.write("\n\n\n")
        with redirect_stdout(text_file):
            model.summary()

    with open(f'{path}/history.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    model.save(f'{path}/model.keras')
    path = path + "/"
    shutil.copyfile(config_path + TP.sConfigFile, path + TP.sConfigFile)


# Visualization ----------------------------------------------------------------------------- #
bPlotGraphics = True
bSave=True

plot_history(history, "mse", bSave=bPlotGraphics,fSavePath=path)
plot_history(history, "loss", bSave=bPlotGraphics,fSavePath=path)

pfYValid_preds = pfYValid_preds.numpy().flatten()
pfYTest_preds = pfYTest_preds.numpy().flatten()
pfYTrain_preds = pfYTrain_preds.numpy().flatten()

pfYTest_preds = norm_area_inverse(pfYTest_preds)
pfYValid_preds = norm_area_inverse(pfYValid_preds)
pfYTrain_preds = norm_area_inverse(pfYTrain_preds)
pfYTrain = norm_area_inverse(pfYTrain)

pfYTest = norm_area_inverse(pfYTest)
pfYValid = norm_area_inverse(pfYValid)


visu_results(pfYTest, pfYTest_preds, "TestParas", bSave, path, bShow=bPlotGraphics)
visu_boxplot(path, pfYTest, pfYTest_preds,pfYValid,  pfYValid_preds, "Static Probes", bSave, bPlotGraphics)
visu_metrics_combines(path, pfYTest, pfYTest_preds,pfYValid,  pfYValid_preds, "Static Probes", bSave, bPlotGraphics)
print("Testing finished.")
