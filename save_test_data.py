"""
Project ：quali_effects_eit_measurements
Directory: 
File : save_test_data.py
Author ：Patricia Fuchs
Date ：25.11.2026 15:04
"""
import warnings
import psutil
process=psutil.Process()
warnings.filterwarnings('ignore')
import gc
from src.nn.help_nn_fcts import *
from src.data.load_data_nn import load_mea_data_static, norm_area_inverse
import os
from src.nn.training_paras import TrainingParas, parse_arguments
from data.measurement_configs.train_configs import return_data_configs
set_global_determinism(26)
import tensorflow as tf
import json
bSaveModel = True
bParseArgs = True
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# Loading config ------------------------------------------------------------------------------------------ #
config_path= "/results/nn/configs/"
data_path = "/Data/"

mprefix = '/results/nn/models/'
savepath = "/results/nn/results/paper/"

# Parsing of arguments ------------------------------------------------------------------------------------------ #


TP, parser = parse_arguments(bParseArgs)
parser.add_argument("-mp", "--modelpath", type=str, help="Model to load path", default="none")
parser.add_argument("-sn", "--savename", type=str, help="savename", default="none")

# TrainingParameters
args = parser.parse_args()
modelpath = args.modelpath

savename= args.savename

# Import Data Configurations
input_folders, probetypes, positions, vents, mealist, testcond, valicond = return_data_configs(TP.sDataConfig)


with open(config_path+TP.sConfigFile, "r") as file:
    config = json.load(file)
print("Starting with config:\n"+str(config))

# Data Loading
TP.iNel = config["iNel"]
TP.iDim = int(TP.iNel*TP.iNel)
TP.fR = config["fR"]
TP.iInjSkip = config["iInjSkip"]
TP.sMeaPattern = config["sMeaPattern"]


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


# Start model initialization --------------------------------------------------------------------- #
model = tf.keras.models.load_model(mprefix+modelpath+"model.keras")

# Create results ----------------------------------------------------------------------------------------------------- #
pfYTest_preds = model(ppfXTest)
del ppfXTest
gc.collect()
gc.collect()


# Saving ----------------------------------------------------------------------------- #
pfYTest_preds = pfYTest_preds.numpy().flatten()
pfYTest_preds = norm_area_inverse(pfYTest_preds)
pfYTest = norm_area_inverse(pfYTest)

np.save(savepath+savename+".npy", pfYTest)
np.save(savepath+savename+"_preds.npy", pfYTest_preds)
