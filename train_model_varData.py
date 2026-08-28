"""
Project ：Qualitative-Effects-of-Blood-Volume-in-EIT
Directory:
File : train_model_varData.py
Author ：Patricia Fuchs
Date ：22.07.2026 11:36
"""

import json
import tensorflow as tf
import warnings
import psutil
process=psutil.Process()
warnings.filterwarnings('ignore')
from src.nn.help_nn_fcts import *
from src.data.load_data_nn import load_mea_data_static, norm_area_inverse, split_train_test2, reshape_eit_static
import os
import time
from contextlib import redirect_stdout
import pickle
import shutil
from src.data.normalization_static import EITNormalizationStatic
from src.nn.visu_nn_results import *

from data.measurement_configs.train_configs import return_data_configs
from src.nn.training_paras import parse_arguments
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
set_global_determinism(3)
bSaveModel = True
bParseArgs = True


def get_model_path(normstrat, baxisdep, normsep):
    if baxisdep and normsep:
        d = {"none": "20260814-032815/",
            "zscore": "20260730-205418/",
            "quantilenormal": "20260824-124439/"}
        return d[normstrat]
    elif not baxisdep and normsep:
        d = {"none": "20260814-032815/",
            "zscore": "20260814-164316/",
            "quantilenormal": "20260824-161107/"}
        return d[normstrat]
    elif baxisdep and not normsep:
        d = {"none": "20260814-032815/",
            "zscore": "20260731-020015/",
            "quantilenormal": "20260805-153908/"}
        return d[normstrat]
    else:
        d = {"none": "20260814-032815/",
            "zscore": "20260804-001625/",
            "quantilenormal": "20260807-231043/"}
        return d[normstrat]


def get_probesizes_from_value(val):
    probeareas= np.linspace(1.4, 12.4, 56)
    if val== 6:
        return np.array([1.6,5.8,10.2,2.00,4.00,4.80])
    elif val==12:
        return np.array([1.6,2,3,4,4.8,5.8,7,8,9,10.2,11])
    elif val== 56:
        return probeareas
    elif val == 28:
        return probeareas[::2]
    elif val ==42:
        mask = np.ones(56, dtype=np.bool)
        mask[::4] = False
        return probeareas[mask]
        

def refill_train_data(ppfXTrain, pfYTrain, pMetaTrain, iTarget):
    N = len(ppfXTrain)
    if N == iTarget:
        return ppfXTrain, pfYTrain, pMetaTrain
    elif N > iTarget:
        return ppfXTrain[:iTarget], pfYTrain[:iTarget], pMetaTrain[:iTarget]
    W = iTarget//N
    for w in range(0, W):
        if w==W-1:
            T = iTarget%N
        else:
            T= N
        ppfXTrain = np.append(ppfXTrain, ppfXTrain[:T], axis=0)
        pfYTrain = np.append(pfYTrain, pfYTrain[:T], axis=0)
        pMetaTrain = np.append(pMetaTrain, pMetaTrain[:T], axis=0)
    return ppfXTrain, pfYTrain, pMetaTrain

    
# Loading config ----------------------------------------------------------------------------------------------------- #
config_path= "../results/nn/configs/"
data_path = "../data/"

mprefix = '../results/nn/models_integration/'
data_path2 = "../data/eit_sim_fullscan/"

# Parsing of arguments ----------------------------------------------------------------------------------------------- #
TP, parser = parse_arguments(bParseArgs)        # TrainingParameters

parser.add_argument("-sd", "--secdataconfig", type=str, help="Network path", default="none")
parser.add_argument("-ns", "--normsep", type=bool, help="Norm separately", default=False)
parser.add_argument("-vf", "--varfac", type=int , help="Variability Factor Number of Probesizes in Sim Data", default=56)
args = parser.parse_args()

secdataconfig = args.secdataconfig
normsep = args.normsep
varfac = args.varfac
normstrat = "none"


# Import Data Configurations
input_folders, probetypes, positions, vents, mealist, testcond, valicond = return_data_configs(TP.sDataConfig)

input_folders2, probetypes2, positions2, vents2, mealist2, testcond2, valicond2 = return_data_configs(secdataconfig)


with open(config_path+TP.sConfigFile, "r") as file:
    config = json.load(file)
print("Starting with config:\n"+str(config))


data_path = config["data_path"]
# Data Loading
TP.iNel = config["iNel"]


normstrat = TP.sNormtype
if not normsep:
    TP.sNormtype = "none"
    reshape = TP.bReshape
    TP.bReshape = False


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# -------------------------------------------------------------------------------------------------------------------- #
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
    normtype=TP.sNormtype,
    axisdep=TP.bAxisdep,
    bReshape = TP.bReshape,
    bShuffle = True,
    bAddDisturbances= TP.bAddDisturbances)

print("Finished loading data.")
ppfXTrain, pMetaTrain, pfYTrain, ppfXValid, pMetaValid, pfYValid = split_train_test2(ppfXTrain, pMetaTrain,valicond, pfYTrain)
print(f"{ppfXTrain.shape=},{ppfXValid.shape=},{pfYTrain.shape=},{pfYValid.shape=}")



probeareas_used = get_probesizes_from_value(varfac)
#Secondary Dataset
print(f"The amount of used probe sizes is {len(probeareas_used)}")
ppfXTrain2, pfYTrain2,pMetaTrain2, ppfXTest2, pfYTest2, pMetaTest2 = load_mea_data_static(
    data_path2,
    input_folders2,
    testcond2,
    probetypes2,
    positions2,
    mealist2,
    vents2,
    iNel = TP.iNel,
    normtype= "none",
    axisdep=TP.bAxisdep,
    bReshape = False,
    bShuffle=True,
    bAddDisturbances=TP.bAddDisturbances,
    sprescale=TP.sPrescale,
    probesizes= probeareas_used)

iLenSim = 318636
ppfXTrain2, pfYTrain2, pMetaTrain2 = refill_train_data(ppfXTrain2, pfYTrain2, pMetaTrain2, iLenSim)

if normsep and normstrat != "none":
    Norm = EITNormalizationStatic(normstrat, axisdependent=TP.bAxisdep)
    ppfXTrain2 = Norm.normalize_forward(ppfXTrain2)
    ppfXTest2 = Norm.normalize_forward(ppfXTest2, bUseFactors=True)
if TP.bReshape:
    ppfXTrain2 = reshape_eit_static(ppfXTrain2)
    ppfXTest2 = reshape_eit_static(ppfXTest2)
print("Finished loading dataset 2.")



ppfXTrain2, pMetaTrain2, pfYTrain2, ppfXValid2, pMetaValid2, pfYValid2 = split_train_test2(ppfXTrain2, pMetaTrain2,valicond2, pfYTrain2)
NtestTank = len(ppfXTest)

print(f"{ppfXTrain2.shape=},{ppfXValid2.shape=},{pfYTrain2.shape=},{pfYValid2.shape=}")
ppfXTrain = np.append(ppfXTrain, ppfXTrain2, axis=0)
pMetaTrain = np.append(pMetaTrain, pMetaTrain2, axis=0)
pfYTrain = np.append(pfYTrain, pfYTrain2, axis=0)
pfYValid = np.append(pfYValid, pfYValid2, axis=0)
pfYTest = np.append(pfYTest, pfYTest2, axis=0)
pMetaTest = np.append(pMetaTest, pMetaTest2, axis=0)
pMetaValid = np.append(pMetaValid, pMetaValid2, axis=0)
ppfXValid= np.append(ppfXValid, ppfXValid2, axis=0)
ppfXTest = np.append(ppfXTest, ppfXTest2, axis=0)


if not normsep:
    print("Start norm united")
    Norm = EITNormalizationStatic(normstrat, axisdependent=TP.bAxisdep)
    ppfXTrain = Norm.normalize_forward(ppfXTrain)
    ppfXValid = Norm.normalize_forward(ppfXValid, bUseFactors=True)      # use previous normalization factors
    ppfXTest = Norm.normalize_forward(ppfXTest, bUseFactors=True)
    if reshape:
        ppfXTrain= reshape_eit_static(ppfXTrain)
        ppfXValid= reshape_eit_static(ppfXValid)
        ppfXTest= reshape_eit_static(ppfXTest)



index = np.arange(len(ppfXTrain))
np.random.shuffle(index)
ppfXTrain, pMetaTrain, pfYTrain= ppfXTrain[index], pMetaTrain[index], pfYTrain[index]


# Start model initialization --------------------------------------------------------------------- #
nwpath = get_model_path(TP.sNormtype, TP.bAxisdep, normsep)
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

pfYTest_preds = model(ppfXTest)
del ppfXTest
gc.collect()
pfYValid_preds = model(ppfXValid)
del ppfXValid
gc.collect()

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
        text_file.write("Prescale = " + str(TP.sPrescale)+"\n")
        text_file.write("Secdataconfig = " + str(secdataconfig)+"\n")
        text_file.write("normsep = " + str(normsep)+"\n")
        text_file.write("varfac = " + str(varfac)+"\n")
        text_file.write("normstrat = " + str(normstrat)+"\n")
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

visu_results(pfYTest[:NtestTank], pfYTest_preds[:NtestTank], "TestParasTank", bSave, path, bShow=bPlotGraphics)
print("Testing finished.")
