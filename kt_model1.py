"""
Project ：Qualitative-Effects-of-Blood-Volume-in-EIT
Directory:
File : kt_model1.py
Author ：Patricia Fuchs
Date ：22.01.2026 11:36
"""
from tensorflow.keras.layers import Input, Dense, Dropout, Normalization, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import warnings
import psutil
process=psutil.Process()
warnings.filterwarnings('ignore')
import gc
import math
import keras_tuner as kt
import keras
from src.nn.help_nn_fcts import *
from src.data.load_data_nn import load_mea_data_static, norm_area_inverse, split_train_test2
import os
import time
from contextlib import redirect_stdout
import pickle
import shutil, json
from src.nn.training_paras import TrainingParas, parse_arguments
from data.measurement_configs.train_configs import return_data_configs
from src.nn.visu_nn_results import *
set_global_determinism(26)
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
bSaveModel = True
bParseArgs = True


# Loading config ------------------------------------------------------------------------------------------ #

config_path= '../nn/configs/'
data_path = "../data/"
mprefix = '../nn/models/'

# Parsing of arguments ------------------------------------------------------------------------------------------ #

TP = parse_arguments(bParseArgs)        # TrainingParameters

# Import Data Configurations
input_folders, probetypes, positions, vents, mealist, testcond, valicond = return_data_configs(TP.sDataConfig)


with open(config_path+TP.sConfigFile, "r") as file:
    config = json.load(file)
print("Starting with config:\n"+str(config))

data_path = config["data_path"]
# Data Loading
TP.iNel = config["iNel"]
# Load data
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
    bAddDisturbances=TP.bAddDisturbances,
    sprescale=TP.sPrescale
)

print("Finished loading data.")


ppfXTrain, pMetaTrain, pfYTrain, ppfXValid, pMetaValid, pfYValid = split_train_test2(ppfXTrain, pMetaTrain,valicond, pfYTrain)
print(f"{ppfXTrain.shape=},{ppfXValid.shape=},{pfYTrain.shape=},{pfYValid.shape=}")


# Start model initialization --------------------------------------------------------------------- #
def tuner_cnn_regression(hp):
    outputdim=1
    mapper_input = Input(shape=(32,32,1))
    x= mapper_input

    for i in range(1,hp.Int("num_convlayers", 2,4)):
        print(f"Init Conv layer {i=}")
        hp_filter = hp.Int("filter_" + str(i), min_value=3, max_value=6, step=1)
        hp_kernel = hp.Int("kernel_" + str(i), min_value=3, max_value=6, step=1)
        # stride dim (0,1)
        hp_strides_0 = hp.Int("stride_0_" + str(i), min_value=1, max_value=4, step=1)
        hp_strides_1 = hp.Int("stride_1_" + str(i), min_value=1, max_value=4, step=1)
        hp_activation = hp.Choice("activation_" + str(i), values=['relu','elu','tanh'])
        hp_dropout = hp.Float("dropout_" + str(i), 0,0.5, step=0.1)
        x = Conv2D(hp_filter, hp_kernel, strides=(hp_strides_0, hp_strides_1), padding="same")(x)
        hp_batchnorm = hp.Boolean("batchnorm_" + str(i), False)
        if hp_batchnorm:
            x = BatchNormalization()(x)
        #x= BatchNormalization()(x)
        x=Activation(hp_activation)(x)
        x=Dropout(hp_dropout)(x)
    x = Flatten()(x)
    for j in range(1,hp.Int("num_dense_layers", 2,3)):
        print(f"Init dense layer {j=}")
        hp_units = hp.Int("units_" + str(j), min_value=2, max_value=64, step=4)
        hp_acticonv= hp.Choice("activation_" + str(j), values=['relu','elu','tanh'])


        x=Dense(units=hp_units, activation=hp_acticonv)(x)
        hp_batchnorm = hp.Boolean("batchnormdense_" + str(j), False)
        if hp_batchnorm:
            x = BatchNormalization()(x)
        #x= BatchNormalization()(x)
        hp_dropout = hp.Float("dropoutdense_" + str(j), 0,0.5, step=0.1)
        x=Dropout(hp_dropout)(x)
    output_mapper = Dense(outputdim, activation='linear')(x)
    model = Model(inputs=mapper_input, outputs=output_mapper)
    hp_learning_rate = hp.Choice("learning_rate", values=[ 1e-3,5e-3, 1e-4])
    hp_loss = hp.Choice("loss" , values=['mae', 'mse'])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss=hp_loss, metrics=['mae', 'mse'])
    model.summary()
    return model




b ="Axisdep" if TP.bAxisdep else "NoAxisDep"
tuner =kt.Hyperband( tuner_cnn_regression,
    objective='val_loss',           # val_loss oder ["val_loss, "loss"]
    max_epochs=30,
    factor=2,
    directory="results/1"+ TP.sNWType+b+"_"+str(TP.iAugFactor),
    project_name=TP.sNormtype+str(TP.iBatchsize)+TP.sDataConfig)


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
tuner.search(ppfXTrain, pfYTrain, epochs=TP.iEpochs,batch_size=TP.iBatchsize,  validation_data=(ppfXValid, pfYValid), callbacks=[stop_early])
# Get the optimal hyperparameters

model, best_hps = get_small_nnw(tuner, maxtrials=5, size=5000)
print(best_hps.values)



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

    # save weights
    #   model.save_weights(f'{path}/weights', overwrite=True)

    # save architecture
    with open(f'{path}/model_config.json', "w") as text_file:
        text_file.write(model.to_json())

    with open(f'{path}/model_summary.txt', "w") as text_file:
        #text_file.write("Test_pig: "+str(test_pig)+"\n\n\n")
        TP.write_configs(text_file)
        text_file.write("\n\n\n")
        with redirect_stdout(text_file):
            model.summary()
        text_file.write("\nBest Results \n")
        text_file.write(str(best_hps.values))


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


# Result plotting
visu_results(pfYTest, pfYTest_preds, "TestParas", bSave, path, bShow=bPlotGraphics)
visu_metrics_combines(path,pfYTest, pfYTest_preds,pfYValid,  pfYValid_preds, "Static Probes",bShow= bPlotGraphics, bSave=bSave)


print("Testing finished.")
