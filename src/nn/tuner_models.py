"""
Project ：quali_effects_eit_measurements
Directory: src/nn
File : tuner_models.py.py
Author ：Patricia Fuchs
Date ：08.01.2026 17:04
"""

from tensorflow.keras.layers import Input, Dense, Dropout, Normalization, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import os
import gc
import math
import keras_tuner as kt
import keras
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

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

def tuner_cnn_regression2(hp):
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
        x = MaxPooling2D(2,2)(x)
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




models_kt= {
           "cnn_reg":tuner_cnn_regression,
            "cnn_reg2":tuner_cnn_regression2,}