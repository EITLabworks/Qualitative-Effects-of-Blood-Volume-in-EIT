"""
Project ：Qualitative-Effects-Blood-Volume-EIT
Directory: src/nn
File : help_nn_fcts.py
Author ：Patricia Fuchs
Date ：08.01.2026 16:21
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
import time
from contextlib import redirect_stdout
import pickle
import shutil

TITLE_SIZE = 24
FONT_SIZE = 18
LEGEND_SIZE = 16
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def set_global_determinism(seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    # see also:
    # https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism



# ------------------------------------------------------------------------------------------------------------- #
def plot_history(history, type, bShow=True, bSave=False, fSavePath="C:/"):
    """
    Plot Training history of the neural network
    :param history: History strucutre
    :param type: Measurement type to plot
    :param bShow: if to show the plot
    :param bSave: if to save the plot
    :param fSavePath: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(17, 9))
    plt.subplots_adjust(left=0.079, top=0.936, right=0.957, bottom=0.1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
    ax.plot(history.history[type], "b", linewidth=2)
    ax.plot(history.history['val_' + type], "r", linewidth=2)
    ax.set_title('Model ' + type, fontsize=TITLE_SIZE)
    ax.set_ylabel(type, loc="top", fontsize=FONT_SIZE)
    ax.set_xlabel('Epoch', loc="right", fontsize=FONT_SIZE)
    ax.legend(['Training', 'Validation'], loc='upper left', fontsize=LEGEND_SIZE)
    ax.grid(linewidth=0.8)
    if bSave:
        fig.savefig(fSavePath + type + "TrainingHistory.png")
    if bShow:
        plt.show()



def get_trainable_parameters(model):
    return np.sum([np.prod(v.shape.as_list()) for v in model.trainable_variables])

def get_small_nnw(tuner, maxtrials=5, size=5000):
    best_hps= tuner.get_best_hyperparameters(maxtrials)
    tp= []
    model = tuner.hypermodel.build(best_hps[0])
    train_paras= get_trainable_parameters(model)
    tp.append(train_paras)
    if train_paras > size:
        print(f"Optimal model has {train_paras} trainable parameters. That exceed max. size.")
    else:
        print(f"Optimal model has {train_paras} trainable parameters. ")
        return model, best_hps[0]
    for i in range(1,maxtrials):
        model = tuner.hypermodel.build(best_hps[i])
        train_paras = get_trainable_parameters(model)
        tp.append(train_paras)
        if train_paras < size:
            print(f"Optimal model (iteration {i} has {train_paras} trainable parameters.")
            return model, best_hps[i]
        else:
            continue
    w= np.argmin(tp)
    model = tuner.hypermodel.build(best_hps[w])
    train_paras = get_trainable_parameters(model)
    print(f"No model underneath the minimum parameter size is found. \nModel (iteration {w} has {train_paras} trainable parameters and is returned.")
    return model, best_hps[w]

