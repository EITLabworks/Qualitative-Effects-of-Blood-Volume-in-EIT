"""
Project ：quali_effects_eit_measurements
Directory: src/nn
File : visu_nn_results.py
Author ：Patricia Fuchs
Date ：27.08.2026 14:48
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import gc
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

def visu_metrics_heat2(y_true_test, y_predict_test, y_true_vali, y_pred_vali, name, pMetaTest, pMetaVali, bSave=True,
                       bShow=False, sSavePath="C/"):
    """
    Visualize certain metrics
    """

    # plt.rcParams["font.family"] = "Times New Roman"
    fig, axs = plt.subplots(figsize=(20, 7.1))

    plt.subplots_adjust(left=0.079, top=0.936, right=0.957, bottom=0.08)
    for axis in ["top", "bottom", "left", "right"]:
        axs.spines[axis].set_linewidth(2)
    axs.set_facecolor("whitesmoke")

    scat = axs.scatter(y_true_vali, y_pred_vali, alpha=0.2, s=25, c="midnightblue", label="Training")
    scattest = axs.scatter(y_true_test, y_predict_test, alpha=0.2, s=25, c="goldenrod", label="Test")


    limits = [np.min(y_true_vali) - 5, np.max(y_true_vali) + 5]
    line = np.arange(limits[0], limits[1], 1)
    axs.plot(line, line, color="black", linewidth=2, linestyle="--")
    axs.legend(fontsize=20)


    axs.set_xlabel("Real Probe Size [cm$^2$]", fontsize=20, loc="right")
    axs.set_ylabel("Estimated Probe Size [cm$^2$]", fontsize=20, loc="top")
    plt.tight_layout()
    if bSave:
        fig.savefig(sSavePath + name + "PredictionErrorHeatmap.png")
    if bShow:
        plt.show()


# -------------------------------------- #
def visu_results(y_true, y_predict, name, bSave, savepath, bShow=False):
    """
    Visualize certain metrics
    """
    fs = 20
    plt.rcParams["font.family"] = "Times New Roman"
    fig, axs = plt.subplots(figsize=(6, 6))
    axs.tick_params(axis='both', which='major', labelsize=fs)
    axs.tick_params(axis='both', which='minor', labelsize=14)

    axs.scatter(y_true, y_predict, alpha=0.2, s=25, color="midnightblue", label=name)
    limits = [0, 12]
    line = np.arange(limits[0], limits[1], 1)
    axs.plot(line, line, color="black", linewidth=2, linestyle="--")

    plt.subplots_adjust(left=0.079, top=0.936, right=0.957, bottom=0.08)
    for axis in ["top", "bottom", "left", "right"]:
        axs.spines[axis].set_linewidth(2)
    # axs.set_facecolor("whitesmoke")

    axs.set_xlabel("Real Probe Size [cm$^2$]", fontsize=fs, loc="right")
    axs.set_ylabel("Estimated Probe Size [cm$^2$]", fontsize=fs, loc="top")
    axs.legend(fontsize=fs)
    # axs.set_ticks(fontsize=20)

    plt.tight_layout()
    if bSave:
        fig.savefig(savepath + name + "PredictionError.png")
        fig.savefig(savepath + name + "PredictionError.svg")
        fig.savefig(savepath + name + "PredictionError.pdf")
    if bShow:
        plt.show()
    gc.collect()


# -------------------------------------- #
def visu_results_double(y_true, y_predict, y_true2, y_predict2, name, name2, bSave, savepath, bShow=False):
    """
    Visualize certain metrics
    """
    fs = 20
    plt.rcParams["font.family"] = "Times New Roman"
    fig, axs = plt.subplots(nrows=2, figsize=(6.5, 10))

    axs[0].scatter(y_true, y_predict, alpha=0.2, s=25, color="midnightblue", label=name)
    limits = [0, 12]
    line = np.arange(limits[0], limits[1], 1)
    axs[0].plot(line, line, color="black", linewidth=2, linestyle="--")

    plt.subplots_adjust(left=0.129, top=0.936, right=0.957, bottom=0.08, )
    for axi in axs.flat:
        for axis in ["top", "bottom", "left", "right"]:
            axi.spines[axis].set_linewidth(2)

    axs[1].scatter(y_true2, y_predict2, alpha=0.2, s=25, color="maroon", label=name2)
    axs[1].plot(line, line, color="black", linewidth=2, linestyle="--")

    #   axs[1].set_title(name2, size=20)
    # axs[0].set_xlabel("Real Probe Size [cm$^2$]", fontsize=20, loc="right")
    axs[0].set_ylabel("Estimated Probe Size [cm$^2$]", fontsize=fs, loc="top")
    axs[1].set_xlabel("Real Probe Size [cm$^2$]", fontsize=fs, loc="right")
    axs[1].set_ylabel("Estimated Probe Size [cm$^2$]", fontsize=fs, loc="top")
    axs[0].legend(fontsize=fs, loc="upper left")
    axs[1].legend(fontsize=fs, loc="upper left")
    axs[0].tick_params(axis='both', which='major', labelsize=fs)
    axs[0].tick_params(axis='both', which='minor', labelsize=14)
    axs[1].tick_params(axis='both', which='major', labelsize=fs)
    axs[1].tick_params(axis='both', which='minor', labelsize=14)
    # fig.suptitle("Plotting predictions for " + name, size=24)
    plt.tight_layout()
    if bSave:
        fig.savefig(savepath + name + name2 + "PredictionError.png")
        fig.savefig(savepath + name + name2 + "PredictionError.pdf")
        fig.savefig(savepath + name + name2 + "PredictionError.svg")
    if bShow:
        plt.show()
    gc.collect()


# -------------------------------------- #
def visu_results_triple_v(y_true, y_predict, y_true2, y_predict2, y_true3, y_predict3, name, name2, name3, bSave,
                          savepath, bShow=False):
    """
    Visualize certain metrics
    """
    fs = 18
    plt.rcParams["font.family"] = "Times New Roman"
    fig, axs = plt.subplots(nrows=3, figsize=(6.5, 15.5))
    axs[0].scatter(y_true, y_predict, alpha=0.2, s=25, color="midnightblue", label=name)
    limits = [0, 12]
    line = np.arange(limits[0], limits[1], 1)
    for i in range(3):
        axs[i].plot(line, line, color="black", linewidth=2, linestyle="--")
        axs[i].tick_params(axis='both', which='major', labelsize=fs)
        axs[i].tick_params(axis='both', which='minor', labelsize=14)
        #   axs[i].set_facecolor("whitesmoke")
        axs[i].set_ylabel("Estimated Probe Size [cm$^2$]", fontsize=fs, loc="top")
    axs[2].set_xlabel("Real Probe Size [cm$^2$]", fontsize=fs, loc="right")
    #   axs[i].set_title(names[i], fontsize=20)
    # axs[2].set_ylabel("Estimated Probe Size [cm$^2$]", fontsize=fs, loc="top")

    plt.subplots_adjust(left=0.079, top=0.936, right=0.957, bottom=0.08)
    for axi in axs.flat:
        for axis in ["top", "bottom", "left", "right"]:
            axi.spines[axis].set_linewidth(2)

    axs[1].scatter(y_true2, y_predict2, alpha=0.2, s=25, color="maroon", label=name2)
    axs[2].scatter(y_true3, y_predict3, alpha=0.2, s=25, color="goldenrod", label=name3)
    for i in range(3):
        axs[i].legend(fontsize=fs)

    # axs[0].set_xlabel("Real Probe Size [cm$^2$]", fontsize=20, loc="right")
    # axs[0].set_ylabel("Estimated Probe Size [cm$^2$]", fontsize=20, loc="top")

    # axs[1].set_ylabel("Estimated Probe Size [cm$^2$]", fontsize=20, loc="top")
    # fig.suptitle("Plotting predictions for " + name, size=24)
    plt.tight_layout()
    if bSave:
        fig.savefig(savepath + name + name2 + name3 + "_v.png")
        fig.savefig(savepath + name + name2 + name3 + "_v.svg")
        fig.savefig(savepath + name + name2 + name3 + "_v.pdf")
    if bShow:
        plt.show()
    gc.collect()


# -------------------------------------- #
def visu_boxplot(savepath, y_test, y_predict_test, y_vali, y_predict_vali, title, bShow=True, bSave=False):
    """
    Plots boxplots for the error of estimated curves
    :param title: Plot title
    :param y_test: Test Data Original parametric aortic pressure curves [Num segments L x individual segment len] or only paras
    :param y_predict_test: Test data Estimated aortic pressure curves as list of segments [Num segments L x individual segment len] or only paras
    :param y_vali: Validation Data Original parametric aortic pressure curves [Num segments L x individual segment len] or only paras
    :param y_predict_vali: Validation data Estimated aortic pressure curves as list of segments [Num segments L x individual segment len] or only paras
    :param recon_given: If the reconstructed curves are already given /or if reconstruction should be performed
    :param bShow: if to show the plot
    :param bSave: if to save the plot
    :param fSavePath: Path to save the plot
    """

    error_test = y_test - y_predict_test
    error_vali = y_vali - y_predict_vali

    fontSize = 22
    title_Size = 24

    plt.rc('xtick', labelsize=fontSize)
    plt.rc('ytick', labelsize=fontSize)
    fig, ax = plt.subplots(figsize=(17, 12))
    # Creating plot
    ax.set_facecolor("whitesmoke")
    bp1 = ax.boxplot(error_test, positions=[1.5], sym='', widths=0.3, meanline=False,
                     showmeans=True, labels=["Testing"], patch_artist=True,
                     boxprops=dict(facecolor='steelblue', color='black', linewidth=3),
                     medianprops=dict(color='maroon', linewidth=3))
    bp2 = ax.boxplot(error_vali, positions=[0.5], sym='', widths=0.3, meanline=False,
                     showmeans=True, labels=["Validation"], patch_artist=True,
                     boxprops=dict(facecolor='lightblue', color='black', linewidth=3),
                     medianprops=dict(color='maroon', linewidth=3))
    ax.plot([], c='lightblue', label="Error Validation Data")
    ax.plot([], c='steelblue', label="Error Testing Data")

    ax.grid(linewidth=0.4)

    ax.set_title(title, size=title_Size)
    ax.set_ylabel("Error", size=fontSize, loc="top")

    plt.subplots_adjust(left=0.079, top=0.936, right=0.957, bottom=0.1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)

    if bSave:
        fig.savefig(savepath + title + ".png")
    if bShow:
        plt.show()


def visu_metrics_combines(savepath, y_true_test, y_predict_test, y_true_vali, y_pred_vali, name, bSave=True,
                          bShow=False):
    """
    Visualize certain metrics
    """
    # plt.rcParams["font.family"] = "Times New Roman"
    fig, axs = plt.subplots(ncols=1, figsize=(20, 7.1))
    axs.scatter(y_true_vali, y_pred_vali, alpha=0.2, s=25, color="steelblue", label="Validation")

    plt.subplots_adjust(left=0.079, top=0.936, right=0.957, bottom=0.08)
    for axis in ["top", "bottom", "left", "right"]:
        axs.spines[axis].set_linewidth(2)
    axs.set_facecolor("whitesmoke")

    limits = [np.min(y_true_vali) - 5, np.max(y_true_vali) + 5]
    line = np.arange(limits[0], limits[1], 1)
    axs.plot(line, line, color="black", linewidth=2, linestyle="--")

    #  fig.suptitle("Actual vs.Predicted values", fonsize=20)
    axs.scatter(y_true_test, y_predict_test, alpha=0.2, s=25, color="maroon", label="Testing")
    limits = [np.min(y_true_test) - 5, np.max(y_true_test) + 5]
    line = np.arange(limits[0], limits[1], 1)
    axs.plot(line, line, color="black", linewidth=2, linestyle="--")
    axs.set_xlabel("Real Probe Size [cm^2]", fontsize=20, loc="right")
    axs.set_ylabel("Estimated Probe Size [cm^2]", fontsize=20, loc="top")
    # axs[1].set_ylabel("Residuals (actual - predicted)", fontsize=20, loc="top")
    #    fig.suptitle("Plotting predictions for " + name, size=24)
    plt.tight_layout()
    if bSave:
        fig.savefig(savepath + name + "PredictionErrorCombine.png")
    if bShow:
        plt.show()
