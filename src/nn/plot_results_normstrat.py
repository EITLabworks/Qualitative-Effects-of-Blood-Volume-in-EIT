"""
Project ：quali_effects_eit_measurements
Directory: src/nn
File : plot_results_normstrat.py
Author ：Patricia Fuchs
Date ：19.01.2026 17:08
"""
import numpy as np
import matplotlib.pyplot as plt
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


def plot_results_normstrat(trials, names, trialnames, mea, bSave=False, bShow=False, sSavepath= "C/"):
    fig, ax= plt.subplots(figsize=(15,8.5))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 18
    })
    fs=18
    plt.subplots_adjust(left=0.052, top=0.936, right=0.986, bottom=0.221)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
    #ax.set_facecolor("whitesmoke")
    x_axis= np.arange(len(names))

    colors= ["steelblue", "maroon", "goldenrod", "midnightblue"]
    for k, t in enumerate(trials):
        ax.plot(x_axis, t,marker="o", markersize=10,color=colors[k], label=trialnames[k])

    ax.set_xticks(x_axis, names, rotation=45, fontsize=fs)
    ax.legend(loc="best", fontsize=fs)
    ax.set_ylabel(mea, fontsize=fs, loc="top")
  #  ax.set_title("Results for different EIT normalization strategies", fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=fs)
    ax.tick_params(axis="both", which="minor", labelsize=fs-2)
    minor_ticks= np.linspace(0,4.5,46)
    print(minor_ticks)
    ax.set_yticks(minor_ticks, minor=True)


    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

  #  plt.tight_layout()
    if bSave:
        fig.savefig(sSavepath + mea + "EITNormResults.png")
    if bShow:
        plt.show()



def plot_results_normstrat_double(trials_mae, trials_r, names, trialnames, mea, bSave=False, bShow=False, sSavepath= "C/"):
    fig, ax= plt.subplots(nrows=2, figsize=(15,8.5))
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 20
    })
    fs=18
    plt.rcParams["font.family"] = "Times New Roman"
    plt.subplots_adjust(left=0.052, top=0.936, right=0.986, bottom=0.221, hspace=0.02)
    for axi in ax.flat:
        for axis in ["top", "bottom", "left", "right"]:
            axi.spines[axis].set_linewidth(2)
    #ax.set_facecolor("whitesmoke")
    x_axis= np.arange(len(names))

    colors= ["steelblue", "maroon", "goldenrod", "midnightblue"]
    for k, t in enumerate(trials_mae):
        ax[1].plot(x_axis, t,marker="o", markersize=10,color=colors[k], label=trialnames[k])
    for k,t in enumerate(trials_r):
        ax[0].plot(x_axis, t,marker="o", markersize=10,color=colors[k], label=trialnames[k])
    ax[0].set_xticks(x_axis, [], )
    ax[1].set_xticks(x_axis, names, rotation=45, fontsize=fs)
    minor_ticks= np.linspace(0,4.5,46)
    print(minor_ticks)
    ax[1].set_yticks(minor_ticks, minor=True)
    minor_ticks= np.linspace(1,3.5,26)
    print(minor_ticks)
    ax[0].set_yticks(minor_ticks, minor=True)
    ax[0].set_yticks([2,3])
    for i in range(2):
        ax[i].legend(loc="upper right", fontsize=fs)
        ax[i].tick_params(axis="both", which="major", labelsize=fs)
        ax[i].tick_params(axis="both", which="minor", labelsize=-2)
        ax[i].grid(which='both')
        ax[i].grid(which='minor', alpha=0.2)
        ax[i].grid(which='major', alpha=0.5)

    ax[1].set_ylabel("MAE", fontsize=fs, loc="top")
    ax[0].set_ylabel("STD of Error", fontsize=fs, loc="top")
   # ax[0].set_title("Results for different EIT Normalization Strategies", fontsize=22)


  #  plt.tight_layout()
    if bSave:
        fig.savefig(sSavepath + mea + "EITNormResultsDouble.png")
    if bShow:
        plt.show()

