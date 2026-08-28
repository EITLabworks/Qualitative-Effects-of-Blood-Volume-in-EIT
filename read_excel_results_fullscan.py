"""
Project ：Qualitative-Effects-of-Blood-Volume-in-EIT
Directory: src
File : read_excel_results_fullscan.py
Author ：Patricia Fuchs
Date ：22.01.2026 09:29
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
def plot_results_normstrat(trials, names, trialnames, mea, bSave=False, bShow=False, sSavepath= "C/"):
    fig, ax= plt.subplots(figsize=(15,6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 18
    })
    fs=18
    plt.subplots_adjust(left=0.052, top=0.98, right=0.986, bottom=0.221)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
    #ax.set_facecolor("whitesmoke")
    x_axis= np.arange(len(names))

    colors= ["steelblue", "maroon", "goldenrod", "midnightblue"]
    colors= ["midnightblue", "maroon", "steelblue", "indianred"]
    for k, t in enumerate(trials):
        ax.plot(x_axis, t,marker="o", markersize=10,color=colors[k], label=trialnames[k])

    ax.set_xticks(x_axis, names, rotation=45, fontsize=fs)
    ax.set_ylim([-0.3,2.5])
    ax.legend(loc="best", fontsize=fs)
    ax.set_ylabel(mea, fontsize=fs, loc="top")
  #  ax.set_title("Results for different EIT normalization strategies", fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=fs)
    ax.tick_params(axis="both", which="minor", labelsize=fs-2)
  #  minor_ticks= np.linspace(0,4.5,46)
    minor_ticks= np.linspace(-0.2,2.5,28)
    print(minor_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    major_ticks= np.linspace(0,2.5,6)
    ax.set_yticks(major_ticks)


    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.7)

    if bSave:
        fig.savefig(sSavepath + mea + "EITNormResults.png")
    if bShow:
        plt.show()


# -------------------------------------------------------------------------------------------------------------------- #
plt.rcParams["font.family"] = "Times New Roman"

df =pd.read_excel("/results/results_compare_fullscan_shifted.xlsx")
x=5

iMeas= 14
iStartidx= 12
normstrat_names= ["None", "z-Score", "Min-Max", "Max-Absolute", "Decimal", "Median", "Med-MAD", "VSS", "Sigmoid", "Tanh", "Scaled Tanh \n", "QT Uniform", "QT Normal", "Power \nTransform"]

names=  [ "None","Zscore", "MinMax", "MaxAbsolute", "Decimal", "Median", "MedMad", "VSS", "Sigmoid", "Tanh", "HyperbolicTangent", "QT uniform", "QT normal", "Powertransform"]


trials_to_rows={"cnnreg_axisdep_mae": 0,       # 28
                "cnnreg_noaxisdep_mae":5,      # 33
                "cnnreg_axisdep_dist_mae":10,
                "cnnreg_noaxisdep_dist_mae":15,
                "cnnreg_axisdep_std": 1,
                "cnnreg_noaxisdep_std":6,
                "cnnreg_axisdep_dist_std":11,
                "cnnreg_noaxisdep_dist_std":16,
                }

maes= []
stds = []
nwtypes= ["_cnnreg"]
bStandard= True
if bStandard:
    for n in ["cnnreg_axisdep","cnnreg_noaxisdep","cnnreg_axisdep_dist","cnnreg_noaxisdep_dist" ]:
  #  for n in ["cnnreg_axisdep", "cnnreg_noaxisdep",  "cnnreg2_axisdep", "cnnreg2_noaxisdep"]:
        aoption = trials_to_rows[n+"_mae"]

        aoption = list(df.iloc[aoption][iStartidx:iStartidx+iMeas])

        aoptionstd= list(df.iloc[trials_to_rows[n+"_std"]][iStartidx:iStartidx+iMeas])
        final_mae = []
        final_std= []
        for k in range(len(aoption)):
            final_mae.append(aoption[k])
            final_std.append(aoptionstd[k])

        maes.append(final_mae)
        stds.append(final_std)

    maes =np.array(maes)
    stds= np.array(stds)
    trials = ["CNN Joint Norm. ", "CNN Independent Norm.", "CNN Joint Norm. Disturbed", "CNN Independent Norm. Disturbed"]


plot_results_normstrat(maes, normstrat_names, trials,"MAE", bShow=True)
