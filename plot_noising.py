"""
Project ：Qualitative-Effects-of-Blood-Volume-in-EIT
Directory: 
File : plot_noising.py
Author ：Patricia Fuchs
Date ：30.07.2026 09:44
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
plt.rcParams["font.family"] = "Times New Roman"

df =pd.read_excel("/results/Results_Noising.xlsx", sheet_name="Sheet2")

print(df.head())
x=5

iMeas = 14
iStartidx=7
normstrat_names = ["None", "z-Score", "Min-Max", "Max-Absolute", "Decimal", "Median", "Med-MAD", "VSS", "Sigmoid", "Tanh", "Scaled Tanh", "QT Uniform", "QT Normal", "Power \nTransform"]
names_plot = [ "None","Zscore", "MinMax", "MaxAbsolute", "Decimal", "Median", "MedMad", "VSS", "Sigmoid", "Tanh", "HyperbolicTangent", "QT uniform", "QT normal", "Powertransform"]


names = []
data = []
iLenMeas = 18
for i in range(iStartidx, iStartidx+iMeas):
    names.append(df.iloc[i]["MAE"])
    data.append(list(df.iloc[i][8: 8+iLenMeas]))


def plot_results_noising(trials, names, trialnames, mea, bSave=False, bShow=False, sSavepath="C/"):
    fig, ax= plt.subplots(figsize=(15,7))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 18
    })
    fs=18
    plt.subplots_adjust(left=0.07, top=0.936, right=0.986, bottom=0.1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
    #ax.set_facecolor("whitesmoke")
    x_axis= np.append([-2],np.arange(len(names)-1))
    factors = [1,0.5,1.5]
    colors= ["grey", "steelblue", "maroon","indianred","crimson", "goldenrod","peru",  "teal", "midnightblue","rebeccapurple", "blueviolet","teal",  "darkgreen", "yellowgreen", "crimson", "dimgrey", "limegreen", "peru", "blueviolet", "skyblue"]
    for k, t in enumerate(trials):
        if trialnames[k] =="VSS":
            continue
        else:
            ax.plot( x_axis, t,marker="o", markersize=7,color=colors[k], label=trialnames[k])

    ax.set_xlim([-2.5,len(names)+4])
    ax.set_xticks(x_axis, names,fontsize=fs)
    ax.legend(loc="best", fontsize=fs)
    ax.set_ylabel(mea, fontsize=fs, loc="top")
  #  ax.set_title("Results for different EIT normalization strategies", fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=fs)
    ax.tick_params(axis="both", which="minor", labelsize=fs-2)
    minor_ticks= np.linspace(0,1.5,16)
    print(minor_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_xlabel("SNR [dB]",fontsize=fs, loc="right")

    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

  #  plt.tight_layout()
    if bSave:
        fig.savefig(sSavepath + mea + "EITNoising.png")
    if bShow:
        plt.show()


xaxis= [ "No Noise", "55", "52.5", "50", "47.5", "45", "42.5", "40", "37.5", "35", "32.5","30", "27.5", "25",
        "22.5","20", "17.5", "15"]

plot_results_noising(data, xaxis, normstrat_names, "MAE", bShow=True)