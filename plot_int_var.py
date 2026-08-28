"""
Project ：Qualitative-Effects-of-Blood-Volume-in-EIT
Directory: /scripts/plotter
File : plot_int_var.py
Author ：Patricia Fuchs
Date ：30.07.2026 09:44
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
plt.rcParams["font.family"] = "Times New Roman"

df =pd.read_excel("/results/Int var results.xlsx", sheet_name="Sheet1")

print(df.head())
trial_names= ["Joint z-Score", "Independent z-Score", "Joint QT Normal", "Independent QT Normal", "None"]
names_xaxis= ["None", "6","12", "28", "42", "56"]

iStartidx= 1
iStart_col=9
data_normsep = []
data_nonormsep = []

for i in [iStartidx,iStartidx+4, iStartidx+8, iStartidx+12, iStartidx+16 ]:
    z= df.iloc[i,10]
    data_normsep.append(list(df.iloc[i][iStart_col: iStart_col+6]))
    data_nonormsep.append(list(df.iloc[i+2][iStart_col: iStart_col+6]))



def plot_results_intvar(names, data1, data2, xaxis, mea, bSave=False, bShow=False, sSavepath= "C/"):
    fig, ax= plt.subplots(figsize=(8,5))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 20
    })
    fs=20
    plt.subplots_adjust(left=0.1, top=0.936, right=0.986, bottom=0.13)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
    #ax.set_facecolor("whitesmoke")
    x_axis= np.arange(len(xaxis))
    colors= ["midnightblue","maroon", "steelblue", "indianred", "goldenrod", "indianred", "forestgreen", "teal","peru", "yellowgreen", "blueviolet", "crimson", "dimgrey", "limegreen", "peru", "blueviolet", "skyblue"]
    for k, t in enumerate(data1):
            ax.plot( t,marker="o", markersize=6,color=colors[k], label=names[k])
            ax.plot(data2[k], ls = "--", marker="o", markersize=6, color=colors[k])

    ax.set_xticks(x_axis, xaxis,fontsize=fs)
    ax.legend(loc="upper right", fontsize=fs)
    ax.set_ylabel(mea, fontsize=fs, loc="top")
    ax.tick_params(axis="both", which="major", labelsize=fs)
    ax.tick_params(axis="both", which="minor", labelsize=fs-2)
    minor_ticks= np.linspace(0,2,21)
    print(minor_ticks)

    ax.set_xlabel("Number of Simulation Probes",fontsize=fs, loc="right")

    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

  #  plt.tight_layout()
    if bSave:
        fig.savefig(sSavepath + mea + "EITIntVar.png")
    if bShow:
        plt.show()


plot_results_intvar(trial_names, data_normsep, data_nonormsep, names_xaxis, "MAE", bShow=True)