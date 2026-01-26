"""
Project ：Qualitative-Effects-Blood-Volume-EIT
Directory: src/nn
File : plotting.py.py
Author ：Patricia Fuchs
Date ：26.01.2026 11:24
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

def visu_metrics_heat(y_true_test, y_predict_test, y_true_vali, y_pred_vali, name, pMetaTest, pMetaVali, bSave=True,
                      bShow=False, sSavePath="C/"):
    """
    Visualize certain metrics
    """
    fontsize = 16
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 16
    })
    fig, axs = plt.subplots(nrows=2, figsize=(6, 8))

    plt.subplots_adjust(left=0.14, top=0.98, right=0.99, bottom=0.095, hspace=0.12)
    for axi in axs.flat:
        for axis in ["top", "bottom", "left", "right"]:
            axi.spines[axis].set_linewidth(2)
    #    for i in range(2):
    #       axs[i].set_facecolor("whitesmoke")

    le_p = LabelEncoder()
    pMetaPosTest = []
    pMetaPosVali = []
    for k in range(len(pMetaTest)):
        pMetaPosTest.append(pMetaTest[k].iPosEncoding)
    for k in range(len(pMetaVali)):
        pMetaPosVali.append(pMetaVali[k].iPosEncoding)
    options = ["-2", "0", "202", "232", "204", "234", "206", "221", "236", "208", "218", "228", "238"]
    le_p.fit(options)
    probes = []
    for p in le_p.classes_:
        probes.append(p)
    labelsvali = le_p.transform(pMetaPosVali) + 1
    labelstest = le_p.transform(pMetaPosTest) + 1
    labels_printvali = []
    for k in range(len(pMetaPosVali)):
        if pMetaPosVali[k] not in labels_printvali:
            labels_printvali.append(pMetaPosVali[k])

    labels_printtest = []
    for k in range(len(pMetaPosTest)):
        if pMetaPosTest[k] not in labels_printtest:
            labels_printtest.append(pMetaPosTest[k])
    scat = axs[0].scatter(y_true_vali, y_pred_vali, alpha=0.25, s=25, c="midnightblue", label="Training")
    #   handles, labels = scat.legend_elements()
    #   axs[0].legend(handles, labels_printvali)
    scattest = axs[1].scatter(y_true_test, y_predict_test, alpha=0.2, s=25, c="maroon", label="Testing")
    #   handles, labels = scattest.legend_elements()
    #   axs[1].legend(handles, labels_printtest)

    limits = [np.min(y_true_vali) - 5, np.max(y_true_vali) + 5]
    line = np.arange(limits[0], limits[1], 1)
    axs[0].plot(line, line, color="black", linewidth=2, linestyle="--")
    axs[1].plot(line, line, color="black", linewidth=2, linestyle="--")

    #  fig.suptitle("Actual vs.Predicted values", fonsize=20)
    # axs[0].set_title("Training", fontsize=20)
    #  axs[1].set_title("Testing", fontsize=20)

    for i in range(2):
        axs[i].legend(fontsize=fontsize)
        #   axs[i].set_xlabel("Real Probe Size [cm^2]", fontsize=20, loc="right")
        axs[i].set_ylabel("Estimated Probe Size [cm$^2$]", fontsize=fontsize, loc="top")
    axs[1].set_xlabel("Real Probe Size [cm$^2$]", fontsize=fontsize, loc="right")
    # axs[1].set_ylabel("Residuals (actual - predicted)", fontsize=20, loc="top")
    #    fig.suptitle("Plotting predictions for " + name, size=24)
    #      plt.tight_layout()
    if bSave:
        fig.savefig(sSavePath + name + "PredictionErrorHeatmap.png")
    if bShow:
        plt.show()


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
    #   handles, labels = scat.legend_elements()
    #   axs[0].legend(handles, labels_printvali)
    scattest = axs.scatter(y_true_test, y_predict_test, alpha=0.2, s=25, c="goldenrod", label="Test")
    #   handles, labels = scattest.legend_elements()
    #   axs[1].legend(handles, labels_printtest)

    limits = [np.min(y_true_vali) - 5, np.max(y_true_vali) + 5]
    line = np.arange(limits[0], limits[1], 1)
    axs.plot(line, line, color="black", linewidth=2, linestyle="--")
    #  axs[1].plot(line, line, color="black", linewidth=2, linestyle="--")
    axs.legend(fontsize=20)
    #  fig.suptitle("Actual vs.Predicted values", fonsize=20)
    #   axs.set_title("Training", fontsize=20)
    #   axs[1].set_title("Testing", fontsize=20)

    axs.set_xlabel("Real Probe Size [cm$^2$]", fontsize=20, loc="right")
    axs.set_ylabel("Estimated Probe Size [cm$^2$]", fontsize=20, loc="top")
    # axs[1].set_ylabel("Residuals (actual - predicted)", fontsize=20, loc="top")
    #    fig.suptitle("Plotting predictions for " + name, size=24)
    plt.tight_layout()
    if bSave:
        fig.savefig(sSavePath + name + "PredictionErrorHeatmap.png")
    if bShow:
        plt.show()



def visu_metrics_combines( y_true_test, y_predict_test, y_true_vali, y_pred_vali, name, bSave=True, bShow=False, sSavepath="C/"):
        """
        Visualize certain metrics
        """
        #plt.rcParams["font.family"] = "Times New Roman"
        fig, axs = plt.subplots(ncols=1, figsize=(20, 7.1))
        axs.scatter(y_true_vali,y_pred_vali, alpha=0.2, s=25, color="steelblue", label="Validation")

        plt.subplots_adjust(left=0.079, top=0.936, right=0.957, bottom=0.08)
        for axis in ["top", "bottom", "left", "right"]:
            axs.spines[axis].set_linewidth(2)
        axs.set_facecolor("whitesmoke")

        limits= [np.min(y_true_vali)-5, np.max(y_true_vali)+5]
        line= np.arange(limits[0], limits[1], 1)
        axs.plot(line, line, color="black", linewidth=2, linestyle="--")

        #  fig.suptitle("Actual vs.Predicted values", fonsize=20)
        axs.scatter(y_true_test,y_predict_test, alpha=0.2, s=25, color="maroon", label = "Testing")
        limits= [np.min(y_true_test)-5, np.max(y_true_test)+5]
        line= np.arange(limits[0], limits[1], 1)
        axs.plot(line, line, color="black", linewidth=2, linestyle="--")
        axs.set_xlabel("Real Probe Size [cm^2]", fontsize=20, loc="right")
        axs.set_ylabel("Estimated Probe Size [cm^2]", fontsize=20, loc="top")
        # axs[1].set_ylabel("Residuals (actual - predicted)", fontsize=20, loc="top")
    #    fig.suptitle("Plotting predictions for " + name, size=24)
        plt.tight_layout()
        if bSave:
            fig.savefig(sSavepath + name + "PredictionErrorCombine.png")
        if bShow:
            plt.show()

def visu_metrics_bypos( y_true_test, y_predict_test, y_true_vali, y_pred_vali, name, pMetaTest, pMetaVali, bSave=True, bShow=False, sSavepath="C/"):
        """
        Visualize certain metrics
        """

        # plt.rcParams["font.family"] = "Times New Roman"
        fig, axs = plt.subplots(ncols=2, figsize=(20, 7.1))

        plt.subplots_adjust(left=0.079, top=0.936, right=0.957, bottom=0.08)
        for axi in axs.flat:
            for axis in ["top", "bottom", "left", "right"]:
                axi.spines[axis].set_linewidth(2)
        for i in range(2):
            axs[i].set_facecolor("whitesmoke")

        le_p = LabelEncoder()
        pMetaPosTest = []
        pMetaPosVali = []
        for k in range(len(pMetaTest)):
            pMetaPosTest.append(pMetaTest[k].iPosEncoding)
        for k in range(len(pMetaVali)):
            pMetaPosVali.append(pMetaVali[k].iPosEncoding)
        options= ["-2", "0", "202", "232", "204", "234", "206", "221", "236", "208", "218", "228", "238"]
        le_p.fit(options)
        probes = []
        for p in le_p.classes_:
            probes.append(p)
        labelsvali = le_p.transform(pMetaPosVali) + 1
        labelstest= le_p.transform(pMetaPosTest) + 1
        labels_printvali = []
        for k in range(len(pMetaPosVali)):
            if pMetaPosVali[k] not in labels_printvali:
                labels_printvali.append(pMetaPosVali[k])

        labels_printtest = []
        for k in range(len(pMetaPosTest)):
            if pMetaPosTest[k] not in labels_printtest:
                labels_printtest.append(pMetaPosTest[k])
        scat= axs[0].scatter(y_true_vali, y_pred_vali, alpha=0.3, s=25, c=labelsvali, cmap="viridis")
        handles, labels = scat.legend_elements()
        axs[0].legend(handles, labels_printvali)
        scattest= axs[1].scatter(y_true_test, y_predict_test, alpha=0.3, s=25, c=labelstest,cmap = "viridis")
        handles, labels = scattest.legend_elements()
        axs[1].legend(handles, labels_printtest)


        limits = [np.min(y_true_vali) - 5, np.max(y_true_vali) + 5]
        line = np.arange(limits[0], limits[1], 1)
        axs[0].plot(line, line, color="black", linewidth=2, linestyle="--")
        axs[1].plot(line, line, color="black", linewidth=2, linestyle="--")

        #  fig.suptitle("Actual vs.Predicted values", fonsize=20)
        axs[0].set_title("Validation", fontsize=20)
        axs[1].set_title("Testing", fontsize=20)

        for i in range(2):
            axs[i].set_xlabel("Real Probe Size [cm^2]", fontsize=20, loc="right")
            axs[i].set_ylabel("Estimated Probe Size [cm^2]", fontsize=20, loc="top")
        # axs[1].set_ylabel("Residuals (actual - predicted)", fontsize=20, loc="top")
        #    fig.suptitle("Plotting predictions for " + name, size=24)
        plt.tight_layout()
        if bSave:
            fig.savefig(sSavepath + name + "PredictionErrorbyPos.png")
        if bShow:
            plt.show()
