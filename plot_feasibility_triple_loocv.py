"""
Project ：quali_effects_eit_measurements
Directory:
File : plot_feasibility_triple_loocv.py
Author ：Patricia Fuchs
Date ：11.08.2026 11:27
"""
from src.nn.help_nn_fcts import *
import os
from src.nn.visu_nn_results import visu_results_triple_vloocv
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# Loading paths ------------------------------------------------------------------------------------------ #

mprefix = "/results/nn/models/"
savepath = "/results/paper/"
res_path = "/results/paper/data/"

# Parsing of arguments ------------------------------------------------------------------------------------------ #


# TrainingParameters
secdataconfig = "8meafull"
modelpath = "/20260811-161643/"
modelpath2 = "/20260422-010501/"
dataconfig3 = "9meafull1pr"
num_data = 6
folders_3 = ["/20261119-052914/", "/20261120-201241/", "/20261122-091312/", "/20261123-152201/", "/20260416-150452/",
             "/20261126-104549/"]
normtype_3 = ["zscore", "minmax", "quantilenormal", "decimal", "none", "medmad"]
names_3 = ["T2-SmaCyl", "T2-MedCyl", "T2-LarCyl", "T2-SmaTub", "T2-MedTub", "T2-LarTub"]

# Load data
pfYTest = np.load(res_path+"pfytest_sim.npy")
pfYTest_preds = np.load(res_path+"pfytest_preds_sim.npy")

pfYTest2 = np.load(res_path + "full.npy")
pfYTest2_preds = np.load(res_path + "full_preds.npy")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

pfY3_preds = []
pfY3 = []
n = ["pr0", "pr1", "pr2", "pr3", "pr4", "pr5"]
for kdata in range(num_data):
    pfYTest3 = np.load(res_path + n[kdata] + ".npy")
    pfYTest3_preds = np.load(res_path + n[kdata] + "_preds.npy")
    pfY3.append(pfYTest3)
    pfY3_preds.append(pfYTest3_preds)

# Visualization ----------------------------------------------------------------------------- #
bSave = False
bPlotGraphics = True

visu_results_triple_vloocv(pfYTest, pfYTest_preds, pfYTest2, pfYTest2_preds, pfY3, pfY3_preds, "Simulation Data (Sim1)",
                           "Tank Data (T1)", names_3, bSave, savepath, bShow=bPlotGraphics)
print("Testing finished.")
