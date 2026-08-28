"""
Project ：quali_effects_eit_measurements
Directory: data/measurement_configs
File : config7_simfull.py.py
Author ：Patricia Fuchs
Date ：03.03.2026 15:16
"""


from src.data.loading_data import MeaMeta
import numpy as np
np.random.seed(9)
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

input_folders = ["Phase2Cylinder/" , "Phase4Tubes/"]
probetypes = ["none", "ballon", "cylinder",  "smallcylinder", "mediumcylinder", "largecylinder", "smalltube", "mediumtube", "largetube"]
# positions = [["none", "none"],[0,0], [2,200], [2,230], [4,200], [4,230], [6,200], [6,215], [6,230], [8,200], [8,210], [8,220],[8,230],]
vents= ["none" ]
mealist = ["static", "low"]          # not timedependent


def make_positions():
    radius = np.linspace(0,6, 31)
    degrees = np.linspace(180,272,24)
    degrees.astype(int)
    np.round(radius, 2)
    c = np.array(np.meshgrid(radius[1:], degrees)).T.reshape(-1, 2)
    c = np.append(np.array([[0, 0]]), c, axis=0)
    c = c.tolist()
    np.round(c,2)
    for j in range(len(c)):
        c[j][0] = round(c[j][0],2)
    return c


positions = make_positions()
percentage = 0.1


def make_test_valicond(percentage,  positions):
    testcond = []
    valicond = []
    probetype= [ "smallcylinder", "mediumcylinder", "largecylinder", "smalltube", "mediumtube", "largetube"]
    pos_array= np.array(positions)
    probeareas= [2.01,5.73,10.18,1.73,3.96,4.87]
    probeareas= [1.54,5.73,10.18,1.99,3.96,4.87]
    Pcombis = len(positions)
    Pslice = int(Pcombis*percentage)
    shuffle= np.arange(Pcombis, dtype=int)
    np.random.shuffle(shuffle)
    pos_shuff= pos_array[shuffle]
    i=0
    for probe in probeareas:
        iT = 0
        while iT < Pslice:
            testcond.append(MeaMeta("", "cylinder", pos_shuff[i], "none", "", probearea=probe))
            i+=1
            if i ==Pcombis:
                i=0
            valicond.append(MeaMeta("", "cylinder", pos_shuff[i], "none", "", probearea=probe))
            i+=1
            if i ==Pcombis:
                i=0
            iT +=1
    for j in range(0,Pcombis):
            testcond.append(MeaMeta("", "cylinder", pos_shuff[j], "none", "", probearea=3.96))

    print("Testing and validation conditions were created.")
    print(testcond[0])
    print(valicond[0])
    print(len(testcond))
    print(len(valicond))
    return  testcond, valicond


testcond, valicond = make_test_valicond(percentage, positions)

positions.append(["none", "none"])