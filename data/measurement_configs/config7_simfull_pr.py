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

input_folders=["Phase2Cylinder/" ]
probetypes = ["none", "ballon", "cylinder"]
#positions = [["none", "none"],[0,0], [2,200], [2,230], [4,200], [4,230], [6,200], [6,215], [6,230], [8,200], [8,210], [8,220],[8,230],]
vents= ["none" ]
mealist= ["static"]          # not timedependent

def make_positions():
    radius = np.linspace(0,6, 31)
    degrees = np.linspace(180,272,24)
    degrees.astype(int)
    radius= np.round(radius, 2)
    degrees = np.round(degrees, 2)
    c = np.array(np.meshgrid(radius[1:], degrees)).T.reshape(-1, 2)
    c = np.append(np.array([[0, 0]]), c, axis=0)
    c= c.tolist()
    print(type(c))
    print(type(c[0]))
    return c
positions= make_positions()
percentage=0.1


def make_test_valicond(percentage,  positions):
    testcond = []
    valicond = []
    pos_array= np.array(positions)
    probeareas= np.linspace(1.4, 12.4, 56)
    np.round(probeareas, 2)
    np.round(pos_array, 2)
    
    shuffle= np.arange(len(probeareas), dtype=int)
    np.random.shuffle(shuffle)
    probes_shuff = probeareas[shuffle]
    Ntest = int(percentage*len(probeareas)) 
    for k in range(0,Ntest):
        for q in range(0,len(pos_array)):
            testcond.append(MeaMeta("", "cylinder", pos_array[q], "none", "", probearea=probes_shuff[k]))
            valicond.append(MeaMeta("", "cylinder", pos_array[q], "none", "", probearea=probes_shuff[Ntest+k]))

    Pcombis = len(positions)* len(mealist) * len(vents)
    print(f"NUMBER of possible combinations per probe {Pcombis}")
    Pslice= int(Pcombis*percentage)
    shuffle= np.arange(Pcombis, dtype=int)
    np.random.shuffle(shuffle)
    pos_shuff= pos_array[shuffle]
    iStart= int(Ntest*2)
    i=0
    for probe in probes_shuff[iStart:]:
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

    print("Testing and validation conditions were created.")
    return  testcond, valicond

    


testcond, valicond = make_test_valicond(percentage, positions)

