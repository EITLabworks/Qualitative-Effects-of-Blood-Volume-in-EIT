"""
Project ：quali_effects_eit_measurements
Directory: src
File : loading_data.py
Author ：Patricia Fuchs
Date ：07.01.2026 14:28
"""

import os
import numpy as np
from sciopy.sciopy.sciopy_dataclasses import EITFrame


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

class MeaMeta:
    def __init__(self, folder, probetype, positions, vent, meatype):
        self.sFolder = folder
        self.sProbetype = probetype
        self.piPositions = positions
        self.iPosEncoding = 0
        if self.piPositions[0] == "none":
            self.iPosEncoding = -2
        else:
            self.iPosEncoding = np.sum(self.piPositions)
        self.sVent = vent
        self.sMeaType = meatype

    def compare(self, other):
        if self.sMeaType == other.sMeaType and self.sVent == other.sVent and self.sProbetype == other.sProbetype and self.piPositions == other.piPositions:
            return True
        else:
            return False




# -------------------------------------------------------------------------------------------------------------------- #
def load_eit_frames_metaclass(path):
    """
    Loads NPZ eit frames and stores them in a list of EITFrame
    Args:
        path: Path of the NPZ eit frames

    Returns:
        List of EITFrame
    """
    loaded = []
    print(path)
    files = os.listdir(path)
    files = sorted(files)
    idx = []
    meta = []
    for f in files:
        l = np.load(os.path.join(path, f), allow_pickle=True)
        e = EITFrame(
            n_el=l["n_el"],
            excitation_stgs=l["excitation_stgs"],
            frequency_stgs=l["frequency_stgs"],
            timestamp1=l["timestamp1"],
            timestamp2=l["timestamp2"],
            timestamp_pc=l["timestamp_pc"],
            ppcData=l["ppcData"],
        )
        loaded.append(e)
        idx.append(int(f[10:16]))
        m = None
        try:
            m = MeaMeta(path, str(l["probetype"]), list(l["probeposition"]), str(l["ventilation"]), str(l["meatype"]))
        except:
            raise ValueError(f)
        meta.append(m)
    return loaded, idx, meta


# -------------------------------------------------------------------------------------------------------------------- #
def load_full_dataset(path, folders, probes, pos, vents=["none", "ballon"], mea=["static", "timedependent", "high", "low"]):
    pData = []
    pMeta = []
    for folder in folders:
        subdirs = os.listdir(os.path.join(path, folder))
        for subdir in subdirs:
            p = os.path.join(path, folder, subdir)
            d, i, m = load_eit_frames_metaclass(p)
            pData.extend(d)
            pMeta.extend(m)

    # Delete unwanted data:
    k = 0
    while k < len(pMeta):
        if pMeta[k].sProbetype in probes and pMeta[k].piPositions in pos and pMeta[k].sVent in vents and pMeta[k].sMeaType in mea:
            k += 1
        else:
            del pMeta[k]
            del pData[k]
    return pData, pMeta


# -------------------------------------------------------------------------------------------------------------------- #
def make_datarray_from_frames(frames):
    res = []
    for frame in frames:
        res.append(frame.ppcData)
    return np.abs(np.array(res))
