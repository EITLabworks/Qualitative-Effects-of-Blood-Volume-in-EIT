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

from src.data.mea_meta import MeaMeta
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


def load_eit_frame(path):
    l = np.load(path, allow_pickle=True)
    e = EITFrame(
        n_el=l["n_el"],
        excitation_stgs=l["excitation_stgs"],
        frequency_stgs=l["frequency_stgs"],
        timestamp1=l["timestamp1"],
        timestamp2=l["timestamp2"],
        timestamp_pc=l["timestamp_pc"],
        ppcData=l["ppcData"],
    )
    return e


def save_eitdata_and_metadata(path: str, dataframe: EITFrame, mea_meta:MeaMeta, iNPZSaveIndex: int, **kwargs):
    """
    Saves a single EIT frame in a npz-file. Based on the EITframe class. Saves it at self.NPZSaveIndex
    Args:
        path: Where to save the EIT frame
        dataframe: EITFrame to be saved
        iNPZSaveIndex: Index of the EIT frame to be saved
    """

    np.savez(
        path + "eitsample_{0:06d}.npz".format(iNPZSaveIndex),
        n_el=dataframe.n_el,
        excitation_stgs=dataframe.excitation_stgs,
        frequency_stgs=dataframe.frequency_stgs,
        timestamp1=dataframe.timestamp1,
        timestamp2=dataframe.timestamp2,
        timestamp_pc = dataframe.timestamp_pc,
        ppcData=dataframe.ppcData,
        probetype = mea_meta.sProbetype,
        probeposition = mea_meta.piPositions,
        ventilation=mea_meta.sVent,
        meatype=mea_meta.sMeaType,
        proberadius=mea_meta.fProbeRadius,
        probearea=mea_meta.fProbeArea,
        **kwargs

    )



# -------------------------------------------------------------------------------------------------------------------- #
def load_eit_frames(path):
    """
    Loads NPZ eit frames and stores them in a list of EITFrame
    Args:
        path: Path of the NPZ eit frames

    Returns:
        List of EITFrame
    """
    loaded = []
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
        try:
            m = []
            m.append(l["probetype"][0])
            m.append(list(l["probeposition"]))
            m.append(l["ventilation"][0])
            m.append(l["meatype"][0])
            m.append(float(l["proberadius"]))
            m.append(l["probearea"])
        except:
            t = 5
        meta.append(m)
    return loaded, meta


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
            m.fProbeRadius = float(l["proberadius"])
            m.fProbeArea= round(float(l["probearea"]), 2)
            if m.piPositions[0]!= "none":
                m.piPositions[0] = round(m.piPositions[0],2)
        except:
            raise ValueError(f)
        meta.append(m)
    return loaded, idx, meta


# -------------------------------------------------------------------------------------------------------------------- #
def load_full_dataset(path, folders, probes, pos, vents=["none", "ballon"], mea=["static", "timedependent", "high", "low"],probesizes=[], bDelete=True):
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
    deleted=0
    print(pos[:10])
    if pos[-1][0] =="none":
        pos =round_list_2d(pos[:-1])
        pos = pos + [["none", "none"]]
    else:
        pos= round_list_2d(pos)
    if bDelete:
        k = 0
        if len(probesizes)==0:
            while k < len(pMeta):
                if pMeta[k].sProbetype in probes and pMeta[k].piPositions in pos and pMeta[k].sVent in vents and pMeta[k].sMeaType in mea:
                    k += 1
                else:
                   # print(pMeta[k].piPositions)
                    del pMeta[k]
                    del pData[k]
                    deleted +=1
        else:
            np.round(probesizes, 2)
            while k < len(pMeta):
                if pMeta[k].sProbetype in probes and pMeta[k].piPositions in pos and pMeta[k].sVent in vents and pMeta[k].sMeaType in mea and round(pMeta[k].fProbeArea, 2) in probesizes :
                    k += 1
                else:
                   # print(pMeta[k].piPositions)
                    del pMeta[k]
                    del pData[k]
                    deleted +=1
    print(f"DELETED {deleted}")
    return pData, pMeta


# -------------------------------------------------------------------------------------------------------------------- #
def make_datarray_from_frames(frames):
    res = []
    for frame in frames:
        res.append(frame.ppcData)
    return np.abs(np.array(res))



def round_list_2d(l, decimals=2):
    my_rounded_list = [ [round(el, decimals) for el in elem] for elem in l ]
    return my_rounded_list



def load_metadata_from_f(f):
    try:
        m = MeaMeta("", str(f["probetype"]), list(f["probeposition"]), str(f["ventilation"]), str(f["meatype"]))
        m.fProbeRadius = float(f["proberadius"])
        m.fProbeArea = round(float(f["probearea"]), 2)
        if m.piPositions[0] != "none":
            m.piPositions[0] = round(m.piPositions[0], 2)
    except:
        raise ValueError()
    return m


def load_metadata_subfolder(path):
    files = os.listdir(path)
    idx= []
    metas= []
    for f in files:
        l = np.load(os.path.join(path, f), allow_pickle=True)
        metas.append(load_metadata_from_f(l))
        idx.append(int(f[10:16]))
    return idx, metas


def load_eitframe_meta(path, index):
    sample_name = "eitsample_{0:06d}.npz".format(index)
    l = np.load(os.path.join(path, sample_name), allow_pickle=True)
    e = EITFrame(
        n_el=l["n_el"],
        excitation_stgs=l["excitation_stgs"],
        frequency_stgs=l["frequency_stgs"],
        timestamp1=l["timestamp1"],
        timestamp2=l["timestamp2"],
        timestamp_pc=l["timestamp_pc"],
        ppcData=l["ppcData"],
    )
    m = None
    try:
        m = MeaMeta(path, str(l["probetype"]), list(l["probeposition"]), str(l["ventilation"]), str(l["meatype"]))
        m.fProbeRadius = float(l["proberadius"])
        m.fProbeArea = round(float(l["probearea"]), 2)
        if m.piPositions[0] != "none":
            m.piPositions[0] = round(m.piPositions[0], 2)
    except:
        raise ValueError(path+index)
    return e, m