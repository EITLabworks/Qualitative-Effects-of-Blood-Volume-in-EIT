"""
Project ：quali_effects_eit_measurements
Directory: 
File : measure_procedures.py
Author ：Patricia Fuchs
Date ：05.12.2025 14:15
"""

import numpy as np
from sciopy.sciopy import EIT_16_32_64_128, EitMeasurementSetup, available_serial_ports
import time
from src.data.loading_data import MeaMeta, load_eit_frame, save_eitdata_and_metadata
import os
from tqdm import tqdm
from src.measurement.printer_control.printer_utils import create_trajectory_rdegbased


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

def start_procedure(sciospec: EIT_16_32_64_128, eitsetup: EitMeasurementSetup):
    print("Water level should have been checked.")
    print("Key dates should have been noted")
    sciospec.SetMeasurementSetup(eitsetup)

    print("Standard measurement setup was set at the EIT device.")


def measurement_static(sciospec, sSavePath):
    sciospec.update_FrameRate(5)
    sciospec.update_BurstCount(0)
    time.sleep(5)
    sciospec.StartStopMeasurement(timeout=10, return_as="eitframe", bSaveData=True, bDeleteData=True,
                                  sSavePath=sSavePath)
    print("Measurement finished.")


def measurement_static_with_meta(sciospec, sSavePath, mea_meta, sleeptime=5, timeout=6, **kwargs):
    # mea_meta class
    sciospec.update_FrameRate(5)
    sciospec.update_BurstCount(0)
    # get last idx
    iStartIdx = get_last_idx(sSavePath) +1
    time.sleep(sleeptime)
    sciospec.StartStopMeasurement(timeout=timeout, return_as="eitframe", bSaveData=True, bDeleteData=True,
                                  sSavePath=sSavePath)
    iLastIdx = get_last_idx(sSavePath)
    if iStartIdx == 0:
        iStartIdx = get_first_idx(sSavePath)
        if iStartIdx == 0:
            raise ValueError("No data was saved.")
    resave_meta_data(sSavePath, iStartIdx, iLastIdx, mea_meta, **kwargs)


#   print("Measurement finished.")


def get_last_idx(sSavePath):
    files = sorted(os.listdir(sSavePath))
    if len(files) == 0:
        return 0
    last_idx = int(files[-1][-10:-4])
    return last_idx


def get_first_idx(sSavePath):
    files = sorted(os.listdir(sSavePath))
    if len(files) == 0:
        return 0
    first_idx = int(files[0][-10:-4])
    return first_idx


def resave_meta_data(sSavePath, iSidx, iLidx, mea_meta, **kwargs):
    for i in range(iSidx, iLidx + 1):
        e = load_eit_frame(sSavePath + "eitsample_{0:06d}.npz".format(i))
        save_eitdata_and_metadata(sSavePath, e, mea_meta, i, **kwargs)



def measurement_2D_trajectory(sciospec, printer, sSavePath, radius, degrees, meta, factor=2):
    printer.move_to_absolute([0, 0])
    positions, meta_positions = create_trajectory_rdegbased(radius, degrees, factor=factor)

    for iIdx, pos in tqdm(enumerate(positions), total=len(positions)):
        printer.move_to_absolute(pos)
        meta.set_position(meta_positions[iIdx])
        temp = printer.read_temperature()
        # for in jupyter print("\r", b, end="")
        print(f"Measuring with {meta.sProbetype} at {meta.piPositions}", end="\r")
        # With delay and automatic saving of meta data
        measurement_static_with_meta(sciospec, sSavePath, meta, timeout=4, printer_temp=temp)
    print(f"Measurement for {meta.sProbetype} finished.")
