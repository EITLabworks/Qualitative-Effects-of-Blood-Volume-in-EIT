"""
Project ：quali_effects_eit_measurements
Directory: src/data
File : mea_meta.py
Author ：Patricia Fuchs
Date ：16.04.2026 15:55
"""
import numpy as np
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
class MeaMeta:
    def __init__(self, folder, probetype, positions, vent, meatype, proberadius=None, probearea=None):
        self.sFolder = folder
        self.sProbetype = probetype
        self.piPositions = positions
        self.iPosEncoding = 0
        if self.piPositions[0] == "none" or self.piPositions[1] == "all" or self.piPositions[0]=="all":
            self.iPosEncoding = -2
        else:
            self.iPosEncoding = np.sum(self.piPositions)
        self.sVent = vent
        self.sMeaType = meatype
        self.fProbeRadius = proberadius
        self.fProbeArea = probearea

    def compare(self, other):
        if other.sMeaType=="":
            if other.sProbetype=="cylinder":
                return self.sVent == other.sVent and list(self.piPositions) == list(other.piPositions) and self.fProbeArea == other.fProbeArea
            else:
                return self.sVent == other.sVent and self.sProbetype == other.sProbetype and list(self.piPositions) == list(other.piPositions)  and self.fProbeArea == other.fProbeArea
        if other.sProbetype=="cylinder":
            if self.sMeaType == other.sMeaType and self.sVent == other.sVent and list(self.piPositions) == list(other.piPositions)  and self.fProbeArea == other.fProbeArea:
                return True
            else:
                return False
        if self.sMeaType == other.sMeaType and self.sVent == other.sVent and self.sProbetype == other.sProbetype and list(self.piPositions) == list(other.piPositions)  and self.fProbeArea==other.fProbeArea:
            return True
        else:
            return False

    def is_subset_of(self, other):
        if is_subset_probetypes(self.sProbetype, other.sProbetype) is False:
            return False
        if is_subset_probeposition(self.piPositions, other.piPositions) is False:
            return False
       # if other.piPositions[1]!= "all" and list(other.piPositions) != list(self.piPositions):
     #       return False
        if other.sVent != self.sVent and other.sVent !="all":
            return False
        if other.sMeaType != self.sMeaType and other.sMeaType !="all":
            return False
        if other.fProbeArea != self.fProbeArea and other.fProbeArea !=None:
            return False
        return True


    def set_position(self, pos):
        self.piPositions = pos
        if self.piPositions[0] == "none":
            self.iPosEncoding = -2
        else:
            self.iPosEncoding = np.sum(self.piPositions)

    def __str__(self):
        return f"MeaMeta of {self.sProbetype} Position: {str(self.piPositions)}, Probearea: {self.fProbeArea}"



def is_subset_probetypes(pr1, prbigger):
    if pr1 == prbigger:
        return True
    elif prbigger=="all":
        return True
    if pr1.endswith("tube") and prbigger=="tube":
        return True
    elif prbigger=="cylinder" and pr1.endswith("cylinder"):
        return True
    elif prbigger =="none" or prbigger=="ballon" and pr1=="none" or prbigger=="ballon":
        return True
    else:
        return False


def is_subset_probeposition(p1, p2):
    if list(p1) == list(p2) or (p2[0]=="all" and p1[1] ==p2[1]) or (p2[1]=="all" and p1[0] ==p2[0])or (p2[0]=="all" and p2[1]=="all"):
        return True
    else:
        return False
