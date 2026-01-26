"""
Project ：quali_effects_eit_measurements
Directory: 
File : load_data_nn.py.py
Author ：Patricia Fuchs
Date ：08.01.2026 14:58
"""

from src.data.loading_data import *
from src.data.normalization_static import EITNormalizationStatic
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


def load_mea_data_static(path, folders,testcond, probes, pos, mea, vents,iNel, fR, injskip=4, meapatt="skip4internal",
                         posfactor="none", normtype="zcore", axisdep=False, bReshape=False, bShuffle=False, bRealPart=False):

    pData, pMeta = load_full_dataset(path, folders, probes, pos, vents, mea)
    ppfData = make_datarray_from_frames(pData)
    pMeta = np.array(pMeta)

    ppfTrain, pMetaTrain, ppfTest, pMetaTest = split_train_test(ppfData, pMeta,testcond)

    Norm = EITNormalizationStatic(normtype, axisdependent=axisdep)

    ppfTrain = Norm.normalize_forward(ppfTrain)
    ppfTest = Norm.normalize_forward(ppfTest, bUseFactors=True)      # use previous normalization factors

    pfYTrain = extract_area_from_meta(pMetaTrain)
    pfYTest = extract_area_from_meta(pMetaTest)
    pfYTrain = norm_area_forward(pfYTrain)
    pfYTest = norm_area_forward(pfYTest)

    if bReshape:
        ppfTrain = reshape_eit_static(ppfTrain, iNel)
        ppfTest = reshape_eit_static(ppfTest, iNel)

    shuffle = create_shuffling_vector(bShuffle, len(ppfTrain))
    print(f"Len TrainingData = {len(ppfTrain)}, Len TestData = {len(ppfTest)}, Percentage Testing {len(ppfTest)/len(ppfData)*100:.2f}")

    return ppfTrain[shuffle],pfYTrain[shuffle] , pMetaTrain[shuffle], ppfTest, pfYTest, pMetaTest




def segment_into_blocks(pData, pMeta):
    ppfArray = make_datarray_from_frames(pData)
    currMeta= pMeta[0]
    idx_started= 0
    timestamp_started = pData[0].timestamp_pc
    currTimestamp= pData[0].timestamp_pc
    delta= 3
    blocks= []
    ppsTimestrings=[]
    pMetaNew= []
    for i in range(1,len(pData)):
        if currMeta.compare(pMeta[i]) and pData[i].timestamp_pc < currTimestamp+delta:
            currTimestamp= pData[i].timestamp_pc
            continue
        else:
            blocks.append(ppfArray[idx_started:i])
            pMetaNew.append(currMeta)
            ppsTimestrings.append([timestamp_started,currTimestamp])
            idx_started= i
            currMeta= pMeta[i]
            timestamp_started = pData[i].timestamp_pc
            currTimestamp= pData[i].timestamp_pc

    blocks.append(ppfArray[idx_started:len(pData)])
    pMetaNew.append(currMeta)
    ppsTimestrings.append([timestamp_started, currTimestamp])
    return blocks,pMetaNew, ppsTimestrings



def split_train_test(ppfData, pMeta, testcond):
    testidx = []
    for i in range(len(ppfData)):
        testdata= True
        for t in testcond:
            if pMeta[i].compare(t):
                testidx.append(i)
    testidx = np.array(testidx, dtype=int)

    mask = np.ones(len(ppfData), dtype=bool)
    mask[testidx] = False
    return ppfData[mask], pMeta[mask], ppfData[testidx], pMeta[testidx]

def split_train_test2(ppfData, pMeta, testcond, pY):
    testidx = []
    for i in range(len(ppfData)):
        testdata= True
        for t in testcond:
            if pMeta[i].compare(t):
                testidx.append(i)
    testidx = np.array(testidx, dtype=int)

    mask = np.ones(len(ppfData), dtype=bool)
    mask[testidx] = False
    return ppfData[mask], pMeta[mask], pY[mask], ppfData[testidx], pMeta[testidx], pY[testidx]




# -------------------------------------------------------------------------------------------------------------------- #
def extract_area_from_meta(pfMeta):
    area= []

    for i in range(len(pfMeta)):
        if pfMeta[i].sProbetype =="none" or pfMeta[i].sProbetype=="ballon":
            area.append(0)
        elif pfMeta[i].sProbetype=="smallcylinder" or (pfMeta[i].sProbetype=="smalltube" and pfMeta[i].sMeaType=="high"):
            area.append(2.01)
        elif pfMeta[i].sProbetype=="mediumcylinder" or (pfMeta[i].sProbetype=="mediumtube" and pfMeta[i].sMeaType=="high"):
            area.append(5.73)
        elif pfMeta[i].sProbetype=="largecylinder" or (pfMeta[i].sProbetype=="largetube" and pfMeta[i].sMeaType=="high"):
            area.append(10.18)
        elif pfMeta[i].sProbetype=="smalltube" and pfMeta[i].sMeaType=="low":
            area.append(1.73)
        elif pfMeta[i].sProbetype == "mediumtube" and pfMeta[i].sMeaType == "low":
            area.append(3.96)
        elif pfMeta[i].sProbetype == "largetube" and pfMeta[i].sMeaType == "low":
            area.append(4.87)
        else:
            raise ValueError("unidentified meta "+str(i)+ pfMeta[i].sProbetype+ pfMeta[i].sMeaType)
    return np.array(area)


# -------------------------------------------------------------------------------------------------------------------- #
def norm_area_forward(pfData):
    return  pfData/11

def norm_area_inverse(pfData):
    return pfData*11


def reshape_eit_static(ppfData, iNel=32):
    ppfData= ppfData.reshape((ppfData.shape[0], iNel, iNel, 1))
    return ppfData


def create_shuffling_vector(bShuffle, N, iMaxLen=0):
    shuffle = np.arange(N)
    if bShuffle:
        np.random.shuffle(shuffle)
    if iMaxLen > 0:
        shuffle = shuffle[:iMaxLen]
    return shuffle




def apply_dataaugmentation(Ntrain, factor, ppfDataTrain, pMeta,pY, bReshape):
    noise_power= np.load("/data/data_aug/pMeaNoise.npz")
    noise_power = noise_power["pMeaNoise"].flatten()
    iStepnumber= int(Ntrain//4)
    i =0.5
    idx= 0
    ppfAugmented= ppfDataTrain.copy()
    factor= min(5,factor)
    while i <= factor:
        noise= []
        for k in range(1024):
            noise.append(np.random.normal(scale=i*noise_power[k], size=iStepnumber))
        noise= np.array(noise).T
        if bReshape:
            noise = np.reshape(noise,(noise.shape[0],32,32,1))
        pAug = noise + ppfDataTrain[idx:idx+iStepnumber]
        ppfAugmented= np.append(ppfAugmented, pAug, axis=0)
        pMeta= np.append(pMeta, pMeta[idx:idx+iStepnumber], axis=0)
        pY= np.append(pY, pY[idx:idx+iStepnumber], axis=0)
        idx += iStepnumber
        if idx + iStepnumber > Ntrain:
            idx = 0
        i+=0.5

    shuffle= create_shuffling_vector(True, len(ppfAugmented))
    print(f"Data with factor={factor}*2 noise steps was augmented from {Ntrain} to {len(ppfAugmented)} samples")
    return ppfAugmented[shuffle], pMeta[shuffle], pY[shuffle]