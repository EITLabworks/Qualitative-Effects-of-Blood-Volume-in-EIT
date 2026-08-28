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
def load_mea_data_static(path, folders,testcond, probes, pos, mea, vents, iNel, normtype="zcore", axisdep=False,
                         bReshape=False, bShuffle=False,  bAddDisturbances=False,
                         sprescale="none", fNoiseSNR=0, probesizes= []):

    pData, pMeta = load_full_dataset(path, folders, probes, pos, vents, mea, probesizes=probesizes)
    ppfData = make_dataarray_real(pData)
    pMeta = np.array(pMeta)

    if len(ppfData) > 500000:
        ppfData, pMeta = reduce_sim_dataset(ppfData, pMeta)

    ppfData = prescale_data(ppfData, sprescale)
    ppfTrain, pMetaTrain, ppfTest, pMetaTest = split_train_test(ppfData, pMeta,testcond)

    if bAddDisturbances:
        ppfD, pMetaD= load_disturbances(path)
        ppfTrain, pMetaTrain= integrate_disturbances(ppfTrain, pMetaTrain, ppfD, pMetaD)

    if fNoiseSNR !=0:
        ppfTrain = add_noise(ppfTrain,fNoiseSNR)

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
    used_testcond = []
    for i in range(len(ppfData)):
        for t in testcond:
            if pMeta[i].compare(t):
                testidx.append(i)
                if t not in used_testcond:
                    used_testcond.append(t)

    testidx = np.array(testidx, dtype=int)
    for k in range(len(testcond)):
        if testcond[k] not in used_testcond:
            print(testcond[k])

    mask = np.ones(len(ppfData), dtype=bool)
    mask[testidx] = False
    return ppfData[mask], pMeta[mask], ppfData[testidx], pMeta[testidx]

def split_train_test2(ppfData, pMeta, testcond, pY):
    testidx = []
    used_testcond= []
    for i in range(len(ppfData)):
        testdata= True
        for t in testcond:
            if pMeta[i].compare(t):
                testidx.append(i)
                if t not in used_testcond:
                    used_testcond.append(t)

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
        elif pfMeta[i].fProbeArea!=None:
            area.append(pfMeta[i].fProbeArea)
        elif pfMeta[i].sProbetype=="smallcylinder" or (pfMeta[i].sProbetype=="smalltube" and pfMeta[i].sMeaType=="high"):
            area.append(2.01)       # cm^2  früher 8.04
        elif pfMeta[i].sProbetype=="mediumcylinder" or (pfMeta[i].sProbetype=="mediumtube" and pfMeta[i].sMeaType=="high"):
            area.append(5.73)          # 22.9
        elif pfMeta[i].sProbetype=="largecylinder" or (pfMeta[i].sProbetype=="largetube" and pfMeta[i].sMeaType=="high"):
            area.append(10.18)         # 40.72
        elif pfMeta[i].sProbetype=="smalltube" and pfMeta[i].sMeaType=="low":
            area.append(1.73)           # 6.91
        elif pfMeta[i].sProbetype == "mediumtube" and pfMeta[i].sMeaType == "low":
            area.append(3.96)          # 15.83
        elif pfMeta[i].sProbetype == "largetube" and pfMeta[i].sMeaType == "low":
            area.append(4.87)          # 19.48
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


def reduce_sim_dataset(ppfData, pMeta, numsamples=5, numdelete=2):
    mask = np.ones(numsamples, dtype=bool)
    for k in range(numdelete):
        mask[k] = False
    N = len(ppfData) // numsamples
    mask_ext= np.tile(mask, N)
    ppfData = ppfData[mask_ext]
    K=len(pMeta)
    pMeta= pMeta[mask_ext]
    print(f"Repetitions reduced from K={K} samples to {len(pMeta)} samples")
    return ppfData,pMeta


def load_disturbances(path,bRealPart):
    pData, pMeta = load_full_dataset(path, ["Phase5Disturbances/"],[], [] , [],[], bDelete=False)
    ppfData=make_dataarray_real(pData)
    pMeta = np.array(pMeta)
    return ppfData, pMeta

def integrate_disturbances(ppfTrain, pMetaTrain, ppfD, pMetaD):
    idx_no_replaced = list(np.arange(len(ppfTrain)))
    idxD = 0
    while idxD < len(ppfD):
        for i in idx_no_replaced:
            if pMetaD[idxD].compare(pMetaTrain[i]):
                ppfTrain[i] = ppfD[idxD]
                pMetaTrain[i] = pMetaD[idxD]
                idx_no_replaced =idx_no_replaced[i+1:]+idx_no_replaced[:i]
                break

        idxD += 1
    return ppfTrain, pMetaTrain

def make_dataarray_real(frames):
    res = []
    for frame in frames:
            res.append(frame.ppcData)
    res = np.array(res)
    res = np.real(res)
    return res

def prescale_data(ppfData, sScale):
    bInvert = True if sScale.endswith("invert") else False
    if bInvert:
        ppfData = ppfData*-1
    if sScale !="none":
        adjustments = np.load("../../data/sim_adjustments.npz")
        if sScale.startswith("simfactor"):
            factor = adjustments["sim_factor"]
            ppfData = ppfData * factor
        elif sScale.startswith("simoffset"):
            offset = adjustments["sim_offset"]
            ppfData = ppfData + offset
    return ppfData



def add_noise(ppfData, fNoiseSNR=0):
    if fNoiseSNR == 0:
        return ppfData
    # Calc signalpower
    sigmaSignal = np.mean(np.square(np.abs(ppfData)))
    print(f"sigma Signal is {sigmaSignal}")
    sigmaN = sigmaSignal/(10**(np.abs(fNoiseSNR)/10))
    if fNoiseSNR > 0:
        # equally distributed noise Gaussian
        n = np.random.normal(0, np.sqrt(sigmaN), size=ppfData.shape)
        return ppfData + n
    else:
        # mean signal power over all channels is sigmaN*1, but unequally distributed over channels based on real life
        # measurement noise
        noise_distribution = np.load("/data/data_aug/pMeaNoise.npz")  # todo path
        noise_distribution = noise_distribution["pMeaNoise"].flatten()
        Nchannels = len(noise_distribution)
        noise_distribution = noise_distribution * (Nchannels/np.sum(noise_distribution))        # Average power of 1

        n=[]
        for k in range(Nchannels):
            n.append(np.random.normal(scale=noise_distribution[k]* np.sqrt(sigmaN), size=len(ppfData)))
        n= np.array(n).T
        return ppfData + n


def get_model_path(normstrat, baxisdep):
    if baxisdep:
        d= {"none":	"20260814-172510/",
            "zscore"	:"20260420-103734/",
            "minmax":	"20260814-174406/",
            "maxabsolute"	:"20260814-175408/",
            "decimal"	:"20260420-205647/",
            "median":	"20260814-184418/",
            "medmad"	:"20260814-190046/",
            "vss":	"20260421-071902/",
            "sigmoid"	:"20260814-192324/",
            "tanh":"20260421-135903/",
            "hyperbolictangent"	:"20260421-172639/",
            "quantileuniform":	"20260814-195628/",
            "quantilenormal"	:"20260814-200648/",
            "powertransform":	"20260422-003531/"}
        return d[normstrat]


    else:
        d= {"none":	"20260814-172510/",
            "zscore"	:"20260816-124655/",
            "minmax":	"20260816-125744/",
            "maxabsolute"	:"20260816-130658/",
            "decimal"	:"20260422-113250/",
            "median":	"20260422-145814/",
            "medmad"	:"20260816-134053/",
            "vss":	"20260422-215936/",
            "sigmoid"	:"20260423-013250/",
            "tanh":"20260423-050039/",
            "hyperbolictangent"	:"20260816-142549//",
            "quantileuniform":	"20260423-115704/",
            "quantilenormal"	:"20260816-144651/",
            "powertransform":	"20260423-153635/"}
        return d[normstrat]