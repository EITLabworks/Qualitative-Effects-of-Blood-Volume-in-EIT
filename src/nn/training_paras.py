"""
Project ：Qualitative-Effects-Blood-Volume-EIT
Directory: src/nn
File : training_paras.py
Author ：Patricia Fuchs
Date ：26.01.2026 11:04
"""
import argparse


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
class TrainingParas:
    def __init__(self):
        self.sNWType = "ann"
        self.iNel = 32
        self.iDim = 32 * 32
        self.fR = 9
        self.iInjSkip = 4
        self.sMeaPattern = "skip4internal"
        self.sPosfactor = "none"
        self.sNormtype = "zscore"
        self.bAxisdep = False
        self.bReshape = False
        self.iBatchsize = 16
        self.iEpochs = 150
        self.sModelSelection = "none"

        self.sConfigFile = "/config_model1.json"
        self.sDataConfig = "config1"
        self.iAugFactor = 0

    # ------------------------------------------------------------------------------------------------------------------- #
    def write_configs(self, f):
        """
        Writes used parameter configuration to file f
        """
        f.write("Epochs: " + str(self.iEpochs) + "\n")
        f.write("Batchsize: " + str(self.iBatchsize) + "\n")
        f.write(f"Network type: {self.sNWType} \n")
        f.write(f"Number Electrodes: {self.iNel} \n")
        f.write(f"Position Factor: {self.sPosfactor} \n")
        f.write("Normalization strategy: " + str(self.sNormtype) + "\n")
        f.write(f"bAxisdependent: {self.bAxisdep} \n")
        f.write(f"sConfigfile: {self.sConfigFile} \n")
        f.write(f"sDataConfig: {self.sDataConfig} \n")
        f.write("Model Selection: " + str(self.sModelSelection) + "\n")
        f.write("iAugFactor: " + str(self.iAugFactor) + "\n")


def parse_arguments(bParseArgs: bool):
    if not bParseArgs:
        return TrainingParas()
    else:
        parser = argparse.ArgumentParser(description="Crossvalidation")
        parser.add_argument("-b", "--batchsize", type=int, help="batchsize", default=16)
        parser.add_argument("-e", "--epochs", type=int, help="epochs", default=300)

        parser.add_argument("-cf", "--config", type=str, help="Config file", default="/config_model1.json")
        parser.add_argument("-dc", "--dataconfig", type=str, help="Data Config file", default="config1")
        parser.add_argument("-nt", "--norm", type=str, help="normalize EIT", default="zscore")
        parser.add_argument("-pf", "--posfac", type=str, help="Positioning Factor EIT", default="none")
        parser.add_argument("-ba", "--axisdep", type=bool, help="bAxisdep norm", default=False)
        parser.add_argument("-nw", "--nwtype", type=str, help="NW Type", default="ann_reg")
        parser.add_argument("-ms", "--modelselection", type=str,
                            help="Select a different model than the standard config", default="none")

        parser.add_argument("-af", "--augfac", type=int, help="Augmentation factor", default=0)
        args = parser.parse_args()
        print(args)
        P = TrainingParas()

        P.iBatchsize = args.batchsize

        P.iEpochs = args.epochs
        P.sConfigFile = args.config
        P.sDataConfig = args.dataconfig

        P.sNormtype = args.norm
        P.sPosfactor = args.posfac
        P.bAxisdep = args.axisdep
        P.sNWType = args.nwtype
        P.iAugFactor = args.augfac
        if P.sNWType.startswith("cnn"):
            P.bReshape = True

        P.sModelSelection = args.modelselection
        return P
