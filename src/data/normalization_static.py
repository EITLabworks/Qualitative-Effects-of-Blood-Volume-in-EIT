"""
Project ：quali_effects_eit_measurements
Directory: src/ultrasound/normalization
File : normalization_static.py.py
Author ：Patricia Fuchs
Date ：16.12.2025 11:20
"""

import numpy as np
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
import matplotlib.pyplot as plt
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

sigmoid_l = lambda d, m, s: 1/(1+ np.exp((-d - m)/s))
inv_sigmoid = lambda d, m,s: (np.log((1/d)-1)*-1*s)-m

tanh_l= lambda d,m,s: 0.5 * (np.tanh(0.01* (d-m)/s)+1 )
inv_tanh = lambda d, m ,s : ((np.arctanh(2*d)-1)*100*s)+m


hyper_l = lambda d,m,s:(1-np.exp(-1 * (d-m)/s))/ (1+np.exp(-1 * (d-m)/s))
inv_hyper= lambda d,m,s: m+ s*-1*np.log((1-d)/(1+d))


class NormMult:
    def __init__(self, factor, offset):
        self.Factor = factor
        self.Offset = offset

    def transform(self, ppfData):
        return (ppfData-self.Offset)/self.Factor

    def inverse_transform(self, ppfData):
        return (ppfData * self.Factor) + self.Offset


class NormSpecial:
    def __init__(self, fct_norm, fct_inv, m, s):
        self.fct_norm = fct_norm
        self.fct_inv= fct_inv
        self.m = m
        self.s = s

    def transform(self, ppfData):
        return self.fct_norm(ppfData, self.m, self.s)

    def inverse_transform(self, ppfData):
        return self.fct_inv(ppfData, self.m, self.s)

class NormIndependent:
    def __init__(self, tfarray):
        self.Transformerarray = tfarray
        self.L= len(self.Transformerarray)

    def transform(self, ppfData):
        ppfDataNormed= np.zeros(ppfData.shape)
        for i in range(self.L):
            ppfDataNormed[:, i] = self.Transformerarray[i].transform(ppfData[:, i].reshape(-1, 1)).flatten()
        return ppfDataNormed


    def inverse_transform(self, ppfData):
        ppfDataNormed= np.zeros(ppfData.shape)
        for i in range(self.L):
            ppfDataNormed[:, i] = self.Transformerarray[i].inverse_transform(ppfData[:, i].reshape(-1, 1)).flatten()
        return ppfDataNormed

class NormDependent:
    def __init__(self, transformer):
        self.transformer= transformer

    def transform(self, ppfData):
        shape= ppfData.shape
        ppfDataNormed = self.transformer.transform(ppfData.flatten().reshape(-1, 1))
        ppfDataNormed= ppfDataNormed.reshape(shape)
        return ppfDataNormed


    def inverse_transform(self, ppfData):
        shape= ppfData.shape
        ppfDataNormed = self.transformer.inverse_transform(ppfData.flatten().reshape(-1, 1))
        ppfDataNormed= ppfDataNormed.reshape(shape)
        return ppfDataNormed


def zscore(ppfEIT, axis= (0)):
    #axis = 0 indepent channels, axis = [0,1], dependent channels
    mx = np.mean(ppfEIT, axis=axis)
    sx = np.std(ppfEIT, axis=axis)
    return (ppfEIT-mx)/sx, NormMult(sx, mx)


def min_max_scaling(ppfEIT, axis = (0)):
    mini = np.min(ppfEIT, axis=axis)
    maxi = np.max(ppfEIT, axis=axis)
    return (ppfEIT-mini)/(maxi-mini), NormMult(maxi-mini, mini)

def max_absolute_scaling(ppfEIT, axis=(0)):
    maxi = np.abs(np.max(ppfEIT, axis=axis))
    return ppfEIT/ maxi, NormMult(maxi, 0)


def decimal_scaling(ppfEIT, axis=(0)):
    mini = np.min(ppfEIT,axis=axis)
    pot = np.ceil(np.log10(np.max(ppfEIT-mini,axis=axis)))
    return ppfEIT/(10**pot), NormMult((10**pot), 0)

def median_normalization(ppfEIT, axis=(0)):
    med = np.median(ppfEIT, axis=axis)
    return ppfEIT/med, NormMult(med, 0)

def med_and_mad_normalization(ppfEIT, axis= (0)):
    med = np.median(ppfEIT,axis=axis)
    ad = np.abs(ppfEIT-med)
    mad = np.median(ad, axis=axis)
    return (ppfEIT-med)/ mad, NormMult(mad, med)

def variable_stability_scaling(ppfEIT, axis=(0)):
    mx = np.mean(ppfEIT, axis=axis)
    sx = np.std(ppfEIT, axis=axis)
    return (ppfEIT-mx)/sx * (mx/sx), NormMult(sx*(mx/sx), mx)


def sigmoid_normalization(ppfEIT,axis=(0)):
    mx = np.mean(ppfEIT, axis=axis)
    sx = np.std(ppfEIT, axis=axis)
    return 1/(1+ np.exp((-ppfEIT - mx)/sx)), NormSpecial(sigmoid_l, inv_sigmoid, mx, sx)

#todo
def tanh_normalization(ppfEIT, axis = (0)):
    mx = np.mean(ppfEIT, axis=axis)
    sx = np.std(ppfEIT, axis=axis)
    return 0.5 * (np.tanh(0.01* (ppfEIT-mx)/sx)+1), NormSpecial(tanh_l, inv_tanh, mx, sx)


#todo
def hyperbolic_tangent_normalization(ppfEIT, axis=(0)):
    mx = np.mean(ppfEIT, axis=axis)
    sx = np.std(ppfEIT, axis=axis)
    e = np.exp(-1 * (ppfEIT-mx)/sx)
    return (1-e)/ (1+e), NormSpecial(hyper_l, inv_hyper, mx, sx)



def quantile_transform_uniform(ppfEIT, axis=(0)):
    if axis == None or axis == (None):
        shape = ppfEIT.shape
        Qt = QuantileTransformer(n_quantiles=100, output_distribution="uniform")
        Qt.fit(ppfEIT.flatten().reshape(-1, 1))
        ppfNormed= Qt.transform(ppfEIT.flatten().reshape(-1, 1))
        return ppfNormed.reshape(shape), NormDependent(Qt)
    else:
        Qt = QuantileTransformer(n_quantiles=100, output_distribution="uniform")
        Qt.fit(ppfEIT)
        return Qt.transform(ppfEIT), Qt



def quantile_transform_normal(ppfEIT, axis=(0)):
    if axis == None or axis == (None):
        shape = ppfEIT.shape
        Qt = QuantileTransformer(n_quantiles=100, output_distribution="normal")
        Qt.fit(ppfEIT.flatten().reshape(-1, 1))
        ppfNormed= Qt.transform(ppfEIT.flatten().reshape(-1, 1))
        return ppfNormed.reshape(shape), NormDependent(Qt)
    else:
        Qt = QuantileTransformer(n_quantiles=100, output_distribution="normal")
        Qt.fit(ppfEIT)
        return Qt.transform(ppfEIT), Qt





def power_transform(ppfEIT, axis=(0)):
    if axis == None or axis == (None):
        shape = ppfEIT.shape
        pt = PowerTransformer()
        pt.fit(ppfEIT.flatten().reshape(-1, 1))
        ppfNormed= pt.transform(ppfEIT.flatten().reshape(-1, 1))
        return ppfNormed.reshape(shape), NormDependent(pt)
    else:
        pt = PowerTransformer()
        pt.fit(ppfEIT)
        return pt.transform(ppfEIT), pt




def none_normalization(ppfEIT, axis=(0)):
    return ppfEIT, NormMult(1,0)


norm_dict= {
    "none": none_normalization,
    "zscore":zscore,
    "minmax": min_max_scaling,
    "maxabsolute": max_absolute_scaling,
    "decimal": decimal_scaling,
    "median": median_normalization,
    "medmad": med_and_mad_normalization,
    "vss":variable_stability_scaling,
    "sigmoid": sigmoid_normalization,
    "tanh":tanh_normalization,
    "hyperbolictangent": hyperbolic_tangent_normalization,
    "quantileuniform": quantile_transform_uniform,
    "quantilenormal": quantile_transform_normal,
    "powertransform": power_transform
}



class EITNormalizationStatic:
    def __init__(self, normtype, axisdependent=False):
        self.normtype= normtype
        self.norm_fct= norm_dict[normtype]
        self.bAxisDependent= axisdependent
        self.Transformer= None
        if self.bAxisDependent:
            self.axis =(None)
        else:
            self.axis = (0)


    def normalize_forward(self, ppfEIT, bUseFactors=False):
        if not bUseFactors:
            ppfNormalized, self.Transformer= self.norm_fct(ppfEIT,self.axis)
        elif self.Transformer == None:
            raise ValueError("No Normalization Parameters are given.")
        else:
            ppfNormalized =self.Transformer.transform(ppfEIT)
        return ppfNormalized

    def inverse_normalize(self, ppfEIT):
        try:
            return self.Transformer.inverse_transform(ppfEIT)
        except:
            raise ValueError("No forward transformation was given that could have been reversed.")

