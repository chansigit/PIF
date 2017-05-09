#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Copyright (c) 2017 yangyuaom. All rights reserved.

# calculation of radiomics features
# 235 features(14 first order statistics + 22 GLCM based + 11 GLRLM based + four-channel wavelet features)
# 2D version, only support rectangle roi, no shape and size based features

import numpy as np

from skimage.feature import greycomatrix, greycoprops

import pywt

# intensity bins & gray levels settings when calculating texture features
HU_BINS = 25
HU_MAX = 400
assert (HU_MAX + 1000) % HU_BINS == 0
DISCRETE_INTENSITY_LEVEL = int((HU_MAX + 1000) / HU_BINS)


# 4 directions for texture features, result as a mean of different directions
# distance is 1 when generating GLCM
# use coif1 in wavelet tranform

# Get the whole feature group name list if header is required in output forms
def GetFGNameList():
    FGIT_NAME = 'energy,entropy,kurtosis,maximum,mean,mean_absolute_deviation,median,minimum,range,RMS,skewness,standard_deviation,uniformity,variance,autocorrelation,cluster_prominence,cluster_shade,cluster_tendency,contrast,correlation,difference_entropy,dissimilarity,energy,entropy,homogeneity1,homogeneity2,IMC1,IMC2,IDMN,IDN,inverse_variance,maximum_probability,sum_average,sum_entropy,sum_variance,variance_GLCM,SRE,LRE,GLN,RLN,RP,LGLRE,HGLRE,SRLGLE,SRHGLE,LRLGLE,LRHGLE'
    FGIT_NAME_LIST = FGIT_NAME.split(',')
    FGW_NAME_LIST = [item + '_LL' for item in FGIT_NAME_LIST] + [item + '_HL' for item in FGIT_NAME_LIST] + [
        item + '_LH' for item in FGIT_NAME_LIST] + [item + '_HH' for item in FGIT_NAME_LIST]
    FG_NAME_LIST = FGIT_NAME_LIST + FGW_NAME_LIST
    return FG_NAME_LIST


# Group 1: First Order statistics
def FGIntensity(X):
    energy = (X ** 2).sum()  # No.1
    maximum = X.max()  # No.4
    median = np.median(X)  # No.7
    minimum = X.min()  # No.8
    rangei = maximum - minimum  # No.9
    RMS = ((X ** 2).mean()) ** 0.5  # No.10
    standard_deviation = X.std(ddof=1)  # No.12
    variance = X.var(ddof=1)  # No.14

    P = np.unique(X, return_counts=True)[1]
    entropy = (P * np.log2(P)).sum()  # No.2
    uniformity = (P ** 2).sum()  # No.13

    mean = X.mean()  # No.5
    k4 = ((X - mean) ** 4).mean()
    k3 = ((X - mean) ** 3).mean()
    k2 = X.var()
    if k2 == 0:
        kurtosis = 0
        skewness = 0
    else:
        kurtosis = k4 / (k2 ** 2)  # No.3
        skewness = k3 / (k2 ** 1.5)  # No.11
    mean_absolute_deviation = (abs(X - mean)).mean()  # No.6

    return [energy, entropy, kurtosis, maximum, mean, mean_absolute_deviation, median, minimum, rangei, RMS, skewness,
            standard_deviation, uniformity, variance]


# Group 3: Textural features

def ResampleIntensity(X):
    temp = X.copy()
    temp[temp < -1000] = -1000
    bins = range(-1000, HU_MAX, HU_BINS)
    return (np.digitize(temp, bins) - 1).astype(dtype=np.float64)


def GLCM(X):
    # set the level or it will be defaulted as 256
    Xint=X.astype(np.uint32)
    return greycomatrix(Xint, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=DISCRETE_INTENSITY_LEVEL, normed=True)


###eg.
##X = np.array([[1,2,5,2,3],[3,2,1,3,1],[1,3,5,5,2],[1,1,1,1,2],[1,2,4,3,5]])-1
##P = GLCM(X)
##print(P[:,:,0,0]);print('----')

def FGGLCM(X):
    P = GLCM(ResampleIntensity(X))  # 4D
    (Ng, Ng2, num_dist, num_angle) = P.shape
    assert Ng == Ng2

    # greycoprops' prop in {'contrast','dissimilarity','ASM','homogeneity'}
    contrast = greycoprops(P, prop='contrast')  # No.27
    dissimilarity = greycoprops(P, prop='dissimilarity')  # No.30
    energy = greycoprops(P, prop='ASM')  # No.31
    homogeneity2 = greycoprops(P, prop='homogeneity')  # No.34

    maximum_probability = np.max(P, axis=(0, 1))  # No.40

    I = np.array(range(1, Ng + 1)).reshape((Ng, 1, 1, 1))
    J = np.array(range(1, Ng + 1)).reshape((1, Ng, 1, 1))

    autocorrelation = np.sum(I * J * P, axis=(0, 1))  # No.23
    homogeneity1 = np.sum(P * 1. / (1 + np.abs(I - J)), axis=(0, 1))  # No.33
    IDN = np.sum(P / (1 + np.abs(I - J) * 1. / Ng), axis=(0, 1))  # No.38
    IDMN = np.sum(P / (1 + ((I - J) * 1. / Ng) ** 2), axis=(0, 1))  # No.37

    temp = np.zeros((Ng, Ng, num_dist, num_angle))
    mask = I != J
    # VisibleDeprecationWarning: boolean index did not match indexed array along dimension 3; dimension is 4 but corresponding boolean dimension is 1
    temp[mask] = P[mask] * 1. / ((I - J) ** 2)[mask]
    inverse_variance = np.sum(temp, axis=(0, 1))  # No.39

    Px = np.sum(P, axis=1, keepdims=True)
    Py = np.sum(P, axis=0, keepdims=True)
    mu_x = np.sum(P * I, axis=1, keepdims=True)
    mu_y = np.sum(P * J, axis=0, keepdims=True)

    cluster_tendency = np.sum(((I - mu_x + J - mu_y) ** 2) * P, axis=(0, 1))  # No.26
    cluster_shade = np.sum(((I - mu_x + J - mu_y) ** 3) * P, axis=(0, 1))  # No.25
    cluster_prominence = np.sum(((I - mu_x + J - mu_y) ** 4) * P, axis=(0, 1))  # No.24
    variance = np.sum((I - mu_x) ** 2 * P, axis=(0, 1))  # No.44

    sigma_x = np.sum((I - mu_x) ** 2 * P, axis=(0, 1)) ** 0.5
    sigma_y = np.sum((J - mu_y) ** 2 * P, axis=(0, 1)) ** 0.5

    temp = np.zeros((num_dist, num_angle))
    mask = sigma_x * sigma_y > 0
    temp[mask] = (np.sum(I * J * P - mu_x * mu_y, axis=(0, 1)))[mask] / (sigma_x * sigma_y)[mask]
    correlation = temp  # No.28

    temp = np.zeros((Ng, 1, num_dist, num_angle))
    mask = Px > 0
    temp[mask] = Px[mask] * np.log2(Px[mask])
    HX = -np.sum(temp, axis=(0, 1))

    temp = np.zeros((1, Ng, num_dist, num_angle))
    mask = Py > 0
    temp[mask] = Py[mask] * np.log2(Py[mask])
    HY = -np.sum(temp, axis=(0, 1))

    temp = np.zeros((Ng, Ng, num_dist, num_angle))
    mask = P > 0
    temp[mask] = P[mask] * np.log2(P[mask])
    HXY = -np.sum(temp, axis=(0, 1))

    temp = np.zeros((Ng, Ng, num_dist, num_angle))
    mask = (Px * Py) > 0
    temp[mask] = P[mask] * np.log((Px * Py)[mask])
    HXY1 = -np.sum(temp, axis=(0, 1))

    temp = np.zeros((Ng, Ng, num_dist, num_angle))
    mask = (Px * Py) > 0
    temp[mask] = (Px * Py)[mask] * np.log((Px * Py)[mask])
    HXY2 = -np.sum(temp, axis=(0, 1))

    entropy = HXY  # No.32

    temp = np.zeros((num_dist, num_angle))
    mask = np.max([HX, HY], 0) > 0
    temp[mask] = (HXY - HXY1)[mask] / np.max([HX, HY], 0)[mask]
    IMC1 = temp  # No.35
    temp = np.zeros((num_dist, num_angle))
    mask = (HXY2 - HXY) > 0
    temp[mask] = (1 - np.e ** (-2 * (HXY2 - HXY)[mask])) ** 0.5
    IMC2 = temp  # No.36

    i, j = np.ogrid[1:Ng + 1, 1:Ng + 1]

    kValuesSum = np.arange(2, (Ng * 2) + 1)
    kValuesDiff = np.arange(0, Ng)
    PxAddy = np.array([np.sum(P[i + j == k], 0) for k in kValuesSum])
    PxSuby = np.array([np.sum(P[np.abs(i - j) == k], 0) for k in kValuesDiff])

    temp = np.zeros((2 * Ng - 1, num_dist, num_angle))
    mask = PxAddy > 0
    temp[mask] = PxAddy[mask] * np.log2(PxAddy[mask])
    sum_entropy = -np.sum(temp, 0)  # No.42

    temp = np.zeros((Ng, num_dist, num_angle))
    mask = PxSuby > 0
    temp[mask] = PxSuby[mask] * np.log2(PxSuby[mask])
    difference_entropy = -np.sum(temp, 0)  # No.29

    sum_average = np.sum((kValuesSum.reshape((2 * Ng - 1, 1, 1)) * PxAddy), 0)  # No.41
    SA = np.sum((kValuesSum.reshape((2 * Ng - 1, 1, 1)) * PxAddy), 0, keepdims=True)
    sum_variance = np.sum((PxAddy * ((kValuesSum.reshape((2 * Ng - 1, 1, 1)) - SA) ** 2)), 0)  # No.43

    result = [autocorrelation, cluster_prominence, cluster_shade, cluster_tendency, contrast, correlation,
              difference_entropy, dissimilarity, energy, entropy, homogeneity1, homogeneity2, IMC1, IMC2, IDMN, IDN,
              inverse_variance, maximum_probability, sum_average, sum_entropy, sum_variance, variance]

    return [item.mean() for item in result]


def GLRLM(X_):
    X=X_.astype(np.uint32)
    (Nh, Nw) = X.shape
    Nr = max(Nh, Nw)
    Ns = min(Nh, Nw)
    Ng = DISCRETE_INTENSITY_LEVEL

    # theta = 0, pi/4, pi/2, 3*pi/4
    P = np.zeros((Ng, Nr, 4))

    # theta = 0, offset as (0,1)
    for i in range(Nh):
        jprev = 0
        for j in range(Nw):
            if X[i, j] != X[i, jprev]:
                rl = j - jprev
                P[X[i, jprev], rl - 1, 0] += 1
                jprev = j
        rl = Nr - jprev
        P[X[i, jprev], rl - 1, 0] += 1

    # theta = pi/2, replace offset as (-1,0) with (1,0)
    for j in range(Nw):
        iprev = 0
        for i in range(Nh):
            if X[i, j] != X[iprev, j]:
                rl = i - iprev
                P[X[iprev, j], rl - 1, 2] += 1
                iprev = i
        rl = Nr - iprev
        P[X[iprev, j], rl - 1, 2] += 1

    # theta = pi/4, offset as (-1,1)
    for pos in range(1, Ns):
        n = Ns - pos
        khprev = 0;
        idxhprev = (Ns - 1 - pos, 0)
        klprev = 0;
        idxlprev = (Ns - 1, pos)

        for k in range(n):
            # higher part
            idxh = (Ns - 1 - pos - k, k)
            if X[idxh] != X[idxhprev]:
                rlh = k - khprev
                P[X[idxhprev], rlh - 1, 1] += 1
                khprev = k;
                idxhprev = idxh

            # lower part
            idxl = (Ns - 1 - k, pos + k)
            if X[idxl] != X[idxlprev]:
                rll = k - klprev
                P[X[idxlprev], rll - 1, 1] += 1
                klprev = k;
                idxlprev = idxl

        rlh = n - khprev;
        P[X[idxhprev], rlh - 1, 1] += 1
        rll = n - klprev;
        P[X[idxlprev], rll - 1, 1] += 1

    # middle part
    for pos in range(Nr - Ns + 1):
        kprev = 0
        if Nh < Nw:
            idxprev = (Ns - 1 - kprev, pos + kprev)
        else:
            idxprev = (Nr - 1 - pos - kprev, kprev)
        for k in range(Ns):
            if Nh < Nw:
                idx = (Ns - 1 - k, pos + k)
            else:
                idx = (Nr - 1 - pos - k, k)
            if X[idx] != X[idxprev]:
                rl = k - kprev
                P[X[idxprev], rl - 1, 1] += 1
                kprev = k;
                idxprev = idx
        rl = Ns - kprev
        P[X[idxprev], rl - 1, 1] += 1

        # theta = 3*pi/4, replace offset as (-1,-1) with (1,1)
    for pos in range(1, Ns):
        n = Ns - pos
        khprev = 0;
        idxhprev = (0, pos)
        klprev = 0;
        idxlprev = (pos, 0)

        for k in range(n):
            # higher half part
            idxh = (k, pos + k)
            if X[idxh] != X[idxhprev]:
                rlh = k - khprev
                P[X[idxhprev], rlh - 1, 3] += 1
                khprev = k;
                idxhprev = idxh

            # lower half part
            idxl = (pos + k, k)
            if X[idxl] != X[idxlprev]:
                rll = k - klprev
                P[X[idxlprev], rll - 1, 3] += 1
                klprev = k;
                idxlprev = idxl

        rlh = n - khprev;
        P[X[idxhprev], rlh - 1, 3] += 1
        rll = n - klprev;
        P[X[idxlprev], rll - 1, 3] += 1

    # middle part
    for pos in range(Nr - Ns + 1):
        kprev = 0
        if Nh < Nw:
            idxprev = (kprev, pos + kprev)
        else:
            idxprev = (pos + kprev, kprev)
        for k in range(Ns):
            if Nh < Nw:
                idx = (k, pos + k)
            else:
                idx = (pos + k, k)
            if X[idx] != X[idxprev]:
                rl = k - kprev
                P[X[idxprev], rl - 1, 3] += 1
                kprev = k;
                idxprev = idx
        rl = Ns - kprev
        P[X[idxprev], rl - 1, 3] += 1

    return P


###eg.
##X = np.array([[5,2,5,4],[3,3,3,1],[2,1,1,1],[4,2,2,2]])-1
##P = GLRLM(X)
##print(P[:,:,0]);print('====');print(P[:,:,2]);print('====')
##print(P[:,:,1]);print('====');print(P[:,:,3]);print('====')

def FGGLRLM(X):
    P = GLRLM(ResampleIntensity(X))
    (Ng, Nr, num_angle) = P.shape

    sum_all = np.sum(P, axis=(0, 1))

    GLN = np.sum((np.sum(P, axis=1)) ** 2, axis=0) * 1. / sum_all  # No.47
    RLN = np.sum((np.sum(P, axis=0)) ** 2, axis=0) * 1. / sum_all  # No.48
    RP = sum_all * 1. / (Nr ** 2)  # No.49

    I = np.array(range(1, Ng + 1)).reshape((Ng, 1, 1))
    J = np.array(range(1, Nr + 1)).reshape((1, Nr, 1))

    SRE = np.sum((P * 1. / (J ** 2)), axis=(0, 1)) / sum_all  # No.45
    LRE = np.sum((P * (J ** 2)), axis=(0, 1)) * 1. / sum_all  # No.46

    LGLRE = np.sum((P * 1. / (I ** 2)), axis=(0, 1)) / sum_all  # No.50
    HGLRE = np.sum((P * (I ** 2)), axis=(0, 1)) * 1. / sum_all  # No.51

    SRLGLE = np.sum((P * 1. / ((I * J) ** 2)), axis=(0, 1)) / sum_all  # No.52
    SRHGLE = np.sum((P * (I ** 2) * 1. / (J ** 2)), axis=(0, 1)) * 1. / sum_all  # No.53

    LRLGLE = np.sum((P * (J ** 2) * 1. / (I ** 2)), axis=(0, 1)) * 1. / sum_all  # No.54
    LRHGLE = np.sum((P * (I ** 2) * (J ** 2)), axis=(0, 1)) * 1. / sum_all  # No.55

    result = [SRE, LRE, GLN, RLN, RP, LGLRE, HGLRE, SRLGLE, SRHGLE, LRLGLE, LRHGLE]
    return [item.mean() for item in result]


# Group 4: Wavelet features
def WT(X):
    coeffs = pywt.swt2(X, 'coif1', 1)
    return coeffs[0]


def FGIT(X):
    return FGIntensity(X) + FGGLCM(X) + FGGLRLM(X)


def FGWavelet(X):
    LL, (HL, LH, HH) = WT(X)
    return FGIT(LL) + FGIT(HL) + FGIT(LH) + FGIT(HH)


# Get the whole feature group
def FG(X):
    # tranfer the format into float, or the calculation may overflow
    # tranfer the box length into even according to swt2's requirement
    if (X.shape[0] % 2 == 1):
        X = X[1:, :]
    if (X.shape[1] % 2 == 1):
        X = X[:, 1:]
    return FGIT(X) + FGWavelet(X)


###==================================================================================
###                Above is Fang Xiang's work, I use an encapsulation of them
###==================================================================================
def featuresV1(X):
    return FGIntensity(X)

def featuresV2(X):
    return FG(X)
