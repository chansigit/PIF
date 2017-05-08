import numpy
from skimage.feature import greycomatrix,greycoprops
import pywt

#Group 1: First Order statistics
def FGIntensity(X):
    energy = (X**2).sum()#No.1
    maximum = X.max()#No.4
    median = numpy.median(X)#No.7
    minimum = X.min()#No.8
    rangei = maximum - minimum#No.9
    RMS = ((X**2).mean())**0.5#No.10
    standard_deviation = X.std(ddof=1)#No.12
    variance = X.var(ddof=1)#No.14

    P = numpy.unique(X, return_counts=True)[1]
    entropy = (P*numpy.log2(P)).sum()#No.2
    uniformity = (P**2).sum()#No.13

    mean = X.mean()#No.5
    k4 = ((X-mean)**4).mean()
    k3 = ((X-mean)**3).mean()
    k2 = X.var()
    if k2 == 0:
        kurtosis = 0
        skewness = 0
    else:
        kurtosis = k4/(k2**2)#No.3
        skewness = k3/(k2**1.5)#No.11
    mean_absolute_deviation = (abs(X-mean)).mean()#No.6

    return [energy,entropy,kurtosis,maximum,mean,mean_absolute_deviation,median,minimum,rangei,RMS,skewness,standard_deviation,uniformity,variance]

#Group 3: Textural features

#intensity bins of 25HU in -1000-400HU, 56 gray levels
def ResampleIntensity(X):
    temp = X.copy()
    temp[temp<-1000] = -1000
    bins = range(-1000,400,25)
    return numpy.digitize(temp,bins)-1

#Get Gray Level Co-occurance Matrix, a mean result of 4 directions at offset=1
def GLCM(X):
    X = ResampleIntensity(X)
    return greycomatrix(X, [1], [0, numpy.pi/4, numpy.pi/2, 3*numpy.pi/4], levels=56, symmetric=True, normed=True)

def FGGLCM(X):
    P = GLCM(X)#4D
    (Ng, Ng2, num_dist, num_angle) = P.shape
    assert Ng == Ng2

    #greycoprops' prop in {'contrast','dissimilarity','ASM','homogeneity'}
    contrast = greycoprops(P, prop='contrast')#No.27
    dissimilarity = greycoprops(P, prop='dissimilarity')#No.30
    energy = greycoprops(P, prop='ASM')#No.31
    homogeneity2 = greycoprops(P, prop='homogeneity')#No.34

    maximum_probability = numpy.max(P,axis=(0,1))#No.40

    I = numpy.array(range(1,Ng+1)).reshape((Ng, 1, 1, 1))
    J = numpy.array(range(1,Ng+1)).reshape((1, Ng, 1, 1))

    autocorrelation = numpy.sum(I * J * P, axis=(0,1))#No.23
    homogeneity1 = numpy.sum(P * 1./(1+numpy.abs(I-J)), axis=(0,1))#No.33
    IDN = numpy.sum(P / (1 + numpy.abs(I-J)*1./Ng), axis=(0,1))#No.38
    IDMN = numpy.sum(P / (1 + ((I-J)*1./Ng)**2), axis=(0,1))#No.37

    temp = numpy.zeros((Ng, Ng, num_dist, num_angle))
    mask = I!=J
    #VisibleDeprecationWarning: boolean index did not match indexed array along dimension 3; dimension is 4 but corresponding boolean dimension is 1
    temp[mask] = P[mask] * 1./((I-J)**2)[mask]
    inverse_variance = numpy.sum(temp, axis=(0,1))#No.39

    Px = numpy.sum(P, axis=1, keepdims=True)
    Py = numpy.sum(P, axis=0, keepdims=True)
    mu_x = numpy.sum(P*I, axis=1, keepdims=True)
    mu_y = numpy.sum(P*J, axis=0, keepdims=True)

    cluster_tendency = numpy.sum(((I-mu_x + J-mu_y)**2) * P, axis=(0,1))#No.26
    cluster_shade = numpy.sum(((I-mu_x + J-mu_y)**3) * P, axis=(0,1))#No.25
    cluster_prominence = numpy.sum(((I-mu_x + J-mu_y)**4) * P, axis=(0,1))#No.24
    variance = numpy.sum((I-mu_x)**2 * P, axis=(0,1))#No.44

    sigma_x = numpy.sum((I-mu_x)**2 * P,  axis=(0,1), keepdims=True)**0.5
    sigma_y = numpy.sum((J-mu_y)**2 * P,  axis=(0,1), keepdims=True)**0.5

    temp = numpy.zeros((Ng, Ng, num_dist, num_angle))
    mask_0 = sigma_x < 1e-15
    mask_0[sigma_y < 1e-15] = True
    mask_1 = mask_0 == False
    temp[mask_1] = (I*J*P-mu_x*mu_y)[mask_1] / (sigma_x*sigma_y)[mask_1]
    correlation = numpy.sum(temp, axis=(0,1))#No.28

    temp = numpy.zeros((Ng, Ng, num_dist, num_angle))
    mask = Px>0
    temp[mask] = Px[mask] * numpy.log2(Px[mask])
    HX = -numpy.sum(temp, axis=(0,1))

    temp = numpy.zeros((Ng, Ng, num_dist, num_angle))
    mask = Py>0
    temp[mask] = Py[mask] * numpy.log2(Py[mask])
    HY = -numpy.sum(temp, axis=(0,1))

    temp = numpy.zeros((Ng, Ng, num_dist, num_angle))
    mask = P>0
    temp[mask] = P[mask] * numpy.log2(P[mask])
    HXY = -numpy.sum(temp, axis=(0,1))

    temp = numpy.zeros((Ng, Ng, num_dist, num_angle))
    mask = (Px*Py)>0
    temp[mask] = P[mask] * numpy.log((Px*Py)[mask])
    HXY1 = -numpy.sum(temp, axis=(0,1))

    temp = numpy.zeros((Ng, Ng, num_dist, num_angle))
    mask = (Px*Py)>0
    temp[mask] = (Px*Py)[mask] * numpy.log((Px*Py)[mask])
    HXY2 = -numpy.sum(temp, axis=(0,1))

    #import pdb;pdb.set_trace()
    entropy = HXY#No.32
    IMC1 = (HXY-HXY1) / numpy.max([HX,HY], axis=(0,1))#No.35
    IMC2 = (1 - numpy.e**(-2*(HXY2-HXY)))**0.5#No.36

    i, j = numpy.ogrid[1:Ng+1, 1:Ng+1]

    kValuesSum = numpy.arange(2, (Ng * 2) + 1)
    kValuesDiff = numpy.arange(0, Ng)
    PxAddy = numpy.array([numpy.sum(P[i+j==k], 0) for k in kValuesSum])
    PxSuby = numpy.array([numpy.sum(P[numpy.abs(i-j)==k], 0) for k in kValuesDiff])

    temp = numpy.zeros((2*Ng-1, num_dist, num_angle))
    mask = PxAddy>0
    temp[mask] = PxAddy[mask] * numpy.log2(PxAddy[mask])
    sum_entropy = -numpy.sum(temp, 0)#No.42

    temp = numpy.zeros((Ng, num_dist, num_angle))
    mask = PxSuby>0
    temp[mask] = PxSuby[mask] * numpy.log2(PxSuby[mask])
    difference_entropy = -numpy.sum(temp, 0)#No.29

    sum_average = numpy.sum((kValuesSum.reshape((2*Ng-1,1,1)) * PxAddy), 0)#No.41
    SA = numpy.sum((kValuesSum.reshape((2*Ng-1,1,1)) * PxAddy), 0, keepdims=True)
    sum_variance = numpy.sum((PxAddy * ((kValuesSum.reshape((2*Ng-1,1,1)) - SA) ** 2)), 0)#No.43

    #result = [autocorrelation,cluster_prominence,cluster_shade,cluster_tendency,contrast,correlation,difference_entropy,dissimilarity,energy,entropy,homogeneity1,homogeneity2,IMC1,IMC2,IDMN,IDN,inverse_variance,maximum_probability,sum_average,sum_entropy,sum_variance,variance]
    result = [autocorrelation,cluster_prominence,cluster_shade,cluster_tendency,contrast,correlation,dissimilarity,energy,entropy,homogeneity1,homogeneity2]#,IMC1,IDMN,IDN,inverse_variance,maximum_probability,sum_average,sum_entropy,sum_variance,variance]

    return [item.mean() for item in result]
###eg.
##X = numpy.array([[1,2,5,2,3],[3,2,1,3,1],[1,3,5,5,2],[1,1,1,1,2],[1,2,4,3,5]])-1
##rf = FGGLCM(X)

def GLRLM(X):
    X = ResampleIntensity(X)
    (Nr,Nr1) = X.shape
    Ng = X.max()-X.min()+1

    #theta = 0, pi/4, pi/2, 3*pi/4
    P = numpy.zeros((Ng,Nr,4),dtype=numpy.uint)

    #theta = 0, offset as (0,1)
    for i in range(Nr):
        jprev = 0
        for j in range(Nr):
            if X[i,j] != X[i,jprev]:
                rl = j-jprev
                P[X[i,jprev],rl-1,0] += 1
                jprev = j
        rl = Nr-jprev
        P[X[i,jprev],rl-1,0] += 1

    #theta = pi/4, offset as (-1,1)
    for pos in range(Nr):
        n = Nr-pos
        khprev = 0; idxh0 = (Nr-1-pos,0); idxhprev = idxh0
        klprev = 0; idxl0 = (Nr-1,pos); idxlprev = idxl0

        for k in range(n):
            #higher half part
            idxh = (Nr-1-pos-k,k)
            if X[idxh]!=X[idxhprev]:
                rlh = k-khprev
                P[X[idxhprev],rlh-1,1] += 1
                khprev = k; idxhprev = idxh

            if pos>0:
                #lower half part
                idxl = (Nr-1-k,pos+k)
                if X[idxl]!=X[idxlprev]:
                    rll = k-klprev
                    P[X[idxlprev],rll-1,1] += 1
                    klprev = k; idxlprev = idxl

        rlh = n-khprev; P[X[idxhprev],rlh-1,1] += 1
        if pos>0:
            rll = n-klprev; P[X[idxlprev],rll-1,1] += 1

    #theta = pi/2, replace offset as (-1,0) with (1,0)
    for j in range(Nr):
        iprev = 0
        for i in range(Nr):
            if X[i,j] != X[iprev,j]:
                rl = i-iprev
                P[X[iprev,j],rl-1,2] += 1
                iprev = i
        rl = Nr-iprev
        P[X[iprev,j],rl-1,2] += 1

    #theta = 3*pi/4, replace offset as (-1,-1) with (1,1)
    for pos in range(Nr):
        n = Nr-pos
        khprev = 0; idxh0 = (0,pos); idxhprev = idxh0
        klprev = 0; idxl0 = (pos,0); idxlprev = idxl0

        for k in range(n):
            #higher half part
            idxh = (k,pos+k)
            if X[idxh]!=X[idxhprev]:
                rlh = k-khprev
                P[X[idxhprev],rlh-1,3] += 1
                khprev = k; idxhprev = idxh

            if pos>0:
                #lower half part
                idxl = (pos+k,k)
                if X[idxl]!=X[idxlprev]:
                    rll = k-klprev
                    P[X[idxlprev],rll-1,3] += 1
                    klprev = k; idxlprev = idxl

        rlh = n-khprev; P[X[idxhprev],rlh-1,3] += 1
        if pos>0:
            rll = n-klprev; P[X[idxlprev],rll-1,3] += 1

    return P

def FGGLRLM(X):
    P = GLRLM(X)
    (Ng,Nr,num_angle) = P.shape

    sum_all = numpy.sum(P,axis=(0,1))

    GLN = numpy.sum((numpy.sum(P,axis=1))**2,axis=0)*1./sum_all#No.47
    RLN = numpy.sum((numpy.sum(P,axis=0))**2,axis=0)*1./sum_all#No.48
    RP = sum_all*1./(Nr**2)#No.49

    I = numpy.array(range(1,Ng+1)).reshape((Ng, 1, 1))
    J = numpy.array(range(1,Nr+1)).reshape((1, Nr, 1))

    SRE = numpy.sum((P*1./(J**2)),axis=(0,1))/sum_all#No.45
    LRE = numpy.sum((P*(J**2)),axis=(0,1))*1./sum_all#No.46

    LGLRE = numpy.sum((P*1./(I**2)),axis=(0,1))/sum_all#No.50
    HGLRE = numpy.sum((P*(I**2)),axis=(0,1))*1./sum_all#No.51

    SRLGLE = numpy.sum((P*1./((I*J)**2)),axis=(0,1))/sum_all#No.52
    SRHGLE = numpy.sum((P*(I**2)*1./(J**2)),axis=(0,1))*1./sum_all#No.53

    LRLGLE = numpy.sum((P*(J**2)*1./(I**2)),axis=(0,1))*1./sum_all#No.54
    LRHGLE = numpy.sum((P*(I**2)*(J**2)),axis=(0,1))*1./sum_all#No.55

    result = [SRE,LRE,GLN,RLN,RP,LGLRE,HGLRE,SRLGLE,SRHGLE,LRLGLE,LRHGLE]
    return [item.mean() for item in result]

###eg.
##X = numpy.array([[5,2,5,4],[3,3,3,1],[2,1,1,1],[4,2,2,2]])-1
##rf = FGGLRLM(X)

#Group 4: Wavelet features
def WT(data):
    coeffs = pywt.swt2(data,'coif1',1)
    return coeffs[0]

def FGIT(X):
    return FGIntensity(X)+FGGLCM(X)+FGGLRLM(X)

def FGWavelet(X):
    LL,(HL,LH,HH) = WT(X)
    return FGIT(LL)+FGIT(HL)+FGIT(LH)+FGIT(HH)

###eg.
##X = numpy.ones((40,40),dtype=numpy.uint);X[:,0]=0
##import pdb;pdb.set_trace()
##rf = FGIT(X)

#Calculate IOU
def IOU_LND(ReOP,ww,GTframe):
    x1 = ReOP[0]
    y1 = ReOP[1]
    width1 = ww
    height1 = ww

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]

    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1+width2-(endx-startx)

    endy = max(y1+height1,y2+height2)
    starty = min(y1,y2)
    height = height1+height2-(endy-starty)

    if width <=0 or height <= 0:
        ratio = 0
        ratio2 = 0
    else:
        Area = width*height
        Area1 = width1*height1
        Area2 = width2*height2
        ratio = Area*1./Area2
        ratio = ratio*max(width2*1./width1,1)*max(height2*1./height1,1)

    return ratio

###==================================================================================
###                Above is Fang Xiang's work, I use an encapsulation of them
###==================================================================================
def featuresV1(X):
    return FGIntensity(X)
