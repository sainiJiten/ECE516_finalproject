"""
The function LEM calculates the clusters based on points in TF plane.
Input=>
signal: Time varying amplitude vector
nc: number of gaussian clusters to fit on signal or number of chirplets in the signal
sr: sampling rate of the signal
width: width of logon in Short Time Fourier Transform
Output+>
returns: tc,fc,dt,c of each cluster/chirplet
"""

import numpy as np
import headers
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as mvn

def dtc_finder(l):
    l = np.array(l)
    l = l[l[:,0].argsort(kind='mergesort')]
    dt = l[-1,0] - l[0,0]
    c = (l[-1,1] - l[0,1])/dt
    return dt,c


def lem(signal,nc,sr,width):
    N = signal.shape[0]
    S = headers.cal_stft(signal,sr,width)
    tf = (N/sr)/S.shape[1]
    ff = (sr/2)/S.shape[0]
    points = []
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
#threshold of 0.2 suggests the minimum strength of the signal in STFT 
            if S[i,j] > 0.2:
                points.append([j,i]) 
           
     
    points = np.array(points)
    gmm = GaussianMixture(n_components=nc)
    gmm.fit(points)
    prediction_gmm = gmm.predict(points)

    centers = np.zeros((nc,2))
    for i in range(nc):
        density = mvn(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(points)
        centers[i, :] = points[np.argmax(density)]
    
    centers[:,0] = centers[:,0]*tf
    centers[:,1] = centers[:,1]*ff
    
    group = {}

    for i in range(nc):
        group[i] = []
    
    for i in range(points.shape[0]):
        x = points[i,0]
        y = points[i,1]
        group[prediction_gmm[i]].append([x,y])

    dtc = []

    for i in range(nc):
        dc = dtc_finder(group[i])
        dtc.append(dc)
    
    dtc = np.array(dtc)
    dtc[:,0] = dtc[:,0]*tf
    dtc[:,1] = dtc[:,1]*ff/tf
    return centers,dtc



  