"""
Sample code for using LEM method
"""
import numpy as np
import LEM
import headers
signal = np.load("sample_chirps/sawchirp3.npy")
N = signal.shape[0]
sampling_rate = 500
width = 128 #width of logon in stft
nc = 3 #number of gaussian clusters
centers,dtc = LEM.lem(signal,nc,sampling_rate,width)
I = np.zeros((nc,4))
I[:,0:2] = centers
I[:,2] = dtc[:,1]
I[:,3] = dtc[:,0]/2

headers.plot_stft(signal,sampling_rate,width)
headers.plot_wvd_chirps(I,N,sampling_rate)
