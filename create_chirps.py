from scipy.signal import chirp, stft
import matplotlib.pyplot as plt
import numpy as np
import headers

def create_linear(t,sampling_rate,sf,ef,te):
    time = np.linspace(0,t,sampling_rate*t)
    w = chirp(time, f0=sf, f1=ef, t1=te, method='linear')
    return w
    
sr = 500
s0 = np.zeros(sr)
w = create_linear(1,sr,100,150,1)
signal = np.concatenate((s0,w,s0),axis=0)
time = np.linspace(0,signal.shape[0]/sr,signal.shape[0])

headers.plot_signal(time,signal)
headers.plot_stft(signal,sr,128)




    
    