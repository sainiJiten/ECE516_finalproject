"""
Common functions used in other programs.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from tftb.processing.cohen import WignerVilleDistribution
import pywt

def chirplet(I,t):
    t = t - I[0]
    g = (1/np.sqrt(np.sqrt(np.pi)*I[3]))*np.exp(-0.5*(t/I[3])**2 + 1j*(I[2]*t+I[1])*t)
    return g

def chirplet_transform(signal,chirplet):
    a = np.abs(np.inner(signal,np.conj(chirplet)))
    return a

def fourier_transform(signal,sr):
    N = signal.shape[0]
    t = np.linspace(0,N/sr,N)
    ff = np.fft.fft(signal)
    T = t[1] - t[0]
    f = np.linspace(0, 1 / T, N)
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency [Hz]")
    plt.bar(f[:N // 2], np.abs(ff)[:N // 2] * 1 / N, width=1.5)  
    plt.show()
   
    
def WVD(I,N,sr):
    tc,fc,c,dt = I
    wvd = np.zeros((N,N))
    Id = np.ones(N)
    t = np.outer(Id, np.arange(N))/(sr)
    f = sr*np.outer(np.arange(N), Id)/(N)
    wvd= 2*np.exp(-1*((t-tc)/dt)**2-(dt*((f-2*fc)-2*c*(t-tc)))**2)
    return wvd


def cal_stft(s,sr,w):
    f,t,S = stft(s,sr,nperseg = w)
    return np.abs(S)
    
def plot_stft(s,sr,w):
    f,t,S = stft(s,sr,nperseg = w)
    plt.pcolormesh(t, f, np.abs(S))
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

   
def plot_wvd_chirps(I,N,sr):
    w = np.zeros((N,N))
    for i in range(I.shape[0]):
        w += WVD(I[i],N,sr)
    w = np.flip(w,axis=0)
    plt.figure()
    plt.imshow(w, cmap="gray",extent=(0, N/sr, 0,sr/2),aspect=2*N/(sr**2))
    plt.xlabel("Time t [sec]")
    plt.ylabel("Frequency f [Hz]")
    plt.show()
    
def plot_wvd(signal):
    wvd = WignerVilleDistribution(signal)
    wvd.run()
    wvd.plot(kind='contour')


def plot_cwt(signal):
    coef, freqs=pywt.cwt(signal,np.arange(1,200),'gaus1')
    plt.matshow(coef)
    plt.show() 
    
def plot_signal(time,signal):
    plt.plot(time, signal)
    plt.title("Signal")
    plt.xlabel('t (sec)')
    plt.ylabel('Amplitude')
    plt.show()
    
    
