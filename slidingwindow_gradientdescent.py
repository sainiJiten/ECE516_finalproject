"""
Intuition behind this method is to create short window of time that will move over the complete signal and 
the assumption is, within each time window, there is only one chirp and hence error = f(t) - ag, where 
tc, dt will be fixed and fc and c will be randomly initialised. Iterations will be done until error is 
minimum or zero.
"""
import numpy as np
import headers
def derivative_fc(ag,t):
    val = -1*np.vdot(ag,1j*t)
    return np.abs(val)

def derivative_c(ag,t):
    val = -1*np.vdot(ag,1j*t**2)
    return np.abs(val)
    

def sw_gd(signal,dt,sr,alpha):
    N = signal.shape[0]
    time = np.linspace(0,N/sr,N)
    nc = int(np.floor(N/dt))
    totalI = np.zeros((nc,4))
    for i in range(nc):
        I = np.zeros((1,4))
        f = np.random.uniform(low=0.0,high=sr/2)
        c= np.random.uniform(low=-100,high=100)
        I = [(i+0.5)*dt/sr,f,c,dt/sr]
        t = time[i*dt:(i+1)*dt]
        error = 1
        iters = 500
        while np.mean(np.abs(error)) > 1e-3 and iters>0:
            g = headers.chirplet(I,t)
            a = headers.chirplet_transform(signal[i*dt:(i+1)*dt],g)
            error = signal[i*dt:(i+1)*dt] - a*g
            
            f = f - alpha*np.mean(np.abs(error))*derivative_fc(a*g,t)
            c = c - alpha*np.mean(np.abs(error))*derivative_c(a*g,t)
            
            I[1] = f
            I[2] = c
            iters -= 1
        
        totalI[i,:] = I

    return totalI
            
        
            
        
    
