"""
Matching Pursuit Algorithm:
Exhaustive search of parameters can be done to find the parameters where strength of the chirplet transform 
is maximum at each level.
Inputs: 
signal : Time series signal, 
I: list of lists of four parameters, I = [[t1,...,tn],[f1,...fn],[c1,....,cn],[dt1,.....,dtn]],
sr : sampling rate of the signal
P: Number of chirplets in the signal

Outputs:
[tc,fc,c,dt] of each chirplet
Visualization:
Approximated signal can be seen via headers.plot_wvd_chirps(chirp_params,signal.shape[0],sampling rate)

"""
import numpy as np
import headers
def matching_pursuit(signal,I,sr,P):
    time = np.linspace(0,signal.shape[0]/sr,signal.shape[0])
    chirps_params = np.zeros((P,4))
    for i in range(P):
        amax = 0
        Imax = [0,0,0,0]
        for t in I[0]:
            for f in I[1]:
                for c in I[2]:
                    for dt in I[3]:
                        tempI = [t,f,c,dt]
                        g = headers.chirplet(tempI,time)
                        a = headers.chirplet_transform(signal,g)
                        if a > amax:
                            amax = a
                            Imax = [t,f,c,dt]
        
        signal = signal - amax*headers.chirplet(Imax,time)                    
        chirps_params[i,:] = Imax
    return chirps_params

