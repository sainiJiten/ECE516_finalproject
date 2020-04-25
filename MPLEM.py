import numpy as np
import headers

def rand_init_chirps(ut,uf,uc,udt,P):
    I = np.zeros((P,4))
    for i in range(P):
        I[i,0] = np.random.uniform(low=0.0,high=ut)
        I[i,1] = np.random.uniform(low=0.0,high=uf)
        I[i,2] = np.random.uniform(low=-1*uc,high=uc)
        I[i,3] = np.random.uniform(low=0.0,high=udt)
        
    return I


def MPLEM(signal,I,I1,P,sr):
    t = np.linspace(0,signal.shape[0]/sr,signal.shape[0])
    sum_chirps = 0
    chirp_signals = np.zeros((P,signal.shape[0]))
    error = 1
    iters = 0
#E Step
    while np.mean(np.abs(error)) > 1e-3:
        for i in range(P):
            g = headers.chirplet(I[i],t)
            a = headers.chirplet_transform(signal,g)
            chirp_signals[i,:] = a*g 
            sum_chirps += a*g
    
        error = signal - sum_chirps
        chirp_signals += error/P
        
#M Step        
        for i in range(P):
            amax = 0
            Imax = 0
            for t in I1[0]:
                for f in I1[1]:
                    for c in I1[2]:
                        for dt in I1[3]:
                            tempI = [t,f,c,dt]
                            g = headers.chirplet(tempI,t)
                            a = headers.chirplet_transform(chirp_signals[i],g)
                            if a > amax:
                                amax = a
                                Imax = [t,f,c,dt]
                            
            I[i] = Imax
        
        iters += 1
        if iters > 100:
            break
        
    return I
        
        

    