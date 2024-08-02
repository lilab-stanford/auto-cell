import numpy as np
import cv2

def normalizeStaining(I, Io=None, beta=None, alpha=None, HERef=None, maxCRef=None):
    
    
    if Io is None:
        Io = 240

    if beta is None:
        beta = 0.15
    
    if alpha is None:
        alpha = 1
    
    if HERef is None:
        HERef = np.array([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]])
    
    if maxCRef is None:
        maxCRef = np.array([[1.9705], [1.0308]])
    
    h, w = I.shape[:2]
    
    I = I.astype(np.float64)
    
    I = I.reshape([-1, 3])
    
    # optical density
    OD = -np.log((I+1)/Io);
    
    # remove transparent 
    ODhat = OD[np.all(OD > beta, axis=1), :]
    
    # eigen vectors
    eig_v, V = np.linalg.eig(np.cov(ODhat.T))
    V = V[:, np.argsort(eig_v)]
    
    # project onto two largest eigen vectors
    That = ODhat @ V[:, 1:]
    
    # find the min and max vectors and project back to OD space
    phi = np.arctan2(That[:,1], That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = V[:, 1:] @ np.reshape([np.cos(minPhi), np.sin(minPhi)], [2,1])
    vMax = V[:, 1:] @ np.reshape([np.cos(maxPhi), np.sin(maxPhi)], [2,1])

    if vMin[0] > vMax[0]:
        HE = np.concatenate([vMin, vMax], axis=1)
    else:
        HE = np.concatenate([vMax, vMin], axis=1)
        
    Y = OD.reshape([-1, 3]).T
    
    # concentration of individual stains
    C, _, _, _ = np.linalg.lstsq(HE, Y) # C.shape = [2, N]
    
    # normalize stain concentrations
    maxC = np.percentile(C, 99, axis=1)
    
    C = C / np.expand_dims(maxC, axis=-1)
    C = C * maxCRef
    
    Inorm = Io * np.exp(-HERef @ C).T
    Inorm = np.clip(Inorm, a_min=0, a_max=255)
    Inorm = Inorm.reshape([h, w, 3])
    Inorm = Inorm.astype(np.uint8)
    
    return Inorm

