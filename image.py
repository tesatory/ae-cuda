import numpy as np
import matplotlib.pyplot as plt

def show(data):
    n = data.shape[0]
    m = data.shape[1]
    s = np.ceil(np.sqrt(n))
    
    if np.floor(np.sqrt(m))**2 == m:
        sz = np.floor(np.sqrt(m))
        img = np.zeros((sz * s, sz * s), data.d.type)
        for i in range(n):
            x = i % s
            y = np.floor(i / s)
            img[sz*y:sz*(y+1),
                sz*x:sz*(x+1)] = data[i,0:sz**2].reshape((sz,sz))
        if data.dtype == 'uint8':
            plt.imshow(img, cmap=plt.cm.gray)
        else:
            img = 1.0 * (img - img.min()) / (img.max() - img.min())
            plt.imshow(img, cmap=plt.cm.gray)
    else:
        sz = np.floor(np.sqrt(m/3))
        img = np.zeros((sz * s, sz * s, 3), data.dtype)
        for i in range(n):
            x = i % s
            y = np.floor(i / s)
            img[sz*y:sz*(y+1),
                sz*x:sz*(x+1),0] = data[i,0:sz**2].reshape((sz,sz))
            img[sz*y:sz*(y+1),
                sz*x:sz*(x+1),1] = data[i,sz**2:2*sz**2].reshape((sz,sz))
            img[sz*y:sz*(y+1),
                sz*x:sz*(x+1),2] = data[i,2*sz**2:3*sz**2].reshape((sz,sz))
        if data.dtype == 'uint8':
            plt.imshow(img)
        else:
            img = 1.0 * (img - img.min()) / (img.max() - img.min())
            plt.imshow(img)

def prepare_patches(data, psz, pnum):
    n = data.shape[0]
    m = data.shape[1]
    sz = np.floor(np.sqrt(m/3))

    patches = np.zeros((pnum, 3*psz**2), 'uint8')
    for i in range(pnum):
        dx = np.random.randint(sz - psz +1)        
        dy = np.random.randint(sz - psz +1)
        img = data[i % n,:].reshape((3,sz,sz))
        patch = img[:,dy:dy+psz,dx:dx+psz]
        patches[i,:] = patch.reshape((1, 3*psz**2))

    return patches

def normalize(data):
    data = data - data.mean(0)
    data = data / data.std(0)
    return data

def whiten(data):
    C = np.cov(data, rowvar=0)
    m = np.mean(data, 0)
    (W,V) = np.linalg.eig(C)
    P = np.dot(np.dot(V, np.diag(np.sqrt(1. / (W + 0.1)))), 
               V.transpose())
    
    return np.dot(data - m, P)
