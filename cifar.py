import cPickle
import numpy as np

CIFAR_PATH = './cifar-10-batches-py'
CIFAR_FILES = ('data_batch_1', 
               'data_batch_2',
               'data_batch_3',
               'data_batch_4',
               'data_batch_5')

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load():
    data = np.zeros((50000, 3072), 'uint8')
    for i in range(len(CIFAR_FILES)):
        dict = unpickle(CIFAR_PATH + '/' 
                        + CIFAR_FILES[i])
        data[10000*i:10000*(i+1),:] = dict['data']
        
    return data
