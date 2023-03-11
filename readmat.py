from os.path import dirname, join as pjoin
import scipy.io as sio
import os
import numpy as np
def readmat(datapath,category):

    data_dir = pjoin(dirname(sio.__file__), datapath)
    finaldatamatrix = []
    finallabelmatrix = []
    labelvector =[]
    for item in os.listdir(data_dir):
        pat = '.mat'
        if pat in item:
            itempath = pjoin(data_dir, item)
            mat_contents = sio.loadmat(itempath)
            # datamatrix = abs(mat_contents['ROICorrelation_FisherZ'])
            datamatrix = abs(mat_contents['subcordata'])
            datamatrix = np.where(np.isinf(datamatrix), 1, datamatrix)
            datamatrix = np.where(np.isnan(datamatrix), 0, datamatrix)
            if category == 'MCI':
                labelvector = np.zeros(np.shape(datamatrix)[0])
                # labelvector=0
            elif category == 'NC':
                labelvector = np.ones(np.shape(datamatrix)[0])
                # labelvector=1
            finaldatamatrix.append(datamatrix)
            finallabelmatrix.append(labelvector)
    return finaldatamatrix, finallabelmatrix

