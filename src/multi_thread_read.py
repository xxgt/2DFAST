import threading
import numpy as np
import cupy as cp
import copy
from readers import FilReader

class MultiThreadReader(threading.Thread):
    """
        to read the fil file with multithreads.
    """
    def __init__(self, fname, blockStart, nsamp):
        super(MultiThreadReader, self).__init__()
        self.fil = FilReader(fname)
        self.blockStart = blockStart
        self.nsamp = nsamp


    def run(self):
        data, _ = self.fil.read_block(self.blockStart, self.nsamp)
        self.result = cp.array(data).astype(np.float32)
    
    def getResult(self):
        return self.result
    

def readFileBlock(fname, tstart, nsamp, nthreads):
    nPerThread = int(nsamp / nthreads)
    threadList = []
    resList = []
    for i in range(nthreads):
        threadList.append(MultiThreadReader(fname, tstart + i*nPerThread, nPerThread))
        threadList[i].start()
    
    for i in range(nthreads):
        threadList[i].join()
        resList.append(threadList[i].getResult())

    result = cp.concatenate(resList, axis=-1)
    return result

def threadRead(fil, start, nsamp):
    data, _ = fil.read_block(start, nsamp)
    data_cuda = cp.array(data).astype(np.float32)
    return data_cuda