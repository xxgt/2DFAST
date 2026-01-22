import numpy as np
import cupy as cp
import math
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#import gc

def rebin(data,fy, nbin,tx=0):
	"""
	.. Note::
			Use histogram method
	Change the freqency axis to wave squre axis
	The signal line will become a straight line after this process.

	:param:
	data:  Input data require I(f,t) formate
	fy	   frequency ^ -2 
	nbin   How many channel of bins left, def =Nch
	tx	   Time axis
	:Output 
	data:  data after rebin with shape of (nbin,t)
	faxis: f axis after rebin
	"""
	f_axis = np.linspace(fy.min(),fy.max(),nbin)
	b_aray, _ = np.histogram(fy,bins=nbin)
	data = np.nan_to_num(data)
	data1 = np.zeros((nbin,data.shape[1]),data.dtype)
	for i in np.arange(nbin):
			length  =  b_aray[i]
			if i == 0:
				index = 0
			elif i == 1:
				index = b_aray[0]
			else:
				index = b_aray[0:i].sum()#when i = 1 ,the bin_array[0:1] doesn't include the bin_array[1]
			for ii in np.arange(length):
					data1[i,:]+=data[ii+index,:]
	tem = np.ones(data.shape[1])
	bin_weight = b_aray[:,None] * tem[None,:]
	data1 = data1 / 1.0 / bin_weight
	data1 = np.nan_to_num(data1)
	#for i in np.arange(5):
	#           y_s     =  data1.sum( axis = 1 )
	#           y_max   =  np.argmax(y_s)
	#           data1[y_max-10:y_max+11,:]=0
	#    del tem,data
	#    gc.collect()

	return data1, f_axis

def rebin_inter(data, fy, nbin, tx):
	"""
	.. Note::
				Use histogram method
	Change the freqency axis to wave squre axis
	The signal line will become a straight line after this process.

	:param:
	data:  Input data require I(f,t) formate
	fy     frequency ^ -2 
	nbin   How many channel of bins left, def =Nch
	tx 	   time axis

	:Output 
	data:  data after rebin with shape of (nbin,t)
	faxis: f axis after rebin
	"""
	
	f		= np.linspace(fy.min(),fy.max(),nbin)
	b_aray 	= fy
	data	= interp1d(b_aray,data,axis=0)
	data	= data(f)
	data	= np.nan_to_num(data)
	#    for i in np.arange(5):
	#               y_s     =  data.sum( axis = 1 )
	#               y_max   =  np.argmax(y_s)
	#               data[y_max-2:y_max+2,:]=0

	#	       x_s     =  data.sum( axis = 0 )
	#               x_max   =  np.argmax(x_s)
	#               data[:,x_max-2:x_max+2]=0

	return data, f

code_1 = r"""
	extern "C"
	__global__ void interp(float* data, float* x, float* new_x, int* idx, float* y, int nbin, int tlen) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < nbin*tlen) {
			int row = tid / nbin;
			int col = tid % nbin;
			if (col == 0 || col == nbin - 1) {
				y[tid] = data[tid];
			}
			else {
				int x_idx = idx[col] - 1;
				int lo = row * nbin + x_idx;
				int hi = lo + 1;
				float ratio = (new_x[col] - x[x_idx])/(x[x_idx + 1] - x[x_idx]);
				y[tid] = (data[hi] - data[lo]) * ratio + data[lo];
			}
		}
	}
"""

code = r"""
	extern "C"
	__global__ void interp(float* data, float* x, float* new_x, int* idx, float* y, int nbin, int tlen) {
		int row = blockIdx.x;
		int col = blockIdx.y * blockDim.x + threadIdx.x;
		if (row < nbin && col < tlen) {
			int tid = row * tlen + col;
			if (row == 0 || row == nbin - 1) {
				y[tid] = data[tid];
			}
			else {
				int x_idx = idx[row] - 1;
				int lo = tlen * x_idx + col;
				int hi = tlen * (x_idx + 1) + col;
				float ratio = (new_x[row] - x[x_idx])/(x[x_idx + 1] - x[x_idx]);
				y[tid] = (data[hi] - data[lo]) * ratio + data[lo];
			}
		}
		
	}
"""
interp_kernel = cp.RawKernel(code_1, 'interp')

def rebin_inter_cuda(data, fy, f, idx, nbin, t_len):
	"""
		using linear interpolation method on cuda
		Change the freqency axis to wave square axis
		The signal line will become a straight line after this process.

	Args:
		data	(cupy.ndarray): 2D input data in I(f, t) format
		fy		(cupy.ndarray):	input data axis 0 in frequency^-2 format
		f		(cupy.ndarray):	target data axis 0 in frequency^-2 format
		idx		(cupy.ndarray):	indexes where each data point should be interpolated

	returns:
		rebin_data_cuda (cupy.ndarray): data after rebin in I(mu, t) format
	"""
	# Commonly there's no nan in input data
	# data = cp.nan_to_num(data)
	rebin_data_cuda = cp.zeros(data.shape,dtype=np.float32)
	data_size = nbin * t_len
	if data_size > 1024:
		num_thread = 1024
		grid_dim = math.ceil(data_size / num_thread)
	else:
		num_thread = data_size
		grid_dim = 1
	interp_kernel((grid_dim,), (num_thread,), (data, fy, f, idx, rebin_data_cuda, nbin, t_len))
	rebin_data_cuda = cp.nan_to_num(rebin_data_cuda)
	return rebin_data_cuda


if __name__ == '__main__':
	exit()
