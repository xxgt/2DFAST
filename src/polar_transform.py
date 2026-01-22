import scipy.signal as signal
#from scipy import ndimage
from cupyx.scipy import ndimage
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def polar_coordinates_convert(data, angle ):
	"""
	Transform the rectanular coordinates into polar coordinates
		Note ..  Take histogram2D method
	param:
	data:	Data want to transform 
	angle:	a list for [Angle_max, Angle_min]	  
	Return: 
	data in polar-coordinate I(r,theta)
	Angle resolution
	Radius Resolution
	"""

	ang_min = angle[0]
	ang_max = angle[1]
	rang    = data.shape
	ang_rsl = (1./rang[1])*180/np.pi
	rad_rsl =  1
	
	#take the left top of the matrix as center
	row     = np.arange(-rang[0]+1,1,dtype=np.float32) 
	rank    = np.arange(rang[1],dtype=np.float32)

	# calculate the angle of each pixe
	angle   = np.nan_to_num(np.arctan(row[:,None]/rank[None,:])) / np.pi * 180 

	# calculate radius of each pixel
	radius  = np.sqrt(row[:,None]**2 + rank[None,:]**2)
	ang     = -angle.reshape(-1)
	rad     = radius.reshape(-1)
	data    = data.reshape(-1)
	rad_bin = np.arange(1,rang[1],rad_rsl)
	ang_bin = np.arange(ang_min,ang_max,ang_rsl)
	data_r, _, __ = np.histogram2d(rad, ang, weights=np.real(data), bins= (rad_bin,ang_bin))
	data_i, _, __ = np.histogram2d(rad, ang, weights=np.imag(data), bins= (rad_bin,ang_bin))
	polar_data   = data_r + data_i * 1j

	return  polar_data,ang_rsl,rad_rsl

def polar_coordinates_convert_inter(data, angle,n_deg,L_fft=0, align='t'):
	"""
	Transform the rectanular coordinates into polar coordinates
	Note ..  Take interpolation method
	param:
		data:   Data want to transform 
		angle:  a list for [Angle_max, Angle_min]         
		n_deg:	number of angle axis length
	Return: 
		data in polar-coordinate I(r,theta) \n with shape [n_rad,n_deg]
	"""

	rang     = data.shape # due to 1stFFT, input here is in square
	ang_min  = angle[0]
	ang_max  = angle[1]
	# tan_min  = np.tan(ang_min*np.pi/180)
	# tan_max  = np.tan(ang_max*np.pi/180)
	
	n_deg	 = n_deg
	n_rad	 = max(data.shape)
	rad_grid = cp.linspace(1,n_rad-1,n_rad)
	tan_min  = cp.tan(ang_min*np.pi/180)
	tan_max  = cp.tan(ang_max*np.pi/180)
	tan_grid = cp.linspace(tan_min,tan_max,n_deg)
	ang_grid = cp.arctan(tan_grid)/np.pi*180
	
	grid_a,grid_r = cp.meshgrid(ang_grid,rad_grid)
	if align == 'f':
		x_p     = grid_r * cp.cos(grid_a*np.pi/180.) -1
		y_p     = rang[0] - grid_r * cp.sin(grid_a*np.pi/180.) 
	else:
		x_p     = grid_r * cp.sin(grid_a*np.pi/180.) -1
		y_p     = rang[0] - grid_r * cp.cos(grid_a*np.pi/180.) 
	x_p     = x_p.reshape(-1)
	y_p     = y_p.reshape(-1)
	cord    = cp.asarray([y_p,x_p])
	# print(cord)
	polar_matrix_r  = ndimage.map_coordinates(cp.real(data),cord,order=0)
	polar_matrix_i  = ndimage.map_coordinates(cp.imag(data),cord,order=0)
	polar_data    = polar_matrix_r+polar_matrix_i*1j
	polar_data    = polar_data.reshape(grid_r.shape)	
	return polar_data


if __name__ == '__main__':

	exit(0)

