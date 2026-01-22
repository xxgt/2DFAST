import sys,os,time
time_ps = time.time()
import numpy as np
import cupy as cp
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from optparse import OptionParser
#import mpi4py.MPI as MPI
from dir_create import dir_create
#from calibrated import calibration
from rebin import rebin_inter_cuda
from FFT import FFT
# print(time.time() - time_ps)
from polar_transform import polar_coordinates_convert_inter
# print(time.time() - time_ps)
from plot_all import plot
# print(time.time() - time_ps)
from DM_calculate import DM_calculate
# print(time.time() - time_ps)
from Signal_finding import Signal_finding
# print(time.time() - time_ps)
from readfile import read_data
from multi_thread_read import MultiThreadReader
# print(time.time() - time_ps)

import warnings
warnings.filterwarnings("ignore")

from matplotlib.ticker import MultipleLocator, FormatStrFormatter , NullFormatter,LogLocator, LogFormatter,AutoLocator

from mpl_toolkits.mplot3d import Axes3D

time_parse_s = time.time()
if __name__ == '__main__':
#	instance for invoking MPI relatedfunctions
#	comm = MPI.COMM_WORLD
#	the node rank in the whole community
#	comm_rank = comm.Get_rank()
	
	# the size of the whole community, i.e.,the total number of working nodes in the MPI cluster
#	comm_size = comm.Get_size()
	
	p = OptionParser()
	p.set_usage("python __main__.py  [options] ")
	p.set_description(__doc__)
	p.add_option('-f', '--file', dest='file', type='str',default='',
	help='Put filterbank file want to search')
	p.add_option('-t','--threshold', dest='threshold', type = 'float',default = 16,
	help='Threshold(sigma) for candidates pick')
	p.add_option('--dm', dest='dm', type = 'int',default = [50,2000],nargs=2,
	help='Set  DM range, Suggest use default [50,2000]')
	p.add_option('--pixel', dest='pixel', type = 'int',default = 5,
	help='Pixels number at 2nd 1DFFT map')
	p.add_option('--nthreads', dest='nthreads', type = 'int',default = 0,
	help='num of IO threads')
	p.add_option('--mask_cycle', dest='msk_cycle', type = 'int',default = 0,
	help='Mask abnormal lines at 1st 2DFFt map, only when RFI is terrible')
	p.add_option('--nsamps_gulp', dest='t_len', type = 'int',default = 0,
	help='Samples number for onece process, suggest use self-calculate(def)')
	p.add_option('--nbin', dest='nbin', type = 'int',default = 0,
	help='number of channels after re-bin step, suggest use self-calculate(def)')
	p.add_option('--wp', dest='Wp', type = 'int',default = 10,
	help='Set width of pulse to search, Suggest use default 10(ms).')
	p.add_option('--angle', dest='angle', type = 'float',default = [0,0],nargs=2,
	help='Angle range within [0,90], For de-buging.')

	p.add_option('-v', '--verbose', dest='verbose',action='store_true',
	help='Show details of process')
	p.add_option('-p', '--plot', dest='plot',action='store_true',
	help='Make overview plot for final result')

	p.add_option( '--Plot_proc', dest='plot_proc', type='str',default='',
	help='Input process step key words want to make signle plot.\
			\nKey words Including:\n\n{raw, rebin, 1stFFT, polar_sets_3D, \
			polar_sets_2D, 2ndFFT_3D, 2ndFFT_2D}\n (This function Remain updates)')
	
	if '-h' in sys.argv or '--help' in sys.argv:
		p.print_help()
		exit(-1)
	opts, args = p.parse_args(sys.argv[1:])

	if opts.file == '' or os.path.exists(opts.file)==False:
		print('Please Checkout file input!')
		p.print_help()
		exit()
#	comm.barrier()
	if opts.plot:
		plot_flag = 1
	else:
		plot_flag = 0
	if opts.verbose:
		SHOW = 1
	else:
		SHOW = 0
#	comm.barrier()


	############
	#Arguments #
	############
	f_name  = opts.file.split('/')[-1]
	file_in   = opts.file
	plot_dir  = '../graph/' + f_name[:-4] + '/'
	dir_create(plot_dir,f_name)
	print('directory build complete!')
#	comm.barrier()
	time_init_s = time.time()
	DM_range  = opts.dm        # Min and Max DM
	Wp        = opts.Wp        # Wp means pulse width in (ms)
	t_len     = opts.t_len     # Time samples for each process chunk.(if 0, decided by DM_max)
	nbin      = opts.nbin      # Number of beams for rebin.if 0,nbin = N_f_chanel      
	ang       = opts.angle     # Angle range for search. If [0,0], Angle got from DM range.
	msk_cycle = opts.msk_cycle # Number of channels to be zeros in 2D-FFT(Noise remove).
	pixel     = opts.pixel     # Number of pixel to sum in 2ndFFT3D SNR compute.
	threshold = opts.threshold
	plot_proc = opts.plot_proc
	nthreads  = opts.nthreads
	######################
	Candidates_l = []   
	Gtran_l = []
	t_consume = 0
	t_io = 0
	t_sigfind = 0
	t_norm = 0
	t_rebin_total = 0
	t_1stfft_total = 0
	t_polar_total = 0
	t_2ndfft_total = 0
	align = "t"
	######################
	#Read Data from file #
	######################


	print('Begin to load data from ' + file_in) 
	fil, num, p_n, freq, t_rsl, t_len, t_gulp, nbin ,nch,T, fy, fy_cuda, f_cuda, idx_cuda,\
		angle, n_deg, L_fft, DM_axis, DM_axis_cuda = read_data(file_in ,t_len, nbin,DM_range,Wp,ang)

	'''	fil		the reader	
		num		Total number of chunks to process 
		p_n		Number of chunks for each thread to process
		freq	list of frequencys, nch in total
		t_rsl	time step
		t_len	time length want to process	
		t_gulp	magnitude of real process time length in power of 2
		nbin	number of channel after rebin step
		nch		number of channel
		T		time scale per time unit
		fy		freq^-2
		angle	angle range [angle_max, angle_min]
		n_deg	Radius resolutiongrid size for interpolate in polar-coordin transform
		L-fft	length of signal after FFT
	'''
	f_axis = np.linspace(fy.min(),fy.max(),nbin)
	########################################
	#Create New comm if time length too short     
	#########################################
#	comm =comm.Split(p_n)
#	comm_rank = comm.Get_rank()
#	comm_size = comm.Get_size()
	if p_n == 0:
		exit(0)

	###################
	#Begin to search  #
	###################
	threadpool = []
	for i in range(nthreads):
		if i == p_n:
			break
		threadpool.append(MultiThreadReader(file_in, i*t_len, t_len))
		threadpool[i].start()
	time_s = time.time()
	t_prepare = time_s - time_ps
	print("Preparation Ready, Take:", t_prepare, "(s)")
	# data_1 = None
	data_1 = cp.zeros((int(t_len), nbin), dtype=np.float32)
	if nbin < 2 * L_fft:
		data_1 = data_1.transpose()
	for  i_ch in range(p_n):  #i_chunk 
		data_0 = data_1
		time_r_s = time.time()
		if nthreads == 0:
			data, _ = fil.read_block(t_len*i_ch,int(t_len))
			data_cuda = cp.array(data).astype(np.float32)
		else:
		#	data_cuda = readFileBlock(file_in, i_ch*t_len, t_len, nthreads)
			threadpool[0].join()
			data_cuda = threadpool[0].getResult()
			threadpool.pop(0)
			if i_ch + nthreads < p_n:
				threadpool.append(MultiThreadReader(file_in, (i_ch + nthreads)*t_len, t_len))
				threadpool[nthreads-1].start()
		seq    = i_ch
		# std_r  = data.std()	
		
		time_norm_s = time.time()
		time_load =  time_norm_s - time_r_s
		data_cuda = cp.nan_to_num(data_cuda)
		data_cuda = cp.array(data_cuda - data_cuda.mean())
		data_cuda = cp.nan_to_num(data_cuda)
		# t_ch_s = t_len*(i_ch-1)*t_rsl   #time of chunck start.
		# t_ch_e = t_len*(i_ch+1)*t_rsl 
		# t_axis = np.linspace(t_ch_s,t_ch_e,t_len) 
		time_norm = time.time() - time_norm_s
		if SHOW ==1:   
			print('\n\n============================================')
			print('Process chunk:(',int(i_ch),'/',int(p_n),')')
			print('Load Data Over, Sub-chunck shape:',data_cuda.shape, 'Take %f (s)'%time_load)
			print('Normalization Take %f (s)'%time_norm)
		t_io += time_load
		t_norm += time_norm
		time_de_s = time.time()
		# re_data, f_axis  =  rebin_inter(data, fy, nbin,t_axis)
		data_1 = rebin_inter_cuda(data_cuda, fy_cuda, f_cuda, idx_cuda, nbin, t_len)
		if nbin < 2 * L_fft:
			data_1 = data_1.transpose()
			align = 'f'
		time_re = time.time() - time_de_s
		t_rebin_total += time_re
		if SHOW ==1:    print('Rebin over.  Take %f (s)'%time_re)
		# if i_ch == 0:
		# 	t_consume += time_re
			
			# if SHOW==1:
			# 	print('Dedispersion Time Cost: ', t_consume ,'seconds')
			# 	print('IO Time Cost: ', t_io, 'seconds')
			# 	print('Equal:',(t_consume+t_io)/60.,'minutes.')
			# continue
		
		# time_cp_s = time.time()
		# re_data_cuda = cp.array(re_data)
		# time_cp  = time.time() - time_cp_s

		time_FFT_s = time.time()
		if nbin < 2 * L_fft:
			re_data_cuda = cp.concatenate((data_0, data_1), axis=1)
		else:
			re_data_cuda = cp.concatenate((data_0, data_1), axis=0)
			
		FFT1st_data_cuda = FFT(re_data_cuda, 2, L_fft, msk_cycle, t_gulp)
		time_FFT  = time.time() - time_FFT_s
		t_1stfft_total += time_FFT		
		if SHOW ==1 :    print('1st FFT over. Take %f (s)'%time_FFT)

		time_polar_s = time.time()
		polar_data_cuda  = polar_coordinates_convert_inter(FFT1st_data_cuda, angle, n_deg, L_fft, align)
		time_polar   = time.time() - time_polar_s
		t_polar_total += time_polar
		if SHOW ==1:    print('Polar transform over,Takes %f (s)'%time_polar,'\nPolar data shape:',polar_data_cuda.shape)
		time_FFT2nd_s = time.time()
		FFT2nd_data_cuda = FFT(polar_data_cuda, 1 )# 1 means 1 Dimension FFT
		# FFT2nd_data = cp.asnumpy(FFT2nd_data_cuda)
		time_FFT2nd = time.time()-time_FFT2nd_s
		t_2ndfft_total += time_FFT2nd
		time_de_e = time.time()
		t_block = time_de_e - time_r_s
		t_consume +=  time_de_e - time_de_s
		if SHOW ==1:
			print('2nd FFT over.Take %f (s)'%time_FFT2nd)
			print('Data Block Time Cost:', t_block ,'seconds')
			print('Accumulated Dedispersion Time Cost:', t_consume ,'seconds')
			print('Accumulated IO Time Cost:', t_io, 'seconds')
			print('Accumulated Normalization Time Cost:', t_norm, 'seconds')
			print('Equal:',(t_consume+t_io)/60.,'minutes.')
			print('Time stamp from ',(seq-1)*t_len,' to ',(seq+1)*t_len)
#		    print ':',p_n*t_len*comm_size,'samples'
			print('Angle search range:[%.2f,%.2f]'%(angle[0],angle[1]))
			print('============================================\n\n')		

			print('Begin to locate the signal and calculate Significance...')
		time_fi_s = time.time()
		# candidate, G_t,dump_flag  = Signal_finding(DM_axis,threshold,FFT2nd_data, pixel, DM_range, seq)
		candidate, G_t,dump_flag  = Signal_finding(DM_axis,DM_axis_cuda,threshold,FFT2nd_data_cuda, pixel, DM_range, seq)
		
		time_fi = time.time() - time_fi_s
		t_sigfind += time_fi
		if SHOW == 1:
			print('Signal Search Time Cost:', time_fi, 'seconds')
		time_dump_s = time.time()
		if dump_flag == 1:
			print('Dumping data into disk...(%d)'%seq)
			np.save('../data/'+f_name[:-4]+'/raw_'+str(seq),data)
			# cp.save('../data/'+f_name[:-4]+'/rebin_'+str(seq),re_data_cuda)
			# cp.save('../data/'+f_name[:-4]+'/polar_'+str(seq),polar_data_cuda)
			# cp.save('../data/'+f_name[:-4]+'/FFT1st_'+str(seq),FFT1st_data_cuda)
			# np.save('../data/'+f_name[:-4]+'/FFT2nd_'+str(seq),FFT2nd_data)
			cp.save('../data/'+f_name[:-4]+'/FFT2nd_'+str(seq),FFT2nd_data_cuda)
		t_io += time.time() - time_dump_s
		Candidates_l.extend(candidate)
		Gtran_l.extend(G_t)

		if plot_proc != '':
			print('\nBegin to plot...\n')
			t_ch_s = t_len*(i_ch-1)*t_rsl   #time of chunck start.
			t_ch_e = t_len*(i_ch+1)*t_rsl 
			t_axis = np.linspace(t_ch_s,t_ch_e,2*t_len)
			if nbin < 2 * L_fft:
				re_data = re_data_cuda
			else:
				re_data = re_data_cuda.T
			plot(t_axis,data_cuda,re_data,polar_data_cuda,FFT1st_data_cuda,FFT2nd_data_cuda,plot_proc,freq,f_axis,n_deg,plot_dir,pixel,angle,i_ch,p_n,SNR=0,DM=0,A_f=0)
			if SHOW ==1 and opts.plot_proc: print('Plot Over...\n\n')	
	#########################################
	# gather the results from all processes #
	#########################################
#	comm.barrier()
	time_e= time.time()
	print("candidate length",len(Candidates_l))
	
	if SHOW ==1 :
		print('\n****************************************')	
		
		print('Total consume:',time_e-time_ps,'seconds')
		print('Preparation Time Cost:', t_prepare, 'seconds')
		print("\tImport Time:", time_parse_s - time_ps, "(s)")
		print("\tParse Args Time:", time_init_s - time_parse_s)
		print("\tInitialization Time:", time_s - time_init_s)
		print('IO Time Cost:', t_io, 'seconds')
		print('Normalization Time Cost:', t_norm, 'seconds')
		print('Dedispersion Time Cost:', t_consume ,'seconds')
		print('Signal Finding Time Cost:', t_sigfind, 'seconds')
		print('****************************************\n')
#	combine_candidates  = [Candidates_l,]
#	combine_Gtransient  = [Gtran_l,]
	 
	c_Candidates = []
	c_Gtransient = []
	if len(Candidates_l) > 0:
		for ii in range(len(Candidates_l)):
			c_Candidates.extend(Candidates_l[ii])

		if len(Gtran_l) > 0:
			for ii in range(len(Gtran_l)):
				c_Gtransient.extend([Gtran_l[ii]])


	c_Candidates = np.array(c_Candidates).reshape(-1,4).T
	c_Gtransient = np.array(c_Gtransient).reshape(-1,3).T

	if len(c_Gtransient[0]) != 0:
		G_snr	= c_Gtransient[2,:]
		lo      = np.where(G_snr == G_snr.max())
		signifi = c_Gtransient[2,lo[0][0]]
		M_seq   = int(c_Gtransient[1,lo[0][0]])
		dm_G    = c_Gtransient[0,lo[0][0]]
		print(" ,{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14}".format(signifi, dm_G, time_e-time_ps, t_prepare, \
			time_parse_s - time_ps, time_init_s - time_parse_s, time_s - time_init_s, t_io, t_norm, t_consume,
			t_rebin_total, t_1stfft_total, t_polar_total, t_2ndfft_total, t_sigfind))
	else:
		print(" ,{0},0,{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}".format(0, time_e-time_ps, t_prepare, \
			time_parse_s - time_ps, time_init_s - time_parse_s, time_s - time_init_s, t_io, t_norm, t_consume,
			t_rebin_total, t_1stfft_total, t_polar_total, t_2ndfft_total, t_sigfind))
	if len(c_Candidates[0,:]) == 0:
		print('Found no transient signal above threshold :(')
		exit(0)

	if  plot_proc!='':
		print('\n\n****************************')
		print('*multiprocess plot over....*')
		print('****************************\n\n')
	if plot_flag==1:
		print('Making Result...')
	else:	
		exit(0)
	###########################
	#Begin to make result Plot#
	###########################
	fig = plt.figure(figsize=(14,8))

	font = {'family' : 'serif',  
			'color'  : 'black',  
			'weight' : 'normal',  
			'size'   : 16,  
			}
	
	font1 = {'family' : 'sans-serif',
			'color'  : 'black',
			'weight' : 'normal',
			'size'   : 10,
			}		

	cm = plt.cm.get_cmap('Greens')
	if len(c_Gtransient[0]) != 0:
		G_snr	= c_Gtransient[2,:]
		lo      = np.where(G_snr == G_snr.max())
		M_seq   = int(c_Gtransient[1,lo[0][0]])
		dm_G    = c_Gtransient[0,lo[0][0]]
		lo_dm	= np.where(DM_axis == dm_G)
		G_data	= np.load('../data/'+f_name[:-4]+'/FFT2nd_'+str(M_seq)+'.npy')
		G_data  = abs(G_data)
		lo_max	= np.where(G_data[:,lo_dm[0]]==G_data[:,lo_dm[0]].max())
		y_axis	= np.linspace(-G_data.shape[0]/2,G_data.shape[0]/2,G_data.shape[0])
		ax1 = fig.add_subplot(2,2,1)
		A	= ax1.pcolormesh(DM_axis,y_axis, G_data,cmap=cm)		
		plt.colorbar(A)
		lo_max = y_axis[lo_max[0]]
		dm_G_p = np.zeros(len(lo_max))+dm_G
		#################################
		#             Plot 1            #
		#################################

		ax1.scatter(dm_G_p,lo_max,marker='o',s=80,edgecolors='r',c="none")
		ax1.set_xlabel('Result DM',fontdict=font1)
		ax1.set_title('2nd FFT data of Tsamps:'+'\n('+str((M_seq-1)*t_len)+'~'+str((M_seq+1)*t_len)+')',fontdict=font)
		ax1.set_ylim(-G_data.shape[0]/2,G_data.shape[0]/2)
		ax1.set_xlim(DM_axis[0]-1,DM_axis[-1]+1)
	else:
		print("Not Found any Giant Transient:(")

	dm_c	= c_Candidates[0,:]
	seq_c	= c_Candidates[1,:]
	snr_c	= c_Candidates[2,:]
	n_count = c_Candidates[3,:]

	dm_bin	= DM_axis
	
	snr_bin	= np.linspace(0,snr_c.max(),20)
	Bins	= (snr_bin, dm_bin)

	d,x,y	= np.histogram2d(snr_c,dm_c,weights=n_count,bins = Bins)

	dm_bin	= np.linspace(dm_bin.min(),dm_bin.max(),d.shape[1])
	snr_bin	= np.linspace(snr_bin.min(),snr_bin.max(),d.shape[0])
	dm_bin ,snr_bin = np.meshgrid(dm_bin,snr_bin)

	#################################
	#             Plot 2            #
	#################################

#       ax2 = fig.add_subplot(2,2,2,projection='3d',azim=-79,elev=30, zlim=(0,n_count.max()))
	ax2 = fig.add_subplot(2,2,2,projection='3d',azim=-67,elev=30)		
	surface = ax2.plot_surface(dm_bin, snr_bin, d, linewidth=0, rstride=1, cstride=1, cmap=plt.cm.jet)
#       surface = ax2.plot_surface(snr_bin, dm_bin, d, linewidth=0, rstride=1, cstride=1, cmap=plt.cm.jet)		
#		plt.colorbar(surface)#,shrink=1)
	
	plt.xlim(DM_axis[0],DM_axis[-1])
	plt.ylim(0,snr_c.max()+1)

	xmajorLocator = MultipleLocator(400)
	xminorLocator = MultipleLocator(100)
	xmajorFormatter = FormatStrFormatter('%1d')

	ax2.xaxis.set_major_locator(xmajorLocator)
	ax2.xaxis.set_major_formatter(xmajorFormatter)

	ax2.xaxis.set_minor_locator(xminorLocator)

	ax2.set_xlabel('DM Value',fontdict=font1)
	ax2.set_ylabel('Significance',fontdict=font1)
	ax2.set_zlabel('Count Number',fontdict=font1)
	ax2.set_title('Distribution of Candidates',fontdict=font)
	ax2.xaxis.grid(True, which='minor')
	#################################
	#             Plot 3			#
	#################################
	ax3	= fig.add_subplot(2,1,2)
	
	cm = plt.cm.get_cmap('Reds')

	snr	= snr_c/snr_c.max()*120
	C 	= ax3.scatter(dm_c,seq_c,c=snr_c,alpha=0.5,edgecolors=['b'],marker='*',s=snr,label='Candidates')
	if len(c_Gtransient[0]) !=0:
		dm_G    = c_Gtransient[0,:]
		seq_G   = c_Gtransient[1,:]
		snr_G   = c_Gtransient[2,:]
		snr     = snr_G/snr_G.max()*200
		G = ax3.scatter(dm_G,seq_G,c=snr_G,alpha=0.5,edgecolors=['r'],s=snr,marker='o',cmap=cm,label='Giant Transient')
			
		for i in range(len(dm_G)):
			cord = [dm_G[i],seq_G[i]]
			text = 'Significance:%1.1f'%(snr_G[i])
			plt.annotate(text, xy = cord, xytext = cord)
		
	plt.colorbar(C)
	ax3.set_xlim(DM_axis[0],DM_axis[-1])
#		ax3.set_xlim(10,DM_axis[-1])
	new_ticks = np.linspace(0,num,5,dtype=np.int)
	ticks_label =[]
	for i in new_ticks:
		label = '('+str(int(i*t_len))+'~'+str(int((i+1)*t_len))+')'
		ticks_label.append(label)
	
	xmajorLocator = MultipleLocator(400)
#		xmajorLocator = LogLocator(10)
	xmajorFormatter = FormatStrFormatter('%1.f')
	xminorLocator = MultipleLocator(100)
#		xminorLocator = LogLocator(1)

	ax3.xaxis.set_major_locator(xmajorLocator)
	ax3.xaxis.set_minor_locator(xminorLocator)
	ax3.xaxis.set_major_formatter(xmajorFormatter)		

	ymajorLocator = MultipleLocator(1)
	ax3.yaxis.set_major_locator(ymajorLocator)
	
#		ax3.semilogx()
	ax3.set_yticks(new_ticks)
	ax3.set_yticklabels(ticks_label)
	ax3.xaxis.grid(True,which = 'major')
	ax3.yaxis.grid(True,which = 'major')
	plt.legend(loc='best',scatterpoints=1,markerscale=0.6,framealpha=0.5,fontsize='xx-small')
	ax3.set_xlabel('DM',fontdict=font1)
	ax3.set_ylabel('Tsamps',fontdict=font1)
	
	plt.tight_layout()
	plt.savefig(plot_dir+'overview')
	plt.show()
	exit(0)
