import pickle
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from dateutil.parser import parse
from scipy.interpolate import interp1d
from matplotlib.dates import date2num, num2date
import scipy.interpolate
from scipy import fft
import math
from scipy import signal
from scipy.optimize import curve_fit
from scipy import integrate
import os
from scipy.integrate import quad
import sys
import time


def lengthfield(Lshell):
    theta0 = 1./( np.arccos(np.sqrt(1.0/Lshell)) )
    
    theta2 = ( np.arccos(np.sqrt(2.5/Lshell)) )
    theta1 = - theta2
    def inte(thetaa):
         return Lshell*(1+3*np.sin(thetaa)**2)**0.5
    Int = quad( inte,(theta1),(theta2)) 
    return Int[0]


def eigen(harmonic,Lshell,num1):


	#nums = {}
	#for Lshell in arange(2.5,12,0.125):
		#nums[Lshell] = pb.Bats2d(sort(os.listdir('./datafilesz'))[1250])['rho'][80][int(Lshell*8)] * 1000000
	
	#num = nums[Lshell]
	#xt,yt = geo_stat(Lshell,angles,dire)
	
	num = num1 * 1000000
	theta2 = ( np.arccos(np.sqrt(2.5/Lshell)) )
	theta1 = - theta2
	Int = quad(integrand,(theta1),(theta2))
	
	theta0 = 1./( np.arccos(np.sqrt(1.0/Lshell)) )
	term = (1./(1+3*np.sin(theta0)**2))**0.5
	
	period = (1.9*10**(-5)) *  np.sqrt(num) * (Lshell**4) * (Int[0])# + 4*(1.9*10**(-5))*(28.0*1000000)**0.5*(2.5**4-1)*term

	return harmonic*(1./period)


def eigens(harmonic,Lshell,time1,angles,dire):


	#nums = {}
	#for Lshell in arange(2.5,12,0.125):
		#nums[Lshell] = pb.Bats2d(sort(os.listdir('./datafilesz'))[1250])['rho'][80][int(Lshell*8)] * 1000000
	
	#num = nums[Lshell]
	xt,yt = geo_stat(Lshell,angles,dire)
	
	num = pb.Bats2d(sort(os.listdir('./datafilesz'))[time1])['rho'][yt[time1]][xt[time1]] * 1000000
	theta2 = 1./( np.cos(np.sqrt(2.5/Lshell)) )
	theta1 = - theta2
	Int = quad(integrand,(theta1),(theta2))
	
	theta0 = 1./( np.cos(np.sqrt(1.0/Lshell)) )
	term = (1./(1+3*np.sin(theta0)**2))**0.5
	
	period = (1.9*10**(-5)) *  np.sqrt(num) * (Lshell**4) * (Int[0])# + 4*(1.9*10**(-5))*(28.0*1000000)**0.5*(2.5**4-1)*term

	return harmonic*(1./period)


def toroid(harmonic,Lshell):
	m=1.41
	alfven = 1000000.
	Re = 6371000
	muh0 = np.cos(1./( np.cos(np.sqrt(2.5/Lshell)) ))
	theta2 = 1./( np.cos(np.sqrt(2.5/Lshell)) )
	v = np.sin(theta2)**2/(Lshell*Re)
	M0= ((v**(m-8))/alfven**2) * Re**(m-3)

	A = M0*(1-muh0**2)**(6-m)
	B = -2*muh0*(6-m)*(1-muh0**2)**(5-m)
	freq = alfven*v*(3./4*(2*harmonic+1)*np.pi-np.pi/4)**(2./3)* ( B**(1./3)* ( A/B + muh0 - (1-Re*v)**(1./2) ))
	
	return freq



def bandpass(dic,freqL,freqH,sample):

    rate = 1./(2*sample)
    band = []
    bL = (freqL/rate)    #split into smaller bands and compare/
    bH = (freqH/rate)
    b,a = scipy.signal.butter(8,bH,btype='lowpass')
    d,c = scipy.signal.butter(8,bL,btype='highpass')

    band = scipy.signal.filtfilt(d,c,(scipy.signal.filtfilt(b,a,dic)))

    return band



def fpsd(data,sample):
	
	samples = 1./sample
	f = fft(data)[1:int(len(data)/2) + 1]
	p = (1./(len(data))) * np.abs(f)**2
	p = 2*p
	freq = np.arange(0,samples/2,samples/len(data))
	
	return freq,p
	

def phase(data1,data2):
  
	d1 = np.array(fft(data1))
	d2 = np.array(fft(data2)).conj()
	
	cros = d1*d2/(len(d1)**2)
	
	cros = 2*cros[1:len(d1)/2+1]
	cros[1] = cros[1]/2
	
	a = np.arctan( cros.imag/cros.real )
	
	return a,cros
      
def coh(data1,data2):
	
	c = (abs( phase(data1,data2)[1] ) **2) / ( fpsd(data1)[1]*fpsd(data2)[1])
	
	return c


def dyn_phase(x,y, fs, framesz, hop):   # fs=0.5,framesz=512,hop=4 works
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hamming(framesamp)
    X = scipy.array([ phase(x[i:i+framesamp],y[i:i+framesamp])[0]
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X

def stft(x, fs, framesz, hop):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
                     for i in range(2, len(x)-framesamp, hopsamp)])
    return X
  
  

def convert(mod,angle):
	angle = np.radians(angle)
	
	for rads in mod.keys():
		for vals in mod[5].keys():
			mod[rads][vals] = np.array(mod[rads][vals])
	
	for rads in mod.keys():
		mod[rads]['br'] = mod[rads]['bx']*np.cos(angle)   + mod[rads]['by']*np.sin(angle)
		mod[rads]['btheta'] = -mod[rads]['bx']*np.sin(angle) + mod[rads]['by']*np.cos(angle) 
		mod[rads]['er'] = mod[rads]['ex']*np.cos(angle)  + mod[rads]['ey']*np.sin(angle)
		mod[rads]['etheta'] = -mod[rads]['ex']*np.sin(angle) + mod[rads]['ey']*np.cos(angle) 
		#mod[rads]['sr'] = mod[rads]['sx']*cos(angle)  + mod[rads]['sy']*sin(angle)
		#mod[rads]['stheta'] = -mod[rads]['sx']*sin(angle) + mod[rads]['sy']*cos(angle) 
		
	return mod


def convertangle(mod):
		
	for rads in mod.keys():
		for vals in mod[-180].keys():
			mod[rads][vals] = array(mod[rads][vals])
	
	for rads in mod.keys():
		mod[rads]['br'] = mod[rads]['bx']*cos(rads)   + mod[rads]['by']*sin(rads)
		mod[rads]['btheta'] = -mod[rads]['bx']*sin(rads) + mod[rads]['by']*cos(rads) 
		mod[rads]['er'] = mod[rads]['ex']*cos(rads)  + mod[rads]['ey']*sin(rads)
		mod[rads]['etheta'] = -mod[rads]['ex']*sin(rads) + mod[rads]['ey']*cos(rads) 
		#mod[rads]['sr'] = mod[rads]['sx']*cos(angle)  + mod[rads]['sy']*sin(angle)
		#mod[rads]['stheta'] = -mod[rads]['sx']*sin(angle) + mod[rads]['sy']*cos(angle) 
		
	return mod


      
def smooth(x,beta):
	""" kaiser window smoothing """
	window_len=11
	# extending the data at beginning and at the end
	# to apply the window at the borders
	s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
	w = np.kaiser(window_len,beta)
	y = np.convolve(w/w.sum(),s,mode='valid')
	return y[5:len(y)-5]
      
      
def dist_mean(func):
	
	def gauss(x, A, mu, sigma):
		return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
	
	distr = plt.hist([func[n] for n in range(len(func))], bins=range(int(round(max(func)))))
	
	return curve_fit(gauss,distr[1][1:],distr[0])

def dcorr(s,keys):
	
	_size = len(data_calcs[s][keys])
	
	pairsdatadist=[]
	pairsdatacol=[]
	pairsmoddist=[]
	pairsdatarow=[]
	
	pairmeandata_cols,pairmeandata_rows=[],[]
	pairmeanmod_cols,pairmeanmod_rows=[],[]
	
	pairmeandata_row,pairmeanmod_row,pairmeanmod_col,pairmeandata_col=np.zeros([_size]),np.zeros([_size]),np.zeros([_size]),np.zeros([_size])
	
	AA,BB,XX,YY = [], [], [], []
	
	AA,BB = meshgrid(data_calcs[s][keys],data_calcs[s][keys])
	XX,YY = meshgrid(mod_calcs[s][keys],mod_calcs[s][keys])
	
	pairsdatadist = np.sqrt( (AA-BB)**2 )
	pairsmoddist= np.sqrt( (XX-YY)**2 )
	
	pairmeandata = np.mean( pairsdatadist )
	pairmeanmod = np.mean( pairsmoddist )
	
	for n in xrange(_size):
		
		pairmeandata_row[n] = (np.mean(pairsdatadist[n]))
		pairmeanmod_row[n]=(np.mean(pairsmoddist[n]))
		pairmeanmod_col[n]=(np.mean(pairsmoddist[:][n]))
		pairmeandata_col[n]=(np.mean(pairsdatadist[:][n]))
		
	pairmeandata_cols,pairmeandata_rows = meshgrid(pairmeandata_row,pairmeandata_row)
	pairmeanmod_cols,pairmeanmod_rows = meshgrid(pairmeanmod_row,pairmeanmod_row)
	
	covdata = pairsdatadist - pairmeandata_rows - pairmeandata_cols + pairmeandata
	covmod = pairsmoddist - pairmeanmod_rows - pairmeanmod_cols + pairmeanmod
	
	dcov = ( np.sum (covdata*covmod) ) / (_size)**2
	dvarx = ( np.sum (covdata*covdata) ) / (_size)**2
	dvary = ( np.sum (covmod*covmod) ) / (_size)**2
	
	dcor = np.sqrt ( dcov/ (np.sqrt(dvarx*dvary)))
	
	return dcor

def dcorrs(s,t,keys,func):
	apples = min(mod_calcs[s][keys].size,mod_calcs[t][keys].size)
	oranges = min(data_calcs[s][keys].size,data_calcs[t][keys].size)
	if func == 'mod_calcs':
		_val = apples
	else:
		_val = oranges
		
	_size = len(data_calcs[s][keys][:_val])
	
	pairsdatadist=[]
	pairsdatacol=[]
	pairsmoddist=[]
	pairsdatarow=[]
	
	pairmeandata_cols,pairmeandata_rows=[],[]
	pairmeanmod_cols,pairmeanmod_rows=[],[]
	
	pairmeandata_row,pairmeanmod_row,pairmeanmod_col,pairmeandata_col=np.zeros([_size]),np.zeros([_size]),np.zeros([_size]),np.zeros([_size])
	
	AA,BB,XX,YY = [], [], [], []
	
	AA,BB = meshgrid(func[s][keys][:_val],func[s][keys][:_val])
	XX,YY = meshgrid(func[t][keys][:_val],func[t][keys][:_val])
	
	pairsdatadist = np.sqrt( (AA-BB)**2 )
	pairsmoddist  = np.sqrt( (XX-YY)**2 )
	
	pairmeandata = np.mean( pairsdatadist )
	pairmeanmod  = np.mean( pairsmoddist )
	
	pairmeandata_row = np.array([(np.mean(pairsdatadist[n])) for n in xrange(_size)])
	pairmeanmod_row  = np.array([(np.mean(pairsmoddist[n])) for n in xrange(_size)])
	pairmeanmod_col  = np.array([(np.mean(pairsmoddist[:][n])) for n in xrange(_size)])
	pairmeandata_col = np.array([(np.mean(pairsdatadist[:][n])) for n in xrange(_size)])
		
	pairmeandata_cols,pairmeandata_rows = meshgrid(pairmeandata_row,pairmeandata_row)
	pairmeanmod_cols,pairmeanmod_rows   = meshgrid(pairmeanmod_row,pairmeanmod_row)
	
	covdata = pairsdatadist - pairmeandata_rows - pairmeandata_cols + pairmeandata
	covmod  = pairsmoddist - pairmeanmod_rows - pairmeanmod_cols + pairmeanmod
	
	dcov    = ( np.sum (covdata*covmod) ) / (_size)**2
	dvarx   = ( np.sum (covdata*covdata) ) / (_size)**2
	dvary   = ( np.sum (covmod*covmod) ) / (_size)**2
	
	dcor    = np.sqrt ( dcov/ (np.sqrt(dvarx*dvary)))
	
	return dcor


def dcorrs2(s,t):

		
	_size = len(s)
	
	pairsdatadist=[]
	pairsdatacol=[]
	pairsmoddist=[]
	pairsdatarow=[]
	
	pairmeandata_cols,pairmeandata_rows=[],[]
	pairmeanmod_cols,pairmeanmod_rows=[],[]
	
	pairmeandata_row,pairmeanmod_row,pairmeanmod_col,pairmeandata_col=np.zeros([_size]),np.zeros([_size]),np.zeros([_size]),np.zeros([_size])
	
	AA,BB,XX,YY = [], [], [], []
	
	AA,BB = meshgrid(s,s)
	XX,YY = meshgrid(t,t)
	
	pairsdatadist = np.sqrt( (AA-BB)**2 )
	pairsmoddist  = np.sqrt( (XX-YY)**2 )
	
	pairmeandata = np.mean( pairsdatadist )
	pairmeanmod  = np.mean( pairsmoddist )
	
	pairmeandata_row = np.array([(np.mean(pairsdatadist[n])) for n in xrange(_size)])
	pairmeanmod_row  = np.array([(np.mean(pairsmoddist[n])) for n in xrange(_size)])
	pairmeanmod_col  = np.array([(np.mean(pairsmoddist[:][n])) for n in xrange(_size)])
	pairmeandata_col = np.array([(np.mean(pairsdatadist[:][n])) for n in xrange(_size)])
		
	pairmeandata_cols,pairmeandata_rows = meshgrid(pairmeandata_row,pairmeandata_row)
	pairmeanmod_cols,pairmeanmod_rows   = meshgrid(pairmeanmod_row,pairmeanmod_row)
	
	covdata = pairsdatadist - pairmeandata_rows - pairmeandata_cols + pairmeandata
	covmod  = pairsmoddist - pairmeanmod_rows - pairmeanmod_cols + pairmeanmod
	
	dcov    = ( np.sum (covdata*covmod) ) / (_size)**2
	dvarx   = ( np.sum (covdata*covdata) ) / (_size)**2
	dvary   = ( np.sum (covmod*covmod) ) / (_size)**2
	
	dcor    = np.sqrt ( dcov/ (np.sqrt(dvarx*dvary)))
	
	return dcor

def energy_map(dire,time0,time1):
  
	import multiprocessing as mp
 
  
	_val = ['e_ez']

	sat = {}
	con = {}
	
	size_x =  read_data('u0050.dat')['rho'].shape[0]
	size_y =  read_data('u0050.dat')['rho'].shape[1]
	

	for v in _val:
		con[v]={}
		for f in arange(0.0,0.01,0.001):
			#con[v][f] = np.zeros([size_x,size_y])
					
	#for v in _val:
		#sat[v] = {}
		#for xt in xrange(size_x):
			#sat[v][xt] = {}
			#for yt in xrange(size_y):
				#sat[v][xt][yt] = []
	 
	#sat['e_bz'] = {}
	#sat['e_bx'] = {}
	#sat['e_by'] = {}
	#con['e_bz'] = {}
	#con['e_by'] = {}
	#con['e_bx'] = {}
	#for xt in xrange(size_x):
		#sat['e_bz'][xt] = {}
		#sat['e_by'][xt] = {}
		#sat['e_bx'][xt] = {}
		#for yt in xrange(size_y):
			#sat['e_bz'][xt][yt] = []
			#sat['e_by'][xt][yt] = []
			#sat['e_bx'][xt][yt] = []
	#for f in arange(0.0,0.006,0.001):
		#con['e_bz'][f] = np.zeros([size_x,size_y])
		#con['e_bx'][f] = np.zeros([size_x,size_y])
		#con['e_by'][f] = np.zeros([size_x,size_y])
	 
	#mu = 1.26*10**-6
	#eps = 8.85 * 10**-12
  
	
	#def get_keys(container):
	
		#for n in xrange(len(container)):
			#container[n]['ex'] = (-container[n]['uy']*container[n]['bz'] + container[n]['uz']*container[n]['by'])*10**-3
			#container[n]['ey'] = (container[n]['ux']*container[n]['bz'] - container[n]['uz']*container[n]['bx'])*10**-3
			#container[n]['ez'] = (-container[n]['ux']*container[n]['by'] + container[n]['uy']*container[n]['bx'])*10**-3
			#container[n]['e_ex'] = container[n]['ex']
			#container[n]['e_ey'] = container[n]['ey']
			#container[n]['e_ez'] = container[n]['ez']
			#container[n]['e_bx'] = container[n]['bx']
			#container[n]['e_by'] = container[n]['by']
			#container[n]['e_bz'] = container[n]['bz']
			#container[n]['s_x'] = container[n]['ey'] * container[n]['bz'] - container[n]['ez'] * container[n]['by']
			#container[n]['s_y'] = -( container[n]['ex'] * container[n]['bz'] - container[n]['ez'] * container[n]['bx'] )
			#container[n]['s_z'] = container[n]['ex'] * container[n]['by'] - container[n]['ey'] * container[n]['bx']
		#return container
	      
	#containers = None
	#ww = 24
	#for n in xrange(time0,time1,ww):
		#del containers
		#pool = mp.Pool(processes=4)
		#containers = pool.map(read_data,sort(os.listdir('.'))[n:n+ww])
		#containers = get_keys(containers)
		#pool.close()
		#for p in range(0,ww,1):
            #for xt in xrange(size_x):
                #for yt in xrange(size_y):
                    #sat['e_ez'][yt][xt].append( containers[p]['ez'][yt][xt] )
                    #sat['e_bz'][yt][xt].append( containers[p]['bz'][yt][xt] )
                    #sat['e_by'][yt][xt].append( containers[p]['by'][yt][xt] )
                    #sat['e_bx'][yt][xt].append( containers[p]['bx'][yt][xt] )
                 
    #bobb = containers[0]['ez'].shape[0]
    #bobbb = containers[0]['ez'].shape[1]
    #del containers
	#rates = 10
	#freq = array(fpsd(sat['e_ez'][1][1],rates)[0]).size
	#for values in ['e_ez','e_bz','e_bx','e_by']:
		#for p in arange(0.0,0.006,0.001):
			#for xt in xrange(bobb):
				#for yt in xrange(bobbb):
					#con[values][p][xt][yt]    = 0.001*sum( fpsd( array(sat[values][xt][yt]),rates  )[1][1000*p*freq/50:(1000*p+2)*freq/50]  )
					##con[values]['medlow'][xt][yt] = sum( fpsd( array(sat[values][xt][yt]),rates  )[1][int(freq/4): int(freq/2)]   )
					##con[values]['medhigh'][xt][yt]= sum( fpsd( array(sat[values][xt][yt]),rates  )[1][int(freq/2): int(3*freq/4)]   )
					##con[values]['high'][xt][yt]   = sum( fpsd( array(sat[values][xt][yt]),rates  )[1][int(3*freq/4): int(freq)]   )


	#del sat
	
	#return con


def calc_dens(lat1,lat2,freq):
    
        avgg = radians(lat1/2.0+lat2/2.0)
        rad = 1./( np.cos( avgg)**2 )
        inte= np.sin( (np.arccos(np.sqrt(1.0/rad)))) - np.sin( -(np.arccos(np.sqrt(1.0/rad))))
        dens= (1./(1.9*10**-5*freq*(rad**4)*inte)**2)/1000000
        
        return dens
    
def read_pickle(filess):
	return pickle.load(open(filess,'rb'),encoding='iso-8859-1')	


class ReadData:
    
    def __init__(self,files,dtype):
        self.files = files
        self.dtype = dtype
        self.container = {}
        #for val in self.dtype:
            #self.container[val] = []
    
    def state_variables(self):
        if self.dtype =='aniso_all':
            values = ['rho','mx','my','mz','bx','by','bz','pe','ppar','p']
        elif self.dtype == 'ideal_ful':
            values = ['rho','ux','uy','uz','bx','by','bz','p','bx1','by1','by2','bz1','g','jx','jy','jz']
        elif self.dtype == 'aniso_ful':
            values = ['rho','ux','uy','uz','bx','by','bz','pe','ppar','p','pperp','b1x','b1y','b1z','jx','jy','jz']
        elif self.dtype == 'field_lines':
            values == [] #['x','y','z','rho','ux','uy','uz','energy','bx','by','bz','p','jx','jy','jz','eta']
        elif self.dtype == 'ideal_mhd':
            values = ['rho','ux','uy','uz','bx','by','bz','p','jx','jy','jz']
        return values
      
      
    def read2d(self):
    
        values = self.state_variables()
    
        data1 = np.genfromtxt(self.files,delimiter=',')
        datascale = [ int(data1.shape[0])/len(values), int(data1.shape[1]) - 1 ]
        
        for num,val in enumerate(values):
            self.container[val] = clean_datas(data1[num*datascale[0]:datascale[0]*(num+1)].T[:datascale[1]].T)
            
        if 'ux' and 'uy' and 'uz' in self.dtype:
            self.container['ex'] = (-self.container['uy']*self.container['bz'] + self.container['uz']*self.container['by'])*10**-3
            self.container['ey'] = (self.container['ux']*self.container['bz'] - self.container['uz']*self.container['bx'])*10**-3
            self.container['ez'] = (-self.container['ux']*self.container['by'] + self.container['uy']*self.container['bx'])*10**-3
        return self.container
 
 
 
class ProcessData:  

    def __init__(self,dire,dtype,inputs):
     
        self.dtype = dtype
        self.inputs = inputs
        self.container = {}
        self.dire = dire
        self.sat = {}
        if dtype == 'field_lines':
            values = ['x','y','z','rho','ux','uy','uz','energy','bx','by','bz','p','jx','jy','jz','eta']
        else:
            values = ReadData(self.dire + os.listdir(self.dire)[1],self.dtype).state_variables()
        for val in values:
            self.container[val] = []


    def mplocation(self):
        
        import os
        
        radius = np.arange(2.5,30,0.125)
        files = os.listdir(self.dire)[1000]
        
        xt = dict([key,[]] for key in radius)
        yt = dict([key,[]] for key in radius)
        sat = dict([key,[]] for key in self.inputs[0])
        
        
        containers = ReadData(self.dire+ files, self.dtype ).read2d()
        centerx,centery = 280,200
        #for x in xrange(int(containers.shape[0])):
        
        for ang in self.inputs[0]:
            if ang < -120:
                radius = np.arange(2.5,30,0.125)
            elif ang < -90 and ang > -120:
                radius = np.arange(2.5,25,0.125)
            elif ang > -90:
                radius = np.arange(2.5,15,0.125)
            for rad in radius:
                xt[rad],yt[rad] = geo_stat(rad,math.radians(ang),self.dire)
                sat[ang].append(  containers['uy'].T[xt[rad][5]][yt[rad][5]]  )

        mp_data,coord,xx,yy = [],[],[],[]
        
        mp_data = [ [( np.gradient(smooth(np.array(sat[ang]),8)) ).argmin(), ang]  for ang in self.inputs[0]  ]
        coord = [ [ centerx+8*(mp_data[alpha][0]/8.0+2.5)*np.cos( np.radians(mp_data[alpha][1]) ), centery+8*(mp_data[alpha][0]/8.0+2.5)*np.sin( np.radians(mp_data[alpha][1])) ] for alpha in xrange(len(mp_data))]
    
        xx = [int(coord[ll][0]) for ll in xrange(len(coord))]    
        yy = [int(coord[ll][1]) for ll in xrange(len(coord))]

        return zip(xx,yy),mp_data

   
    def magnetopause(self):
        ## inputs for this function are [0],[1],[2],[3] where [0] is the list of angles along magnetopause, [1] is 'time0' or the first data file, [2] is the last data file, and [3] is the number of files to pool in an async process band
        
        import numpy as np
        import os
        from multiprocessing import Pool
        
        locxy = self.mplocation()
        values = ReadData(os.listdir(self.dire)[1],self.dtype).state_variables()

        if __name__ == "__main__":
            from mc import make_applicable, make_mappable
            
            locxy = self.mplocation()
            values = ReadData(self.dire+os.listdir(self.dire)[1],self.dtype).state_variables()
    
            for points,(j,k) in enumerate(reversed(locxy)):
                self.sat[points] = {}
                for val in values+['ex','ey','ez']:
                    self.sat[points][val] = []
            
        
            def mp_data(n):
                
                dire ='./flowline0/'
                import numpy as np
                import os
                container = {}
                values = ['x','y','z','rho','ux','uy','uz','energy','bx','by','bz','p','jx','jy','jz','eta']

                data = np.genfromtxt(dire+'/'+np.sort(os.listdir(dire))[n],delimiter=',')
                datascale = [ int(data.shape[0])/len(values), int(data.shape[1]) - 1 ]

                for num,val in enumerate(values):
                    container[val] = (data[num*datascale[0]:datascale[0]*(num+1)].T[:datascale[1]].T)
                if 'ux' and 'uy' and 'uz' in values:
                    container['ex'] = (-container['uy']*container['bz'] + container['uz']*container['by'])*10**-3
                    container['ey'] = (container['ux']*container['bz'] - container['uz']*container['bx'])*10**-3
                    container['ez'] = (-container['ux']*container['by'] + container['uy']*container['bx'])*10**-3
                return container
            
            kk = None
            for lengths in xrange(self.inputs[1],self.inputs[2],self.inputs[3]):
                del kk
                pool    = Pool(processes=6)
                results = [pool.apply_async(*make_applicable(mp_data,x)) for x in xrange(lengths,lengths+self.inputs[3])]
                kk = [result.get(timeout=100000) for result in results]
                pool.close()
                for p in range(0,self.inputs[3],1):
                    for i,(j,k) in enumerate(reversed(locxy)):
                        for values in kk[p].keys():
                            self.sat[i][values].append( kk[p][values][k][j] )

        return self.sat


    def stat_line(self):
        
        values = ['x','y','z','rho','ux','uy','uz','energy','bx','by','bz','p','jx','jy','jz','eta']

        self.satstat = {}
        for val in values:
            self.satstat[val] = []
            
        for j,n in enumerate(np.sort(os.listdir(self.dire))):
            containers = read_data_line(n)
            for val in values:
                self.satstat[val].append( containers[val] )
        return self.satstat
        
    def stat_largeangles(self):

        self.satstat = {}
        xt,yt = {},{}
        
        radius = np.arange(2.5,self.inputs[5],0.125)
    
        for rad in radius:
            xt[rad],yt[rad] = [],[]

        values = ReadData(self.dire+ os.listdir(self.dire)[100],self.dtype).state_variables()
        for rad in radius:
            self.satstat[rad] = {}
            for value in values:
                self.satstat[rad][value] = []
                        
        containers = None

        def get_keys(container):

            container['ex'] = (-container['uy']*container['bz'] + container['uz']*container['by'])*10**-3
            container['ey'] = (container['ux']*container['bz'] - container['uz']*container['bx'])*10**-3
            container['ez'] = (-container['ux']*container['by'] + container['uy']*container['bx'])*10**-3

            return container
        
        for j,n in enumerate(np.sort(os.listdir(self.dire))):
            containers = ReadData(self.dire + n,self.dtype).read2d()
            if 'ux' and 'uy' and 'uz' in values:
                containers = get_keys(containers)
            for rad in radius:
                xt[rad],yt[rad] = geo_stat(rad,math.radians(self.inputs[4]),self.dire)
                for values in values:
                    self.satstat[rad][values].append( containers[values].T[xt[rad][5]][yt[rad][5]] )

        return self.satstat
                
    def stat_largeangles_angles(self):

        self.satangles = {}
        xt,yt = {},{}
        values = ReadData(self.dire+os.listdir(self.dire)[100],self.dtype).state_variables()

        angle = np.arange(radians(self.inputs[7]),radians(self.inputs[6]),radians(1))
    
        for ang in angle:
            xt[ang],yt[ang] = [],[]

        for ang in angle:
            self.satangles[int(degrees(ang))] = {}
            for value in values:
                self.satangles[int(degrees(ang))][value] = []
            
            
        containers = None

        def get_keys(container):

            container['ex'] = (-container['uy']*container['bz'] + container['uz']*container['by'])*10**-3
            container['ey'] = (container['ux']*container['bz'] - container['uz']*container['bx'])*10**-3
            container['ez'] = (-container['ux']*container['by'] + container['uy']*container['bx'])*10**-3

            return container


        for j,n in enumerate(sort(os.listdir(self.dire))):
            containers = ReadData(self.dire+os.listdir(self.dire)[j],self.dtype).read2d()
            if 'ux' and 'uy' and 'uz' in values:
                containers = get_keys(containers)
            for ang in angle:
                xt[ang],yt[ang] = geo_stat(self.inputs[5],ang,self.dire)
                for val in values:
                    self.satangles[int(degrees(ang))][values].append( containers[val].T[xt[ang][5]][yt[ang][5]] )

        return self.satangles
             

def read_pickle(filess):
	return pickle.load(open(filess,'rb'))	

class spectral:
    def __init__(self,data,freq1,freq2,step):
        self.freq = fpsd(data,step)[0]
        self.psd = fpsd(data,step)[1]
        self.band = bandpass(detrend(data),freq1,freq2,step)

        

def radial_psd(quant,data,sample,beta,start,end):
  
	r = []
	
	length = len(data[6][quant][start:end])
	
	radrange = np.arange(3,max(data.keys())+1,1)
	
	for i in radrange:
		#r.append ( smooth(log(abs(fft(data[i][quant][1200:2200])[:length/2]**2)),beta))
		r.append ( smooth (np.log( fpsd(data[i][quant][start:end],10)[1][5:]) ,beta))
		#r.append ( smooth (( fpsd(data[i][quant][start:end],10)[1]) ,beta))

	
	
	freq = fpsd(data[5][quant][1:length/2-1],sample)[0]
	
	locx = np.arange(0,len(radrange),len(radrange)/18.0)
	lblx = np.arange(3,max(data.keys()),0.5)
	
	locy = np.arange(0,len(r[0]),len(r[0])/20.0)
	lbly = np.arange(0,1000.0*1.0/(2*sample)+(1.0/(2*sample))/20.0,1000.0*(1.0/(2*sample))/20.0)
	
	plt.pcolor( (np.array(r)).T )
	plt.xticks(locx,lblx)
	plt.yticks(locy,lbly)
	
	return np.array(r)


def va(mod,radius):
	
	alf = (10**-9)/(np.sqrt(1000000.0))*(np.array(mod[radius]['bz'])/np.sqrt((4*np.pi*10**-7)*(1.67*10**-27)*np.array(mod[radius]['rho'])))/1000
	
	return alf

def sound(data,rad,quant):
    
    if quant == 'pperp':
        cs = np.sqrt(1.4*np.array(anisopress(data,rad))*1e-9/(np.array(data[rad]['rho'])*1.6*10**-27*1e6))
    elif quant != 'pperp':
        cs = np.sqrt(1.4*np.array(data[rad][quant])*1e-9/(np.array(data[rad]['rho'])*1.6*10**-27*1e6))
    
    return cs/1000.0

def beta(data,rad,quant):
    
    #if quant == 'pperp':
    _beta = sound( data, rad,quant)/va(data,rad)
    
    return _beta

def anisopress(data,rad):

    pperp = 0.5*(3*np.array(data[rad]['p'])-np.array(data[rad]['ppar']))

    return pperp

def findzeros(data,rad,quant,start):
    zero=[]
    zero1=[]
    for i in xrange(len(smooth(np.diff(data[rad][quant]),2)[start:])):
        if smooth(np.diff(data[rad][quant]),2)[start:][i] <0.001 and smooth(np.diff(data[rad][quant]),2)[start:][i]>-0.001:
            if data[rad][quant][start:][i] > 0:
                zero.append(i+start)
                zero1.append(data[rad][quant][start:][i])
    return zero,zero1



def evoldens_2ax(data):
    
    freq = 1000*np.array(fpsd(data[20]['rho'],10)[0])

    ymaxrho,ymaxbz = [],[]
    for i in [20,40,60,80,100]:
        ymaxrho.append(max( fpsd(data[i]['rho'][360:],10)[1][8:70] ))
        ymaxbz.append(max( fpsd(data[i]['bz'][360:],10)[1][8:70] ))
    
    ymaxr = max( ymaxrho )
    ymaxb = max( ymaxbz )
    plt.figure(figsize=(12,6))
    for i,j in enumerate([[20,1040],[40,920],[60,800],[80,640],[100,520]]):
        plt.subplot(1,5,i+1)
        plt.rc_context({'axes.edgecolor':'black', 'xtick.color':'black', 'ytick.color':'red', 'figure.facecolor':'white'})    
        plt.title(str(j[1])+' ' +'LT')
        plt.xticks(rotation='vertical')
        plt.plot(freq[8:72],fpsd(data[j[0]]['rho'][360:],10)[1][8:72])
        plt.xlim(freq[8],freq[72])
        plt.ylim(0,ymaxr)
        if i+1==1:
            plt.ylabel('PSD $n$ [$(Mp/cc)^2/Hz$]',color='blue')
        if i+1==3:
            plt.xlabel('Frequency [mHz]')
        plt.twinx()
        plt.rc_context({'axes.edgecolor':'black', 'xtick.color':'black', 'ytick.color':'blue', 'figure.facecolor':'white'})    
        plt.plot(freq[8:72],fpsd(scipy.signal.detrend(data[j[0]]['bz'][360:]),10)[1][8:72],color='red')
        plt.xlim(freq[8],freq[72])
        plt.ylim(0,ymaxb)
        if i+1==5:
            plt.ylabel('PSD $B_{1 z}$ [$nT^2/Hz$]',color='red',rotation=-90,labelpad=20)
    plt.subplots_adjust(wspace=0.5)   
    #plt.tight_layout()
    
plt.rc('font', family='serif')    
    
    
def evoldens_3ax(data):

    freq = 1000*np.array(fpsd(data[20]['rho'],10)[0])
    _a,_b = 8,48
  
    ymaxrho,ymaxbz,ymaxbx = None,None,None
    fig,ax = plt.subplots(1,5,figsize=(12,6))
    ymaxrho,ymaxbz,ymaxbx = [],[],[]
    for i in [20,40,60,80,100]:
        ymaxrho.append(max( fpsd(data[i]['rho'][360:],10)[1][_a:_b] ))
        ymaxbz.append(max( fpsd(data[i]['bz'][360:],10)[1][_a:_b] ))
        ymaxbx.append ( max( fpsd(data[i]['bx'][360:],10)[1][_a:_b] ))
    ymaxr = max( ymaxrho )
    ymaxb = max( ymaxbz )
    ymaxbb = max(ymaxbx)
    for k,j in enumerate([[20,1040],[40,920],[60,800],[80,640],[100,520]]):
      
        #fig,ax = plt.subplots(1,k)
        axes = [ax[k],ax[k].twinx(),ax[k].twinx()]
        fig.subplots_adjust(right=1)
        if j[0] == 20:
            axes[0].set_ylabel('PSD $n$ [$(Mp/cc)^2/Hz$]',color='red',fontsize=15) 
        if j[0] == 60:
            axes[0].set_xlabel('Frequency [mHz]',fontsize=15)

        for ax1,val,clr,lim  in zip(axes, ['bz','rho','bx'],['blue','red','green'],[ymaxb,ymaxr,ymaxbb]):
            #ax1.spines['right'].set_color(clr)
            #ax1.spines['left'].set_color('blue')
            #ax1.yaxis.label.set_color(clr)
            
            if j[0] != 100:
                axes[-1].set_frame_on(False)
                axes[-1].patch.set_visible(False)
                plt.setp( axes[-1].get_yticklabels(),visible=False)
                plt.setp( ax1.get_xticklabels(),rotation = 90)
                axes[-1].yaxis.set_ticks_position('none')
                ax1.plot(freq[_a:_b],fpsd(data[j[0]][val][360:],10)[1][_a:_b],color=clr)
                ax1.set_xlim(freq[_a],freq[_b])
                ax1.set_ylim(0,lim)
                plt.title(str(j[1])+' ' +'LT',fontsize=15)


            elif j[0] == 100:
                axes[-1].spines['right'].set_position(('axes',1.45))
                axes[-1].set_frame_on(True)
                axes[-1].patch.set_visible(False)
                ax1.plot(freq[_a:_b],fpsd(data[j[0]][val][360:],10)[1][_a:_b],color=clr)
                ax1.set_xlim(freq[_a],freq[_b])
                ax1.set_ylim(0,lim)
                plt.ylabel('PSD $B_{1 y}$ [$nT^2/Hz$]',color='green',rotation=-90,labelpad=20,fontsize=15)
                axes[-2].set_ylabel('PSD $B_{1 z}$ [$nT^2/Hz$]',color='blue',rotation=-90,labelpad=20,fontsize=15)
                plt.title(str(j[1])+' ' +'LT')
                plt.setp( ax1.get_xticklabels(),rotation = 90)

    axes[1].set_xlabel('X-axis')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)   
    plt.show()


def sfft(data,sample):
    hops = []
    for i in xrange(len(data)-501):
        hops.append( smooth(np.log(fpsd(data[i:i+500],10)[1][5:]),8 ))
        
    return np.array(hops).T

def mp_shape(angle,subsolar):
    
    rad = subsolar*(2./(1+np.cos((angle))))**0.5
    
    return rad

def mp_arclength(subsolar,angles):
    
    from scipy.integrate import quad
    
    def integrand(x,a):
        arg = -0.5*a*((2./(1+np.cos(x)))**(0.5-1)) * np.sin(x)/(1+np.cos(x))**2
        return np.sqrt(mp_shape(x,a)**2+arg**2)
    
    arc = quad(integrand,np.radians(angles[0]),np.radians(angles[1]),args=(subsolar))
    
    return np.abs(arc[0])
   
def evoldens_3ax(data,names):

    freq = 1000*np.array(fpsd(data[20]['rho'],10)[0])
    _a,_b = 8,48
  
    ymaxrho,ymaxbz,ymaxbx = None,None,None
    fig,ax = plt.subplots(1,5,figsize=(12,6))
    ymaxrho,ymaxbz,ymaxbx = [],[],[]
    for i in [20,40,60,80,100]:
        ymaxrho.append(max( fpsd(data[i]['rho'][360:],10)[1][_a:_b] ))
        ymaxbz.append(max( fpsd(data[i]['bz'][360:],10)[1][_a:_b] ))
        ymaxbx.append ( max( fpsd(data[i]['bx'][360:],10)[1][_a:_b] ))
    ymaxr = max( ymaxrho )
    ymaxb = max( ymaxbz )
    ymaxbb = max(ymaxbx)
    for k,j in enumerate([[20,1040],[40,920],[60,800],[80,640],[100,520]]):
      
        #fig,ax = plt.subplots(1,k)
        axes = [ax[k],ax[k].twinx(),ax[k].twinx()]
        fig.subplots_adjust(right=1)
        if j[0] == 20:
            axes[0].set_ylabel('PSD $n$ [$(Mp/cc)^2/Hz$]',color='red',fontsize=15) 
        if j[0] == 60:
            axes[0].set_xlabel('Frequency [mHz]',fontsize=15)
            axes[0].set_title(names,y=1.06,fontsize=15)
            
        for ax1,val,clr,lim  in zip(axes, ['bz','rho','bx'],['blue','red','green'],[ymaxb,ymaxr,ymaxbb]):
            #ax1.spines['right'].set_color(clr)
            #ax1.spines['left'].set_color('blue')
            #ax1.yaxis.label.set_color(clr)
            
            if j[0] != 100:
                axes[-1].set_frame_on(False)
                axes[-1].patch.set_visible(False)
                plt.setp( axes[-1].get_yticklabels(),visible=False)
                plt.setp( ax1.get_xticklabels(),rotation = 90)
                axes[-1].yaxis.set_ticks_position('none')
                ax1.plot(freq[_a:_b],fpsd(data[j[0]][val][360:],10)[1][_a:_b],color=clr)
                ax1.set_xlim(freq[_a],freq[_b])
                ax1.set_ylim(0,lim)
                plt.title(str(j[1])+' ' +'LT',fontsize=15)


            elif j[0] == 100:
                axes[-1].spines['right'].set_position(('axes',1.45))
                axes[-1].set_frame_on(True)
                axes[-1].patch.set_visible(False)
                ax1.plot(freq[_a:_b],fpsd(data[j[0]][val][360:],10)[1][_a:_b],color=clr)
                ax1.set_xlim(freq[_a],freq[_b])
                ax1.set_ylim(0,lim)
                plt.ylabel('PSD $B_{1 y}$ [$nT^2/Hz$]',color='green',rotation=-90,labelpad=20,fontsize=15)
                axes[-2].set_ylabel('PSD $B_{1 z}$ [$nT^2/Hz$]',color='blue',rotation=-90,labelpad=20,fontsize=15)
                plt.title(str(j[1])+' ' +'LT')
                plt.setp( ax1.get_xticklabels(),rotation = 90)

    axes[1].set_xlabel('X-axis')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)   
