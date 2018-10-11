
import numpy as np

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
            
            self.container['ex'] = (-self.container['uy']*self.container['bz'] + self.container['uz']*self.container['by'])*10**-3
            self.container['ey'] = (self.container['ux']*self.container['bz'] - self.container['uz']*self.container['bx'])*10**-3
            self.container['ez'] = (-self.container['ux']*self.container['by'] + self.container['uy']*self.container['bx'])*10**-3
        return self.container
 
 
