import numpy as np
 
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
             
