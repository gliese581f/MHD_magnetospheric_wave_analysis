import matplotlib.pyplot as plt
import numpy as np


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
