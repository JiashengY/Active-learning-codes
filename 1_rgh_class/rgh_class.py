# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 12:13:02 2023

@author: Jiasheng
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import pandas as pd
import time
from tkinter import filedialog
from tensorflow.keras.models import load_model
import multiprocessing as mp
plt.rcParams['text.usetex'] = True
from scipy.interpolate import RegularGridInterpolator
class rgh():
    def __init__(self,x,z,y):
        if (y.shape[0]!=len(z)) | (y.shape[1]!=len(x)):
            print("Spanwise pixels of the roughness patch:" + str(len(z)))
            print("Streamwise pixels of the roughness patch:" + str(len(x)))
            print("Resolution of the roughness patch:" + str(y.shape))
            raise ValueError("Incompatible roughness map size to the given coordinates")
        if y.std()<0.001:
            print("The detected wall fluctuation is too small, please check height scaling before use.")
        x=x.reshape((-1))
        z=z.reshape((-1))
        ##### Interpolating surface to desired resolution
        xnew=np.linspace(0,np.max(x),int(np.max(x)*500))
        znew=np.linspace(0,np.max(z),int(np.max(z)*500))
        [xnewM,znewM]=np.meshgrid(xnew,znew)
        f=RegularGridInterpolator((x,z),y.T)
        ynew=f((xnewM,znewM))
        ##### Calculating desired quantities
        del x,z,y
        self.y=ynew
        self.x=xnew
        self.z=znew
        self.Lx=np.max(xnew)
        self.Lz=np.max(znew)
        self.Nx=len(xnew)
        self.Nz=len(znew)
        self.dx=self.Lx/self.Nx
        self.dz=self.Lz/self.Nz
        self.kt=np.max(ynew)
        self.sk=sps.skew(ynew,axis=None)
        self.ku=sps.kurtosis(ynew,fisher=False,axis=None)
        self.krms=np.std(ynew)
        self.kmd=np.mean(ynew)
        self.ra=np.mean(np.abs(ynew-np.mean(ynew)))
        self.por=1-np.mean(ynew)/np.max(ynew)
        self.ESx=np.mean(np.abs((np.roll(ynew,1,axis=0)-ynew)/self.dx))
        self.ESz=np.mean(np.abs((np.roll(ynew,1,axis=1)-ynew)/self.dz))
        self.incx=np.arctan(sps.skew((np.roll(ynew,1,axis=0)-ynew)/self.dx,axis=None)/2)
        self.incz=np.arctan(sps.skew((np.roll(ynew,1,axis=1)-ynew)/self.dz,axis=None)/2)
        self.PS=np.fft.fft2(ynew-np.mean(ynew))## Cautions: this is not PS

###### Calculating k99
        surface=self.y-np.min(self.y)
        
        self.n99,self.bin99=np.histogram(surface.reshape((-1)),density=True,bins=100)
        self.bin99=(self.bin99[1:]+self.bin99[:-1])/2
        C_I=0.01
        flag=0
        for i in range(len(self.n99)):
            s=np.sum(self.n99[:i])/np.sum(self.n99)
            if (s>C_I/2) & (flag==0):
                LBound=self.bin99[i]
                flag=1
            if (s>1-C_I/2) & (flag==1):
                UBound=self.bin99[i]
                break
        
        
        self.k99=UBound-LBound

    def show_surface(self,representation="2D"): ## Plotting surface
        if representation =="2D":
            plt.imshow(self.y,extent=[self.x.min(),self.x.max(),self.z.min(),self.z.max()])
            plt.colorbar()
        elif (representation !="3D") & ((representation !="3d")):
            print("Only 3D or 2D are the potions, setting to default 2D")
            plt.imshow(self.y,extent=[self.x.min(),self.x.max(),self.z.min(),self.z.max()])
            plt.colorbar()
        else:
            plt.figure()
            ax = plt.axes(projection='3d')
            XM,ZM=np.meshgrid(self.x,self.z)
            ax.plot_surface(XM,ZM,self.y,linewidth=0,antialiased=False,cmap="viridis")
            ax.set_xlim3d([self.x.min(),self.x.max()])
            ax.set_ylim3d([self.z.min(),self.z.max()])
            ax.set_zlim3d([0,4*(self.y.max()-self.y.min())])
            ax.set_box_aspect([self.x.max(), self.z.max(), 1.0])

    def print_stat(self): ## Outputing roughness statistical parameters as a table
        return pd.DataFrame({"Length":[self.Lx],"Width":[self.Lz],"Sk":[self.sk],"Ku":self.ku,"k_RMS":self.krms,"k_md":self.kmd,
                "kt":[self.kt],"k99":[self.k99],"Ra":[self.ra],"Por":[self.por],"ES_x":[self.ESx],
                "ES_z":[self.ESz],"Inc_x":[self.incx],"Inc_z":[self.incz]}).style.hide(axis='index')
    

    def plot_PDF(self,n_bins=10,Normalization=True): ## Plotting roughness PDF with desired number of bins
        plt.hist(np.reshape(self.y,(-1)),bins=n_bins,density=Normalization)


    def plot_PS(self,Normalization=True,azimuthal_average=False,moving_average=False,n_iter=3): ## Plotting roughness PS with desired setting
        if (azimuthal_average==False) & (moving_average==True): ## Always do azimuthal average before moving average
            print("moving average only applicable for azimuthal_average=True, thus azimuthal_average is automatically set to True")
            azimuthal_average=True

        # create "wave number" vectors, the unit is grid size
        qx = np.linspace(-self.Nx/2, self.Nx/2-1, self.Nx)
        qz = np.linspace(-self.Nz/2, self.Nz/2-1, self.Nz)
        qx = qx * 2 * np.pi/self.Nx
        qz = qz * 2 * np.pi/self.Nz
        qxM, qzM = np.meshgrid(qx, qz)
        ## Calculate the map of wavenumbers with same resolution as the surface, the origin is centered
        q_radius = np.sqrt(qxM**2 + qzM**2)

        # compute absolute value of power spectral density (imaginary part is not zero)
        if Normalization==True:
            self.PS[0,0]=0
            PS_toplot = abs(self.PS)**2/(np.sum(abs(self.PS)**2)/(self.Nx*self.Nz))
        else:
            PS_toplot =abs(self.PS)**2
        ## Do fftshift so that the PS map and wavenumber map are correspondent
        PS_toplot=np.fft.fftshift(PS_toplot)

        if azimuthal_average:
            q_radius=np.reshape(q_radius,(-1))
            PS_toplot=np.reshape(PS_toplot,(-1))
            #sorting PS and q based on q
            ind_q=np.argsort(q_radius)
            PS_sorted=PS_toplot[ind_q]
            q_sorted=q_radius[ind_q]
            q_before=q_sorted[0]
            q_averaged=np.zeros(len(np.unique(q_sorted)))
            PS_averaged=np.zeros(len(np.unique(q_sorted)))
            ind=0
            ind_q=0
            counter=0
            for i in q_sorted:
                if i==q_before:
                    #sum PS with same q
                    PS_averaged[ind]=PS_averaged[ind]+PS_sorted[ind_q]
                    counter=counter+1
                else:
                    #divided by the number of same q
                    PS_averaged[ind]=PS_averaged[ind]/counter
                    q_averaged[ind]=q_before
                    counter=1
                    ind=ind+1
                    q_before=i
                    #start to sum PS with next same q
                    PS_averaged[ind]=PS_averaged[ind]+PS_sorted[ind_q]
                if ind_q==len(q_sorted):# ending
                    PS_averaged[ind]=PS_averaged[ind]/counter
                    q_averaged[ind]=q_before
                ind_q=ind_q+1
            if moving_average:
                q_radius_1d=q_averaged
                PS_1d=PS_averaged
                PS_1d_averaged=np.zeros(len(q_radius_1d))
                for iter in range(n_iter):
                    for i in range(len(q_radius_1d)):
                        if i==0:
                            PS_1d_averaged[i]=PS_1d[0]
                        elif i==1:
                            PS_1d_averaged[i]=np.mean([PS_1d[0],PS_1d[1]])
                        elif i==len(q_radius_1d)-2:
                            PS_1d_averaged[i]=np.mean([PS_1d[i],PS_1d[i+1]])
                        elif i==len(q_radius_1d)-1:
                            PS_1d_averaged[i]=np.mean([PS_1d[i]])
                        else:
                            PS_1d_averaged[i]=np.mean([PS_1d[i+2],PS_1d[i+1],PS_1d[i-2],PS_1d[i-1],PS_1d[i]])
                    PS_1d=PS_1d_averaged
                PS_1d_pooled=[]
                Q_1d_bak=q_radius_1d
                pool=0
                Q_1d_pooled=[]
                for i in range(len(q_radius_1d)):
                    if i==0:
                        Q_1d_pooled.append(Q_1d_bak[i])
                        PS_1d_pooled.append(max([PS_1d[0]]))
                        continue
                    elif pool==3:
                        Q_1d_pooled.append(Q_1d_bak[i-1])
                        PS_1d_pooled.append(max([PS_1d[i],PS_1d[i-1],PS_1d[i-2]]))
                        pool=0
                        continue
                    elif i==len(q_radius_1d)-1:
                        Q_1d_pooled.append(Q_1d_bak[i-1])
                        PS_1d_pooled.append(max([PS_1d[i],PS_1d[i-1],PS_1d[i-2]]))
                    pool=pool+1
                q_radius_1d=np.array(Q_1d_pooled)
                PS_toplot_1d=np.array(PS_1d_pooled)
            else:
                q_radius_1d=q_averaged
                PS_toplot_1d=PS_averaged
        else:
            ind = np.argsort(q_radius.reshape(-1))    
            q_radius_1d = q_radius.reshape(-1)[ind]        
            PS_toplot_1d = PS_toplot.reshape(-1)[ind]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(q_radius_1d, PS_toplot_1d,c='k',marker="*")
        #ax.scatter(q_sorted, PS_sorted,c='k',marker="*")
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylabel("PS")
        ax.set_xlabel(r"$2\pi \Delta/\lambda$")
        
    def FFT_filter(self,lmax,lmin):
        qx = np.linspace(-self.Nx/2, self.Nx/2-1, self.Nx)
        qz = np.linspace(-self.Nz/2, self.Nz/2-1, self.Nz)
        qx = qx * 2 * np.pi/self.Nx
        qz = qz * 2 * np.pi/self.Nz
        qxM, qzM = np.meshgrid(qx, qz)
        q_radius = np.sqrt(qxM**2 + qzM**2)
        lmax = lmax * 500
        lmin = lmin * 500
        q0 = 2 * np.pi/lmax 
        q1 = 2 * np.pi/lmin
        PS_new = np.fft.fftshift(self.PS)
        # filter out PS contents out side of the desired range
        PS_new[q_radius>q1] = 0
        PS_new[q_radius<q0] = 0
        PS_new=np.fft.fftshift(PS_new)
        # return a new rgh object
        return rgh(self.x,self.z,np.real(np.fft.ifft2(PS_new))-np.min(np.real(np.fft.ifft2(PS_new))))
        


    def get_model_input(self,lmax=2,lmin=0.04,azimuthal_average=False,moving_average=False,n_iter=3,do_plots=False):
        surface=self.y-np.min(self.y)
        # lambda values for output, normalized by k99
        lambda_0=lmax/self.k99
        lambda_1=lmin/self.k99


        # convert to same unit 
        lmax = lmax * 500
        lmin = lmin * 500
        q0 = 2 * np.pi/lmax 
        q1 = 2 * np.pi/lmin
        
        # This part similar to plot_PS()
        if (azimuthal_average==False) & moving_average:
            print("moving average only applicable for azimuthal_average=True, thus azimuthal_average is automatically set to True")
            azimuthal_average=True
        # create "wave number" vectors
        qx = np.linspace(-self.Nx/2, self.Nx/2-1, self.Nx)
        qz = np.linspace(-self.Nz/2, self.Nz/2-1, self.Nz)
        qx = qx * 2 * np.pi/self.Nx
        qz = qz * 2 * np.pi/self.Nz
        qxM, qzM = np.meshgrid(qx, qz)
        q_radius = np.sqrt(qxM**2 + qzM**2)
          
        PS_toplot = abs(self.PS)/np.sqrt((np.sum(abs(self.PS)**2)/(self.Nx*self.Nz)))
        PS_toplot=np.fft.fftshift(PS_toplot)
        if azimuthal_average:
            q_radius=np.reshape(q_radius,(-1))
            PS_toplot=np.reshape(PS_toplot,(-1))
            ind_q=np.argsort(q_radius)
            PS_sorted=PS_toplot[ind_q]
            q_sorted=q_radius[ind_q]
            q_before=q_sorted[0]
            q_averaged=np.zeros(len(np.unique(q_sorted)))
            PS_averaged=np.zeros(len(np.unique(q_sorted)))
            ind=0
            ind_q=0
            counter=0
            for i in q_sorted:
                if i==q_before:
                    PS_averaged[ind]=PS_averaged[ind]+PS_sorted[ind_q]
                    counter=counter+1
                else:
                    PS_averaged[ind]=PS_averaged[ind]/counter
                    q_averaged[ind]=q_before
                    counter=1
                    ind=ind+1
                    q_before=i
                    PS_averaged[ind]=PS_averaged[ind]+PS_sorted[ind_q]
                if ind_q==len(q_sorted):
                    PS_averaged[ind]=PS_averaged[ind]/counter
                    q_averaged[ind]=q_before
                ind_q=ind_q+1
            if moving_average:
                q_radius_1d=q_averaged
                PS_1d=PS_averaged
                PS_1d_averaged=np.zeros(len(q_radius_1d))
                for iter in range(n_iter):
                    for i in range(len(q_radius_1d)):
                        if i==0:
                            PS_1d_averaged[i]=PS_1d[0]
                        elif i==1:
                            PS_1d_averaged[i]=np.mean([PS_1d[0],PS_1d[1]])
                        elif i==len(q_radius_1d)-2:
                            PS_1d_averaged[i]=np.mean([PS_1d[i],PS_1d[i+1]])
                        elif i==len(q_radius_1d)-1:
                            PS_1d_averaged[i]=np.mean([PS_1d[i]])
                        else:
                            PS_1d_averaged[i]=np.mean([PS_1d[i+2],PS_1d[i+1],PS_1d[i-2],PS_1d[i-1],PS_1d[i]])
                    PS_1d=PS_1d_averaged
                PS_1d_pooled=[]
                Q_1d_bak=q_radius_1d
                pool=0
                Q_1d_pooled=[]
                for i in range(len(q_radius_1d)):
                    if i==0:
                        Q_1d_pooled.append(Q_1d_bak[i])
                        PS_1d_pooled.append(max([PS_1d[0]]))
                        continue
                    elif pool==3:
                        Q_1d_pooled.append(Q_1d_bak[i-1])
                        PS_1d_pooled.append(max([PS_1d[i-1],PS_1d[i],PS_1d[i-2]]))
                        pool=0
                        continue
                    elif i==len(q_radius_1d)-1:
                        Q_1d_pooled.append(Q_1d_bak[i-1])
                        PS_1d_pooled.append(max([PS_1d[i-1],PS_1d[i],PS_1d[i-2]]))
                    pool=pool+1
                q_radius_1d=np.array(Q_1d_pooled)
                PS_toplot_1d=np.array(PS_1d_pooled)
            else:
                q_radius_1d=q_averaged
                PS_toplot_1d=PS_averaged
        else:
            ind = np.argsort(q_radius.reshape(-1))    
            q_radius_1d = q_radius.reshape(-1)[ind]        
            PS_toplot_1d = PS_toplot.reshape(-1)[ind]
        ## Selecting 30 points from processed PS curve
        ## equidistant in log scaling
        Qquerys=10**np.linspace(np.log10(q0),np.log10(q1),30)
        PSquerys=np.zeros(len(Qquerys))
        PSquerys[0]=PS_toplot_1d[q_radius_1d>Qquerys[0]][0]
        if do_plots==True:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(q_radius_1d, PS_toplot_1d,c='k',marker="*")
            ax.set_yscale('log')
            ax.set_xscale('log')
        # Interpolating PS values
        for i in range(1,len(Qquerys)):
            PSright=PS_toplot_1d[q_radius_1d>Qquerys[i]][0]
            PSleft=PS_toplot_1d[q_radius_1d<Qquerys[i]][-1]
            Qright=q_radius_1d[q_radius_1d>Qquerys[i]][0]
            Qleft=q_radius_1d[q_radius_1d<Qquerys[i]][-1]
            PSquerys[i]=PSleft+(PSright-PSleft)*(Qquerys[i]-Qleft)/(Qright-Qleft)
        ## Plotting the used PS values 
        if do_plots==True:
            ax.scatter(Qquerys, PSquerys,c='r',marker="o")
        PSquerys=np.log10(PSquerys)

        surface_kt=surface/np.max(surface)
        n0,bin0 = np.histogram(surface_kt.reshape((1,-1)), density=True,bins=30)
        bin0=(bin0[1:]+bin0[:-1])/2
        if do_plots==True:
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(1, 1, 1)
            ax2.plot(self.bin99/np.max(self.bin99), self.n99*self.kt,c='k')
            ax2.scatter(bin0,n0,marker="o",c="r")

        return np.concatenate((np.array([self.kt/self.k99,lambda_0,lambda_1]),n0,PSquerys),axis=None)
def single_predict(file_path,model_num,Input):
    Model=load_model(file_path+"/model"+str(model_num))
    return Model.predict(Input.reshape(1,-1),verbose=0)

def collect_prediction(output):
    global predict_train
    predict_train.append(output)

def predict(surface_input,n_models=50,n_p=4):
    # Select root folder of the model members
    file_path = filedialog.askdirectory()
    pool=mp.Pool(n_p)
    global predict_train
    predict_train=[]
    st=time.time()

    ## Asynchronous predictions
    for i in range(n_models):
        pool.apply_async(single_predict, args=(file_path,i,surface_input.reshape(1,-1)), callback=collect_prediction)
    pool.close()
    pool.join()
    ## Synchronous predictions
    #predict_train=[pool.apply(single_predict, args=(file_path,i,surface_input.reshape(1,-1))) for i in range(n_models)]


    ## predict one-by-one
    #for i in range(n_models):
    #    Model=load_model(file_path+"/model"+str(i))
    #    predict_train.append(Model.predict(surface_input.reshape(1,-1),verbose=0))
    et=time.time()
    pool.close()
    executionT=et-st
    print("\n Execution time:"+str(np.round(executionT,2)))
    predict_train=np.array(predict_train)
    predict_uncertainty=np.std(predict_train)
    prediction=np.mean(predict_train)
    print("\n Predicted ks/k99="+str(np.round(prediction,2)))
    return prediction,predict_uncertainty
