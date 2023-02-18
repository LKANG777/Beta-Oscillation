#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:29:17 2023

@author: kangling

Project: Calculate phase spatial features to distinguish different types of waves. See paper: 
"""
 
import matplotlib.pyplot as plt
import numpy as np
import cmath
import scipy.signal
from scipy.signal import hilbert
from scipy.signal.signaltools import convolve2d
 
# =============================================================================
# Functions 
# =============================================================================
def function_wave_phase_speed(phase_speed,wave):
    '''
    Calculate the speed for different types of waves.
    
    Parameters
    ----------
    phase_speed : list [T]
       List of speed.
    wave : list
       List of different types of waves(bool). 
       
    Returns
    -------
    np.array
        The speed for different types of waves.
    '''
    speed= np.ma.array(phase_speed,mask= ~wave)
    return speed  

def function_wave_direction(gradient_coherence,wave):
    '''
    Calculate the wave direction for different types of waves.
    
    Parameters
    ----------
    gradient_coherence : list [T]*[10*10]
       List of speed.
    wave : list
       List of different types of waves(bool). 
       
    Returns
    -------
    np.array
       The wave direction for different types of waves.
    '''
    direction=[]
    for i in range(len(wave)):
        if (wave[i]==1): 
            direction.append(gradient_coherence[i])
    return direction  

def function_phase_gradient( phase):
    '''
    Calculate the phase gradient (see Methods in the paper) 
    The gradient along the first dimension (x) is encoded in the real part, 
    and the gradient along the second dimension (y) is encoded in the imaginary
    part. 

    Parameters
    ----------
    phase : np.array [10]*[10]
        ND numpy array of phase.
        
    Returns
    -------
    np.array
        The phase gradient.  
    '''
    phase_gradient=np.zeros((10,10),dtype=complex)
    for x in range(10)[:]:
        for y in range(10)[:]:
            add_x=0
            add_number_x=0
            add_y=0
            add_number_y=0
            for row_r in (-2,-1,1,2):
                if(9>=x+row_r>=0):
                    if(row_r<0):alpha=0
                    else:alpha=np.pi
                    ele_i= x+row_r 
                    # print(ele_i)
                    add_x+=((( (phase[ele_i,y]- phase[x,y])+np.pi)%(2*np.pi)-np.pi)/np.abs(row_r)*cmath.exp(1j*alpha))
                    add_number_x+=1
                    
            for col_r in (-2,-1,1,2):
                if(9>=y+col_r>=0):
                    if(col_r<0):alpha=0.5*np.pi
                    else:alpha=1.5*np.pi
                    ele_j= y+col_r 
                    # print(ele_j)
                    add_y+=((( (phase[x,ele_j]- phase[x,y])+np.pi)%(2*np.pi)-np.pi)/np.abs(col_r)*cmath.exp(1j*alpha))
                    add_number_y+=1
            phase_gradient[x,y]=(add_x/add_number_x+add_y/add_number_y)
    return  phase_gradient 

def function_gradient_coherence(phase_directionality):
    '''
    Calculate the gradient coherence (see Methods in the paper). 

    Parameters
    ----------
    phase_directionality : np.array [10]*[10]
        ND numpy array of phase.
        
    Returns
    -------
    np.array
        The gradient coherence.  
    '''
    gradient_coherence=np.zeros((10,10),dtype=complex)
    for x in range(10) :
        for y in range(10) :
            add=0
            add_number=0
            for row_r in range(-2,3):
                for col_r in range(-2,3):
                    if((9>=x+row_r>=0)&(9>=y+col_r>=0)):
                        ele_i= x+row_r
                        ele_j= y+col_r 
                        # print(ele_i,ele_j)
                        add+= phase_directionality[ele_i,ele_j] 
                        add_number+=1
            
            gradient_coherence[x,y]=add/add_number

    return  gradient_coherence

def function_phase_speed(fre_m):
    '''
    Calculate the speed (see Methods in the paper). 

    Parameters
    ----------
    fre_m : float
         The mean frequency of the respective beta bands.
        
    Returns
    -------
    list
        The phase speed.  
    '''
    electrode_spacing=0.4  #Spacing between electrodes for the Utah arrays (mm) 
    phase_speed=(2*np.pi*fre_m/(np.abs(temporary_phase_gradient).mean())*electrode_spacing*10*1e-2)
    return phase_speed

def function_sigma_p(phase):
    '''
    Calculate the circular of the phase. 

    Parameters
    ----------
    phase : np.narry [10]*[10]
         ND numpy array of phase.
        
    Returns
    -------
    int
        Sigma_p.  
    '''
    sigma_p=0
    for p_i in phase.flatten():
        sigma_p+=cmath.exp(1j*p_i)
    sigma_p=np.abs(sigma_p/100)
    return sigma_p

def function_sigma_g(phase_directionality):
    '''
    Calculate the circular of the phase_directionality. 

    Parameters
    ----------
    phase_directionality : np.narry [10]*[10]
         ND numpy array of phase_directionality.
        
    Returns
    -------
    int
       Sigma_g.  
    '''
    sigma_g=np.abs(np.mean(phase_directionality))
 
    return sigma_g
    

def function_count_critical(phase_gradient_list ):
    '''
    Find critical points in the phase gradient map.

    Parameters
    ----------
    phase_gradient_list : np.array [T]*[10*10]
    The list of the phase gradient.
        
    Returns
    -------
    nclockwise : np.array
        The number of clockwise centers found at each time point.
    nanticlockwise : np.array
        The number of anticlockwise centers found at each time point.
    nsaddles : np.array
        The number of saddle points.
    nmaxima : np.array
        The number of local maxima.
    nminima : np.array
        The number of local minima.
    '''
    data =  phase_gradient_list
    
    # curl  
    curl = np.complex64([[-1-1j,-1+1j],[1-1j,1+1j]])
    curl = convolve2d(curl,np.ones((2,2))/4,'full')
    winding = np.array([convolve2d(z,curl,'same','symm').real for z in data])

    # cortical points
    ok        = ~(np.abs(winding)<1e-1)[...,:-1,:-1]
    ddx       = np.diff(np.sign(data.real),1,1)[...,:,:-1]/2
    ddy       = np.diff(np.sign(data.imag),1,2)[...,:-1,:]/2
    saddles   = (ddx*ddy==-1)*ok
    maxima    = (ddx*ddy== 1)*(ddx==-1)*ok
    minima    = (ddx*ddy== 1)*(ddx== 1)*ok
    sum2 = lambda x: np.sum(np.int32(x),axis=(1,2))
    nclockwise = sum2(winding>3)
    nanticlockwise = sum2(winding<-3)
    nsaddles   = sum2(saddles  )
    nmaxima    = sum2(maxima   )
    nminima    = sum2(minima   )
    return nclockwise, nanticlockwise, nmaxima, nminima


#Effective period
def function_get_edges(wave):
    '''
    Find the starts and the ends of the wave.
     
    Parameters
    ----------
    wave : list (bool)
        List of waves.
    Returns
    -------
    np.array
        The array of starts and ends.  
    '''
 
    if len(wave)<1:
        return np.array([[],[]])
    starts =  np.where(np.diff(np.int32(wave))==1)
    stops  =  np.where(np.diff(np.int32(wave))==-1)
    if wave[0]: 
       starts=np.insert(starts,0,int(0))
    if wave[-1]: 
       stops=np.insert(stops,int(len(stops[0])),int(len(wave))) 
       
    if (isinstance(stops,tuple)):  
            stops=np.array(stops[0])
    if (isinstance(starts,tuple)):
            starts=np.array(starts[0])
    return np.array([starts+1, stops+1])
 
   
def function_set_edges(edges,L):
    '''
    Set the starts and the ends of the wave.
     
    Parameters
    ----------
    edges :  np.array 
        The array of the starts and the ends of the waves.
    L :  int
        The length of the wave.
        
    Returns
    -------
    np.array
        The array of wave.  
    '''
    x = np.zeros(shape=(L,),dtype=np.int32)
    for (a,b) in edges:
        x[a:b]= 1
    return x
 
def function_remove_short(wave,cutoff):
    '''
    Remove the short duration (less than the cutoff) of the waves.
     
    Parameters
    ----------
    wave :  list (bool) 
        List of wave.
    cutoff :  int
        The threshold for the duration.
        
    Returns
    -------
    np.array
        The array of the effective wave. 
    '''
    a,b  = function_get_edges(wave)
    gaps = b-a
    keep = np.array([a,b])[:,gaps>cutoff]
    newgaps = function_set_edges(keep.T,len(wave))
    return newgaps 
 
def function_wave_duration(wave,cutoff):
    '''
    Calculate the duration of the waves.
     
    Parameters
    ----------
    wave :  np.array 
        The array of the waves.
    cutoff :  int
        The threshold for the duration.
        
    Returns
    -------
    np.array
        The array of wave duration.  
    '''
    a,b  = function_get_edges(wave)
    gaps = b-a
    effective_gaps= np.ma.array(gaps,mask= ~(gaps >cutoff) )
    return effective_gaps.compressed()

def function_find_delete_list(N,width):
    '''
    Find the effective modules (See the introduction of the network in the paper).
     
    Parameters
    ----------
    N : int
        The length of the array.
    width : int
        The length of the fixed modules of the array.
        
    Returns
    -------
    list
        The list of effective modules.  
    '''
    
    delete_list=[]
    for i in range(sur_width):
        for j0 in range(i*N,(i+1)*N):
            delete_list.append(j0)
        for j1 in range(N):
            jj1=j1*N+i
            delete_list.append(jj1)
    for i in range(N-sur_width,N):
        for j0 in range(i*N,(i+1)*N):
            delete_list.append(j0)
        for j1 in range(N):
            jj1=j1*N+i
            delete_list.append(jj1)
    return delete_list
    

# =============================================================================
# Data analysis
# =============================================================================

#%%
# Load data  
# =============================================================================

# Parameters about Network
N=28
sur_width=7
fix_width=2
sim_width=5
delete_list=[]
delete_list=function_find_delete_list(N-2*fix_width,sur_width)
n_skiprows=5000
row_n=10000 # The duration of simulation (ms).
 
total_wave_kind_pie=[]
total_wave_kind=[]  
total_wave_speed=[]
total_amplitude=[]
total_sigma_p=[]
total_planar_direction=[]

# Parameters about the model (See parameter table in the paper) 
N_noise=2e4 # Finite-size noise, N_E,N_I=N_noise*[0.8,0.2]
l =2.0 # Excitatory connectivity range
D=1.3 #Propagation delay between to nearest E-I modules (ms)
omega_ie=1.0  #Recurrent synaptic coupling strength (E to I)
omega_ei=2.08 #Recurrent synaptic coupling strength (I to E)
omega_ee=0.96 #Recurrent synaptic coupling strength (E to E)
omega_ii=0.87 #Recurrent synaptic coupling strength (I to I)
niu=3 #External input amplitude fluctuations (Hz)
eta_c=0.4  #Proportion of global external inputs
tau_ext=25 #Correlation time of external input fluctuations (ms)
 
filename=str("%d"% N)+"_"+str("%.2f"% D)+"_"+str("%d"%N_noise)+"_"+str("%.2f"%l)+"_"+str("%.2f"%omega_ee)+"_"+str("%.2f"%omega_ei)+"_"+str("%.2f"%omega_ie)+"_"+str("%.2f"%omega_ii)+"_"+str("%.2f"%niu)+"_"+str("%.2f"%eta_c)+"_"+str("%.2f"%tau_ext)

# Excitatory Current 
data_current = np.load(filename+"_current_tau_fix_e.npy")
data_current1=np.delete(data_current,delete_list,axis=1) #Data shape [T]*[10*10]
# Common external input
data_external_input= np.load(filename+"_current_tau_fix_xi_g.npy")
data_external_input1=np.delete( data_external_input,delete_list,axis=1)
data_external_input_average=data_external_input1[:row_n] 
#%%
# Perform a series transform to get analytical signals 
 
# Butterworth filter    
# =============================================================================
channel_analogsignal=data_current1.T  
b, a = scipy.signal.butter(3, [13,30], 'bandpass',fs=1e3)
filter_channel_analogsignal = scipy.signal.filtfilt(b, a, channel_analogsignal)
 
# Z-transform
# =============================================================================
z_filter_channel_analogsignal=(filter_channel_analogsignal - np.mean(filter_channel_analogsignal,axis=1)[:,None]) / np.std(filter_channel_analogsignal,axis=1)[:,None]  
    
# Hilbert transform
# =============================================================================
h_channel_analogsignal = hilbert(z_filter_channel_analogsignal,axis=1 )

#%%
# Plot the raw signals and analytical signals
plot_time=1000
fig=plt.figure(figsize=(8,4))
shape=(1,2)
rowspan=1
colspan=1
ax=plt.subplot2grid(shape, (0,0),rowspan ,colspan  )   
sp=ax.imshow( channel_analogsignal.real,aspect='auto' )
ax.set_xlim([0,plot_time])
cb=fig.colorbar(sp,ax=ax,shrink=0.5)
ax.set_title('Raw signals')
ax.set_ylabel('Position')
ax.set_xlabel('Time (ms)')

ax=plt.subplot2grid(shape, (0,1),rowspan ,colspan  )   
sp=ax.imshow( h_channel_analogsignal.real,aspect='auto' )
ax.set_xlim([0,plot_time])
cb=fig.colorbar(sp,ax=ax,shrink=0.5)
ax.set_title('Signals after Hibert transform')
ax.set_ylabel('Position')
ax.set_xlabel('Time (ms)')
plt.tight_layout()
#%%
 
# Calculate the phases, amplitudes, phase gradients, phase gradient coherence, 
# directionalities of the analytical signals.
 
signal_list=[]           # Hilbert signal
phase_list=[]             # Hilbert signal phase
amplitude_list=[]
phase_gradient_list=[]
phase_directionality_list=[]
gradient_coherence_list=[]
phase_speed_list=[]

sigma_p=[]
sigma_g=[]


for t_i in range(len(channel_analogsignal[0]))[:row_n]:
    temporary_phase=[]
    temporary_amplitude=[]
    temporary_signal=[]
    
    temporary_signal= h_channel_analogsignal[:,t_i]
    re_temporary_signal=temporary_signal.reshape((10,10))            
    temporary_phase=np.angle(re_temporary_signal)
    temporary_amplitude=np.abs(re_temporary_signal)
    
# Phase gradient
# =============================================================================
    temporary_phase_gradient=function_phase_gradient(temporary_phase)

# Phase speed          
# =============================================================================
    fre_m=21.5 #Hz
    temporary_phase_speed=function_phase_speed(fre_m)

# Phase directionality
# =============================================================================
    temporary_phase_directionality=temporary_phase_gradient/np.abs(temporary_phase_gradient)
 
# Gradient coherence
# =============================================================================
    temporary_gradient_coherence=function_gradient_coherence(temporary_phase_directionality)
     
# Circular variance of phases (sigma_p)       
# =============================================================================   
    temporay_sigma_p=function_sigma_p(temporary_phase)
 
# Circular variance of phase directionality (sigma_g)       
# =============================================================================
    temporay_sigma_g=np.abs(np.mean(temporary_phase_directionality))
    
#Save data 
    sigma_p.append(temporay_sigma_p)
    sigma_g.append(temporay_sigma_g)
    
    signal_list.append( temporary_signal)           
    phase_list.append(temporary_phase.flatten())              
    amplitude_list.append(temporary_amplitude.flatten()) 
    phase_gradient_list.append((temporary_phase_gradient.flatten()))
    phase_directionality_list.append(temporary_phase_directionality.flatten())
    gradient_coherence_list.append(temporary_gradient_coherence.flatten())
    phase_speed_list.append(temporary_phase_speed)
            
            
#%%
# Plot the characteristics of the analytical signals

fig=plt.figure(figsize=(9,8))
shape=(3,2)
rowspan=1
colspan=1
ax=plt.subplot2grid(shape, (0,0),rowspan ,colspan  )
 
sp=ax.imshow(np.array(signal_list).real,   cmap='terrain' ,aspect='auto'  )
ax.set_title('Signals')
cb=fig.colorbar(sp,ax=ax,shrink=0.5)
ax.set_xlabel('Position')
ax.set_ylabel('Time (ms)')
ax.set_ylim([0,plot_time]) 

ax=plt.subplot2grid(shape, (0,1),rowspan ,colspan  )
im = ax.imshow(phase_list,cmap='twilight_shifted',vmin=-np.pi,vmax=np.pi ,aspect='auto' )
ax.set_title('Phases')
cb=fig.colorbar(im,ax=ax,shrink=0.5)
ax.set_xlabel('Position')
ax.set_ylabel('Time (ms)')
ax.set_ylim([0,plot_time]) 

ax=plt.subplot2grid(shape, (1,0),rowspan ,colspan  )
sp=ax.imshow(amplitude_list,cmap='terrain',aspect='auto'  )
ax.set_title('Amplitudes')
cb=fig.colorbar(sp,ax=ax,shrink=0.5)
ax.set_xlabel('Position')
ax.set_ylabel('Time (ms)')
ax.set_ylim([0,plot_time]) 

ax=plt.subplot2grid(shape, (1,1),rowspan ,colspan  )
sp=ax.imshow(np.abs(phase_gradient_list),cmap='terrain',aspect='auto'  )
ax.set_title('$Phase\ gradients$')
cb=fig.colorbar(sp,ax=ax,shrink=0.5)
ax.set_xlabel('Position')
ax.set_ylabel('Time (ms)')
ax.set_ylim([0,plot_time]) 
   
ax=plt.subplot2grid(shape, (2,0),rowspan ,colspan  )
sp=ax.imshow(np.angle(phase_directionality_list),cmap='terrain',aspect='auto'  )
ax.set_title('$Phase\ directionalities$')
cb=fig.colorbar(sp,ax=ax,shrink=0.5)
ax.set_xlabel('Position')
ax.set_ylabel('Time (ms)')
ax.set_ylim([0,plot_time]) 

ax=plt.subplot2grid(shape, (2,1),rowspan ,colspan  )
sp=ax.imshow(np.abs(gradient_coherence_list),cmap='terrain',aspect='auto'  )
ax.set_title('$Gradient\ coherences$')
cb=fig.colorbar(sp,ax=ax,shrink=0.5)
ax.set_xlabel('Position')
ax.set_ylabel('Time (ms)')
ax.set_ylim([0,plot_time]) 
plt.tight_layout( )

#%%
# Plot sigma_p, sigma_g, and speed 
   
fig=plt.figure(figsize=(6,9))
shape=(3,1)
rowspan=1
colspan=1

ax=plt.subplot2grid(shape, (0,0),rowspan ,colspan  )
sp=ax.plot(sigma_p  )
ax.set_ylabel(r'$\sigma_p$')
ax.set_xlabel('Time (ms)')
 
ax=plt.subplot2grid(shape, (1,0),rowspan ,colspan  )
sp=ax.plot(sigma_g  )
ax.set_ylabel(r'$\sigma_g$') 
ax.set_xlabel('Time (ms)')

ax=plt.subplot2grid(shape, (2,0),rowspan ,colspan  )
sp=ax.plot(phase_speed_list)
ax.set_ylabel(r'$Speed\ (cm/s)$')
ax.set_xlabel('Time (ms)')
plt.tight_layout( )
 
#%%
# =============================================================================
# Wave classification
# =============================================================================

# Synchronized wave and planar wave
# The threshold for the circulars phase and phase gradient to distinguish planar waves and synchronized waves
judge_theta=[0.85,0.5]
syn=((np.array(sigma_p) >judge_theta[0]) & (np.array(sigma_g)<=judge_theta[1]))
planar=(np.array(sigma_g)>judge_theta[1])

# Radial wave and spiral wave   
cps = function_count_critical(np.array(phase_gradient_list).reshape((len(phase_gradient_list),10,10))) 
nclockwise, nanticlockwise, nmaxima, nminima=cps
clockwise = nclockwise+nanticlockwise
peaks = nmaxima+nminima
radial = (peaks ==1) & (clockwise==0)  & (~planar) & (~syn) 
circular = (clockwise==1) & (peaks ==0) & (~planar) & (~ radial)  & (~syn) 

wave_kind=syn,planar,radial,circular

# Remove too short time 
duration_threshold=5 
effective_wave =[]  
for wave_idx, wave in enumerate(wave_kind):
    effective_wave.append (function_remove_short(wave,duration_threshold))
    
unclass = ~((np.sum(effective_wave,axis=0))>0 )
syn, planar,radial,circular =effective_wave 
all_wave_kind= syn, planar,radial,circular,(unclass+0)

wave_kind_pie=np.array(np.array(all_wave_kind).mean(1)) 

##%%     
 
# The characteristics for different types of wave

# Speed
# =============================================================================
wave_speed=[]    
for wave_idx, wave in enumerate(all_wave_kind ):
    bool_wave=(wave>0)
    wave_speed.append(function_wave_phase_speed(phase_speed_list, bool_wave))
 
# # Duration 
# # =============================================================================
# speed_duration=[]    
# for wave_idx, wave in enumerate(all_wave_kind ):
#     bool_wave=(wave>0)
#     speed_duration.append(function_wave_duration(bool_wave, duration_threshold) )

# Common external input
# =============================================================================
wave_noise=[]    
for wave_idx, wave in enumerate(all_wave_kind ):
    bool_wave=(wave>0)
    noise=(data_external_input_average[bool_wave,:])
    wave_noise.append(noise.flatten() )
 
# Amplitude
# =============================================================================
amplitude_average=np.array(amplitude_list)
wave_amplitude=[]    
for wave_idx, wave in enumerate(all_wave_kind ):
    bool_wave=(wave>0)
    amplitude=(amplitude_average[bool_wave,:])
    wave_amplitude.append(amplitude.flatten() )
 
 
#%%
# Plot the characteristics of different waves

lbs=['Syn.', 'Pla.','Rad.','Spi.','Ran.']     
fig=plt.figure(figsize=(14,8))
shape=(4,6)
rowspan=1
colspan=1
r_ight,t_op=0.6,1.3


for j in range(len(lbs)):
    ax=plt.subplot2grid(shape, (0,j),rowspan ,colspan )
    sp=ax.plot(all_wave_kind[j]  )
    ax.set_title(lbs[j])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Bool value of wave')
    ax=plt.subplot2grid(shape, (1,j),rowspan ,colspan  )
    ax.hist(wave_speed[j].compressed(), bins=20, density=True, facecolor="blue", edgecolor="black", alpha=0.7) 
    ax.text(r_ight,t_op,'Mean=%.2f'%np.mean(wave_speed[j]),  ha='left', va='top',transform= ax.transAxes)
    if j==0:
        ax.set_ylabel('Speed(cm/s)')
    ax=plt.subplot2grid(shape, (2,j),rowspan ,colspan  )
    ax.hist(wave_noise[j] , bins=20, density=True, facecolor="blue", edgecolor="black", alpha=0.7) 
    ax.text(r_ight,t_op,'Mean=%.2f'%np.mean(wave_noise[j]),  ha='left', va='top',transform= ax.transAxes)
    if j==0:
        ax.set_ylabel(r'$\eta_c$')
    ax=plt.subplot2grid(shape, (3,j),rowspan ,colspan  )
    ax.hist(wave_amplitude[j] , bins=20, density=True, facecolor="blue", edgecolor="black", alpha=0.7) 
    ax.text(r_ight,t_op,'Mean=%.2f'%np.mean(wave_amplitude[j]),  ha='left', va='top',transform= ax.transAxes)
    if j==0:
        ax.set_ylabel('Amplitude')
plt.tight_layout()
#%% 
# Plot wave pie
fig=plt.figure(figsize=(8,4))
shape=(1,1)
rowspan=1
colspan=1
color_lists=['#1b9e92','#277FB0' ,'#ffa473','#8D376E','#7b4b99',] 
ax=plt.subplot2grid(shape, (0,0),rowspan  ,colspan  )  
wedges, texts  = ax.pie(wave_kind_pie,colors=color_lists)
wave_kind_pie_por=wave_kind_pie/np.sum(wave_kind_pie)
def func(pct, data):
    return "{:.1f}%\n".format(pct/np.sum(data)*100 )
 
new_wave_lbs=[] 
for i in range(5):
    new_wave_lbs.append(lbs[i]+" "+func(wave_kind_pie[i],wave_kind_pie))
ax.legend(wedges,  new_wave_lbs  ,
          title="wave",
          loc="center left",
          bbox_to_anchor=(1.0, 0.4, 0.2, 0.05))
ax.set_title('Wave types') 
plt.tight_layout()
 
#%%
# =============================================================================
# Power spectra & beta duration
# =============================================================================

f, t, Sxx = scipy.signal.spectrogram(channel_analogsignal,window='hanning', nperseg=512, noverlap=500, fs=1e3 )
#%%
# Plot power spectra and beta burst statistics
fig=plt.figure(figsize=(8,8))
shape=(4,2)
rowspan=1
colspan=1

ax=plt.subplot2grid(shape, (0,0),rowspan ,colspan=2  ) 
sp=ax.pcolormesh(t*1e3, f, Sxx.mean(0), shading='gouraud',cmap='terrain')
cb=fig.colorbar(sp,ax=ax,shrink=0.5)
ax.set_xlabel('Time' )
ax.set_ylabel('Frequency (Hz)' )
ax.set_ylim((5,45))

ax=plt.subplot2grid(shape, (1,0),rowspan  ,colspan=2   )
ax.plot(channel_analogsignal.mean(0))
ax.set_xlabel('Time' )
ax.set_ylabel('Raw signal' )

ax=plt.subplot2grid(shape, (2,0),rowspan  ,colspan=2   )
ax.plot(h_channel_analogsignal.mean(0), label='Real part of signal')
ax.plot(np.abs( (h_channel_analogsignal.mean(0))),label='Amplitude')
amplitude_sort=np.sort(np.array(amplitude_list).flatten())
amplitude_threshold= amplitude_sort[int(0.75*len(amplitude_sort))]
ax.hlines(amplitude_threshold,0,len(h_channel_analogsignal.mean(0)),linestyle='dashed',color='k',label='Threshold')
ax.set_xlabel('Time' )
ax.set_ylabel('Analytical signal' )
ax.legend()


ax=plt.subplot2grid(shape, (3,0),rowspan ,colspan=1  )
amplitude_duration=[]
amplitude_array=np.array(amplitude_list).mean(1)
bool_amplitude_duration=(amplitude_list>amplitude_threshold)
for i in range(100):
    amplitude_duration.append(function_wave_duration(bool_amplitude_duration[:,i], 0) )
ax.hist( np.concatenate(amplitude_duration) , bins=50, density=True,color='grey')
ax.text(r_ight,t_op,'Mean=%.1f ms'%np.mean(np.array(np.concatenate(amplitude_duration))),  ha='left', va='top',transform= ax.transAxes)
ax.set_ylabel('Beta burst duration (ms)')
plt.tight_layout()

