#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_simulation.py
---------------------

Script to plot simulated network activity of 
2D rate model of motor cortex, see paper
Kang, Ranft & Hakim. eLife (2023).

To load specific simulation output change parameters/
filename accordingly.

Created March 2023

Author: L. Kang, kangling2017@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
 

plt.rcParams.update({'font.size': 14, 
                     'font.family': 'Times New Roman', 
                     'text.usetex': False})

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

# Network and simulation parameters 
N = 28 # linear size of grid
fix_width = 2 # number of rows of midules with fixed firing rates
sim_width = 5 # number of rows (from center) of modules kept for analysis 
              # (5 corresponds to central 10x10 grid) 
sur_width = int(N/2-fix_width-sim_width) # surplus modules simulated but not considered

# remove data not used for analysis using 'delete_list'
delete_list=[]
delete_list=function_find_delete_list(N-2*fix_width,sur_width)        

# Model parameters (example data corresponds to parameters SN of the 
# paper, see also Table 1); change accordingly for simulation data with
# different parameters 
N_noise = 2e4 # Finite-size noise, N_E,N_I=N_noise*[0.8,0.2]
l = 2.0 # Excitatory connectivity range
D  =  1.3 # Propagation delay between to nearest E-I modules (ms)
omega_ie = 1.0  # Recurrent synaptic coupling strength (E to I)
omega_ei = 2.08 # Recurrent synaptic coupling strength (I to E)
omega_ee = 0.96 # Recurrent synaptic coupling strength (E to E)
omega_ii = 0.87 # Recurrent synaptic coupling strength (I to I)
nu = 3 # External input amplitude fluctuations (Hz)
eta_c = 0.4  # Proportion of global external inputs
tau_ext = 25 # Correlation time of external input fluctuations (ms)
 

# Load data 
# ---------

filename = str("%d"% N)+"_"+str("%.2f"% D)+"_"+str("%d"%N_noise)+"_"+str("%.2f"%l)+"_"+str("%.2f"%omega_ee)+"_"+str("%.2f"%omega_ei)+"_"+str("%.2f"%omega_ie)+"_"+str("%.2f"%omega_ii)+"_"+str("%.2f"%nu)+"_"+str("%.2f"%eta_c)+"_"+str("%.2f"%tau_ext)

# Excitatory Current 
data_current = np.loadtxt(filename+"_current_tau_fix_e.txt",delimiter=',',usecols=range((N-2*fix_width)*(N-2*fix_width)))
data_current1=np.delete(data_current,delete_list,axis=1) #Data shape [T]*[10*10]

# Inhhibitory Current 
data_current_i = np.loadtxt(filename+"_current_tau_fix_i.txt",delimiter=',',usecols=range((N-2*fix_width)*(N-2*fix_width)))
data_current_i1=np.delete(data_current_i,delete_list,axis=1) 

# common input

data_external_input= np.loadtxt(filename+"_current_tau_fix_xi_g.txt",delimiter=',')

# Plot simulated data
# -------------------

# create figure
fig = plt.figure(figsize = (8,9))
shape = (2,1) 
rowspan = 1
colspan = 1 
ax = plt.subplot2grid(shape, (0,0),rowspan  ,colspan  )

# plot excitatory current for all modules in a color plot
sp = ax.imshow(data_current1.T,aspect = 'auto' ) 
ax.set_xlabel('Time(ms)') 
ax.set_ylabel('Position')
plt.colorbar(sp,label = '$I_E$')

# plot exc. and inh. current for a single module 
model_id = 1 
ax = plt.subplot2grid(shape, (1,0),rowspan  ,colspan  )
ax.plot(np.arange(len(data_current[:,model_id])),data_current1[:,model_id],label='$I_E$ of module '+str(model_id) )  
ax.plot(np.arange(len(data_current[:,model_id])),data_current_i1[:,model_id],label='$I_I$ of module '+str(model_id) ) 
ax.set_xlabel('Time(ms)') 
ax.legend()

# Export data as .npy for wave analysis script
# --------------------------------------------
np.save(filename+"_current_tau_fix_e.npy", data_current  )
np.save(filename+"_current_tau_fix_xi_g.npy", data_external_input  )

