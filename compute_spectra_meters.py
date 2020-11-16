#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:06:20 2017

@author: Alexis Klimasewski
revised by: Tara Nye 

inputs: reads in paths to the instrument corrected  and cut HHN and HHE channel sac files

method: reads in trace data from sac files
calls bin_spec function to return evenly spaced log bins and binned data,
takes the average of the N and E components

outputs: writes bins and binned spectra into the record_spectra directory
"""

# Standard Library Imports 
# import matplotlib.pyplot as plt
# plt.style.use("ggplot")
from obspy import read
from mtspec import mtspec
import os
import os.path as path
from glob import glob
import numpy as np
import pandas as pd
import time

# Local Imports
from spec_func import bin_spec
from spec_func import bin_max_err
import eqs_main_fns as emf


# working directory here
working_dir = '/home/tnye/kappa/'

# path to corrected seismograms
event_dirs = sorted(glob(working_dir + '/data/waveforms/filtered' + '/Event*'))
outpath = working_dir + 'data/record_spectra'

# Local Q
Q = 427.5 

# Local shear wave velocity (B)
B = 3.1

# Number of frequency bins
nbins = 150

# Events flatfile
event_df = pd.read_csv(f'{working_dir}/data/flatfiles/SNR_5_file.csv')

# Flatfile station and event data
df_stns = np.array(event_df['Name'])
df_stlons = np.array(event_df['Slon'])
df_stlats = np.array(event_df['Slat'])
df_stelvs = np.array(event_df['Selv'])

df_origins = np.array(event_df['OrgT'])
df_hyplons = np.array(event_df['Qlon'])
df_hyplats = np.array(event_df['Qlat'])
df_hypdepths = np.array(event_df['Qdep'])

##make event directories within corrected local data
##make a directory for each event
events = []
for i in range(len(event_dirs)):
    events.append(path.basename(event_dirs[i]))
for i in range(len(events)):
    if not path.exists(outpath + '/' + events[i]):
        os.makedirs(outpath + '/'  + events[i])

for i in range(len(event_dirs)):
    
    # Start timing 
    t1 = time.time()
    
    # Event name and coordiantes
    event = events[i][6:]
    yr = event.split('_')[0]
    mth = event.split('_')[1]
    day = event.split('_')[2]
    hr = event.split('_')[3]
    minute = event.split('_')[4]
    sec = event.split('_')[5]
    hyplon = df_hyplons[np.where(df_origins==f'{yr}-{mth}-{day} {hr}:{minute}:{sec}')[0][0]]
    hyplat = df_hyplats[np.where(df_origins==f'{yr}-{mth}-{day} {hr}:{minute}:{sec}')[0][0]]
    hypdepth = df_hypdepths[np.where(df_origins==f'{yr}-{mth}-{day} {hr}:{minute}:{sec}')[0][0]]/1000
    
    print(i)
    print('binning and fft of event: ' + event)
    
    # Get list of HNN files to obtain list of station names from 
    # recordpaths = glob(working_dir + 'data/waveforms/cut_WFs/Event_' + event +'/*_*_HHN*.sac')#full path for only specified channel
    #stns = [(x.split('/')[-1]).split('_')[1] for x in recordpaths]
    stns = np.unique(df_stns)

    # Loop through stations 
    for j in range(len(stns)):
        
        # Get East and North components
        recordpath_E = glob(working_dir + '/data/waveforms/filtered/Event_' + event +'/*_' + stns[j] + '_HHE*.sac')
        recordpath_N = glob(working_dir + '/data/waveforms/filtered/Event_' + event +'/*_' + stns[j] + '_HHN*.sac')
        
        # Make sure both a North and East component exist
        if(len(recordpath_E) == 1 and len(recordpath_N) == 1):

            base_N = path.basename(recordpath_N[0])
            base_E = path.basename(recordpath_E[0])
    
            # Get sac info
            network = base_N.split('_')[0]
            station = base_N.split('_')[1]
            full_channel_N = base_N.split('_')[2]
            full_channel_E = base_E.split('_')[2]
            
            # Read in North stream
            N_stream = read(recordpath_N[0])
            N_tr = N_stream[0]
            N_data = N_tr.data
            
            # Read in East stream
            E_stream = read(recordpath_E[0])
            E_tr = E_stream[0]
            E_data = E_tr.data
            
            # Get station coordinates
            stlon = df_stlons[np.where(df_stns==stns[j])[0][0]]
            stlat = df_stlats[np.where(df_stns==stns[j])[0][0]]
            stelv = df_stelvs[np.where(df_stns==stns[j])[0][0]]/1000
            
            # Calculate hypocentral distance (km)
            rhyp = emf.compute_rhyp(stlon, stlat, stelv, hyplon, hyplat, hypdepth)
            
            
            ############## Begin fourier transform to get spectra #############
            
            # mtspec returns power spectra (square of spectra)
            
            ##### North component 
            N_spec_amp, N_freq , N_jack, N_fstat, N_dof =  mtspec(N_data, delta=N_tr.stats.delta, time_bandwidth=4, number_of_tapers=7, quadratic=True, statistics=True)
			
            # Find standard deviationnm 
            sigmaN = (N_jack[:,1] - N_jack[:,0])/3.29
           
            # Get arrays of power spectra and frequencies 
            spec_array_N = np.array(N_spec_amp)
            freq_array_N = np.array(N_freq)
            
            
            ##### East component 
            E_spec_amp, E_freq , E_jack, E_fstat, E_dof =  mtspec(E_data, delta=E_tr.stats.delta, time_bandwidth=4, number_of_tapers=7, quadratic=True, statistics=True)
       
            # Find standard deviation
            sigmaE = (E_jack[:,1] - E_jack[:,0])/3.29          
            
            # Get arrays of power spectra and frequencies 
            spec_array_E = np.array(E_spec_amp)
            freq_array_E = np.array(E_freq)
            
            # If evenly sampled
            if(len(spec_array_E)==len(spec_array_N)):
                
                # here we bin into evenly spaced bins with frequency
                # spectra is power spectra so add the two components
                data_NE_2 = spec_array_E + spec_array_N
                freq_NE_2 = freq_array_E + freq_array_N
                
                #now data is NE power spectra
                #take the square root for normal amplitude spectra
                data_NE = np.sqrt(data_NE_2)
                freq_NE = np.sqrt(freq_NE_2)

                sigma = np.sqrt((spec_array_N/data_NE**2.)*sigmaN + ((spec_array_E/data_NE**2.)*sigmaE))
                
                # Remove path
#                data_NE = data_NE/(np.exp(-np.pi*rhyp*freq_NE/Q*B))
                
                #0.1-end
                bins, binned_data = bin_spec(data_NE[6:-1], freq_NE[6:-1], num_bins=nbins)
                bins_sig, binned_sig = bin_max_err(sigma[6:-1], freq_NE[6:-1], num_bins=nbins)

                #make sure that all spec is a number
                if (np.isnan(binned_data).any() == False):
                ##write to file
                    outfile = open(outpath + '/Event_'+ event + '/'+ network + '_' + station + '_' + 'HHNE' + '__' + event + '.out', 'w')
                    data = np.array([bins, binned_data, binned_sig])
                    data = data.T
                    outfile.write('#bins \t \t vel_spec_NE_m \t binned_sig \n')
                    np.savetxt(outfile, data, fmt=['%E', '%E', '%E'], delimiter='\t')
                    outfile.close()
    t2 = time.time()
    print('time for event: (s)', (t2-t1))
        
