# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:45:32 2024

@author: presvotscor
"""

# Import necessary modules and functions
from Normalize import normalize  # Used for normalizing signals
from Measures import get_quality# Evaluation metrics
import numpy as np  # Numeric operations
import matplotlib.pyplot as plt  # Plotting utilities

# Import custom encoding and dictionary modules
from Dico import Dico_Cos_Sin_Dirac  # ICS dictionary generation
from Encode_amplitudes import encoded_gamma, decoded_gamma  # Gamma coefficient encoding
from Encode_positions import Golomb, Golomb_encoding_pos, Golomb_decoding_pos  # Position encoding
from OMP import OMP  # Orthogonal Matching Pursuit
import time  # Timing operations

# Import test signal generation function
from get_test_signal import get_RTE_signal as get_signal

# General parameters
fs = 6400  # Sampling frequency in Hz
fn = 50    # Nominal signal frequency in Hz
N = 128    # Number of samples per window

# Input parameters
nb_signal = 2      # Number of signals to be encoded
nb_phase = 1       # Number of signal phases (1 = single phase, 3 = voltage phases, 6 = voltages + currents)
nb_w = 50          # Number of non-overlapping windows per signal


# Define the performance metric and target quality
metric = "RMSE"  # Options: "SNR", "RMSE", "MSE"
quality = 200 # For SNR: quality in -dB; for RMSE: in volts; for MSE: in volts squared

# Encoding settings
adaptive = True  # Enable adaptive encoding for positions
polarity = True  # Differential coding of positions followed by Golomb coding if True

# Initialize encoding objects
Gamma_enc = encoded_gamma()  # Gamma coefficients encoder
Gamma_dec = decoded_gamma()  # Gamma coefficients decoder
G = Golomb()  # Golomb coder instance
G_enc = Golomb_encoding_pos(adaptive)  # Golomb position encoder
G_dec = Golomb_decoding_pos(adaptive)  # Golomb position decoder
O = OMP()  # Orthogonal Matching Pursuit instance

# Header and coefficient bit allocations
nb_kx = 5      # Bits for the scaling factor
nb_n_vect = 7  # Maximum bits for encoding the number of significant vectors
nb_n_gamma = 8 # Maximum bits per gamma coefficient

# Metrics and results storage
SNR = np.zeros((nb_signal, nb_phase, nb_w))  # Signal-to-noise ratio for each window
RMSE = np.zeros((nb_signal, nb_phase, nb_w))  # Root mean square error for each window

R_h = np.ones((nb_signal, nb_phase, nb_w)) * (nb_kx + nb_n_vect + nb_n_gamma)  # Header bits
R_gamma = np.zeros((nb_signal, nb_phase, nb_w))  # Bits used for gamma coefficients
R_pos = np.zeros((nb_signal, nb_phase, nb_w))  # Bits used for position encoding

x_real = np.zeros((nb_signal, nb_phase, nb_w * N))  # Original signal windows
x_rec = np.zeros((nb_signal, nb_phase, nb_w * N))  # Reconstructed signal windows

# ICS dictionary parameters
f_factor = 1      # Frequency factor for the ICS dictionary
f_max = fs / 2    # Maximum frequency considered in the ICS dictionary (Nyquist limit)

# Create ICS dictionary
Dico = Dico_Cos_Sin_Dirac(fs, f_factor)  # Initialize dictionary object
D = Dico.Creat_dico(N, f_max)  # Generate dictionary with N samples per window

size_D = np.size(D, 1)  # Number of atoms in the dictionary

# Golomb coding parameters
m_values = 5  # Golomb parameter m
n_value_max = 8  # Maximum bits for representing positions
dico_values = G_enc.creat_dico(m_values, size_D, None)  # Create Golomb coding dictionary for positions




# Record the start time for performance measurement
start_time = time.time()

# Loop through all signals to encode
for id_signal in range(nb_signal):
    # Retrieve signal components (voltage and current phases)
    v1 = get_signal(id_signal)[0]
    v2 = get_signal(id_signal)[1]
    v3 = get_signal(id_signal)[2]
    i1 = get_signal(id_signal)[3]
    i2 = get_signal(id_signal)[4]
    i3 = get_signal(id_signal)[5]

    # Prepare the input array for MMC encoding (use the first N*nb_w samples for each phase)
    x = [
        v1[:N * nb_w], v2[:N * nb_w], v3[:N * nb_w],
        i1[:N * nb_w], i2[:N * nb_w], i3[:N * nb_w]
    ]

    # Process each phase of the signal
    for id_phase in range(nb_phase):
        # Store the original signal for later comparison
        x_real[id_signal][id_phase] = x[id_phase]
        
        # Process each window within the current phase
        
        
        for w in range(nb_w):
            quality_test=quality
            # Extract the current window of samples
            x_test = x[id_phase][w * N:(w + 1) * N]

            # Initialize variables
            RMSE_min = np.infty  # Minimal RMSE (initialized to infinity)

            # Normalize the signal and retrieve the scaling factor
            x_n, kx = normalize(x_test)
            
            if metric=="RMSE" :
                quality_test *=2**(-kx)  
            elif metric=="MSE":
                quality_test *=2**(-2*kx)  
                
                
            # Start searching for the number of significant vectors
            L = []  # List of selected dictionary indices
            L_diff = []  # Differences in positions for Golomb coding
            r = x_n  # Residual signal to be encoded
            gamma = []  # Amplitudes of the selected vectors
     

            # Select the best vectors to approximate the residual signal
            L_i, gamma_i = O.best_vect(r, D,metric,quality_test,np.infty)
          
            # Append the new vectors and amplitudes
            L.extend(L_i)
            gamma.extend(gamma_i)
            
            # Reconstruct the signal using the selected vectors and amplitudes
            x_rec_test = O.get_x_rec(D, gamma, L)
            
            # Update the residual signal
            r = x_n - x_rec_test

            # Prepare the list of selected indices for Golomb coding
            L_test = np.array(L) + 1  # Add 1 so all indices are nonzero

            # Apply differential coding if polarity is enabled
            if polarity:
                # Differential encoding of positions
                L_diff = [L_test[0]]
                for k in range(len(L_test) - 1):
                    L_diff.append(L_test[k+1] - L_test[k])
                    
                # Encode positions using Golomb coding
                binarisation_L, code_L, R_opt_binarisation, R1_opt_binarisation, R_opt_data = \
                    G_enc.golomb_encoding(L_diff, dico_values, n_value_max, polarity)
                
                # Decode the Golomb-coded positions
                L_diff_dec = G_dec.golomb_decoding(code_L, dico_values, len(L_test), n_value_max, polarity)
                
                # Reconstruct original indices from differential coding
                L_dec = [L_diff_dec[0]]
                for k in range(len(L_test) - 1):
                    L_dec.append(L_diff_dec[k+1] + L_dec[k])
                    
            else:
                # Encode positions directly (without differential coding)
                binarisation_L, code_L, R_opt_binarisation, R1_opt_binarisation, R_opt_data = \
                    G_enc.golomb_encoding(list(L_test), dico_values, n_value_max, polarity)
                
                # Decode the Golomb-coded positions
                L_dec = G_dec.golomb_decoding(code_L, dico_values, len(L_test), n_value_max, polarity)
            
            # Adjust indices back to zero-based
            L_test -= 1
            L_dec = np.array(L_dec) - 1
            
            # Check if decoding matched the original indices
            if np.mean(L_test) != np.mean(L_dec):
                print("Error: Decoded positions do not match the original")
                print("L_test    ", list(L_test))
                print("L_dec", list(L_dec))
                print("L_diff    ", list(L_diff))
                print("L_diff_dec", list(L_diff_dec))                
            
       
            

            ### Encode Gamma ###
            # Iterate over possible quantization levels
            for n_gamma in range(2, 2**nb_n_gamma):
                # Perform quantization on the gamma coefficients
                code_gamma, gamma_q_enc = Gamma_enc.quantization_gamma(gamma, n_gamma, N)
                gamma_q_dec = Gamma_dec.inv_quantization_gamma(code_gamma, n_gamma, len(gamma), N)
                
                # Reconstruct the signal using the quantized gamma coefficients and the selected dictionary indices
                x_rec_test = O.get_x_rec(D, gamma_q_dec, L)
                
                # Compute RMSE and SNR for the reconstructed signal
                
                quality_test_q=get_quality(x_test,x_rec_test*2**kx,metric)
                
                
                # Compute the total number of bits used
                R_real = len(code_gamma) + len(code_L)
       

                # Update the minimum RMSE and associated parameters if this configuration is better
                if  quality_test_q<quality :
                    R_min = R_real + R_h[id_signal][id_phase][w]
                   
                    # Store the current best configuration's performance metrics
                    
                    SNR[id_signal][id_phase][w] = -get_quality(x_test,x_rec_test*2**kx,"SNR")
                    RMSE[id_signal][id_phase][w] = get_quality(x_test,x_rec_test*2**kx,"RMSE")
                    
                    # Record the bits used for gamma coefficients and positions
                    R_gamma[id_signal][id_phase][w] = len(code_gamma)
                    R_pos[id_signal][id_phase][w] = len(code_L)
                    
                    # Calculate the unused bits and save the reconstructed signal
                
                    x_rec[id_signal][id_phase][w * N:w * N + N] = x_rec_test * 2**kx
                    
                    break
                        
         
            # Print the current encoding results for diagnostics
            print("id={}, phase={}, w={}, nb vect={},  Ntot={:.0f}, R_gamma={:.1f} b/coef, R_pos={:.1f} b/coef, RMSE={:.1f} V, SNR={:.1f} dB".format(
                id_signal, id_phase, w, len(L_test),
                R_gamma[id_signal][id_phase][w] + R_pos[id_signal][id_phase][w] + R_h[id_signal][id_phase][w],
                R_gamma[id_signal][id_phase][w] / len(L),
                R_pos[id_signal][id_phase][w] / len(L),
                RMSE[id_signal][id_phase][w],
                SNR[id_signal][id_phase][w]
            ))
 

#Record the end time
end_time = time.time()

#Calculate the elapsed time
elapsed_time = end_time - start_time

print("time to encode {:.0f} signal(s) ({} phase(s)) of {:.2f} seconde(s): {:.2f} secondes".format(nb_signal,nb_phase,N*nb_w/fs,elapsed_time))    



t=np.linspace(0,(nb_w)*(N-1)*(1/fs),nb_w*N)

  
for id_signal in range(nb_signal):
    
    for id_phase in range(nb_phase):

        #### Reconstructed signal
        plt.figure(figsize=(10, 4), dpi=100)
        plt.plot(t, x_real[id_signal][id_phase] / 1000, lw=1, label='x (original)')
        plt.plot(t, x_rec[id_signal][id_phase] / 1000, lw=1, label="x_rec (reconstructed)")
        plt.xlabel('t (s)')
        plt.ylabel('Magnitude x10³')
        plt.legend()
        plt.title(f"Reconstructed Signal: {id_signal}, Phase: {id_phase + 1}, "
                  f"SNR Mean = {np.mean(SNR[id_signal][id_phase]):.2f} dB, "
                  f"RMSE Mean = {np.mean(RMSE[id_signal][id_phase]):.2f} V")
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
    
        #### Reconstruction error
        plt.figure(figsize=(10, 4), dpi=100)
        plt.plot(t, (x_real[id_signal][id_phase] - x_rec[id_signal][id_phase]), lw=1, label='x - x_rec (error)')
        plt.xlabel('t (s)')
        plt.ylabel('Magnitude')
        plt.title(f"Reconstruction Error: {id_signal}, Phase: {id_phase + 1}")
        plt.legend()
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()       
        
        largeur_barre=0.6
        plt.figure(figsize=(8,4), dpi=80)
        plt.bar([i for i in range(nb_w)],SNR[id_signal][id_phase], width = largeur_barre,color='b',label="SNR")
        plt.xlabel('index')
        plt.ylabel('SNR (dB)')
        plt.legend()
        plt.title('SNR mean={:.2f} dB'.format(np.mean(SNR[id_signal][id_phase])))
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)  
        
        #### RMSE for each window
        plt.figure(figsize=(10, 4), dpi=100)
        plt.plot([t[k] for k in range(0, nb_w * N, N)], RMSE[id_signal][id_phase], '-o', lw=1, label='RMSE')
        plt.xlabel('t (s)')
        plt.ylabel('RMSE ')
        plt.title(f"RMSE for Each Window, Mean RMSE = {np.mean(RMSE[id_signal][id_phase]):.0f} V, "
                  f"Signal: {id_signal}, Phase: {id_phase + 1}")
        plt.legend()
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.show()
        
        plt.figure(figsize=(8,4), dpi=80)
        plt.bar([i for i in range(nb_w)],R_h[id_signal][id_phase], width = largeur_barre,color='g',label="R h")
        plt.bar([i for i in range(nb_w)],R_gamma[id_signal][id_phase], width = largeur_barre,bottom =R_h[id_signal][id_phase],color='b',label="R gamma")
        plt.bar([i for i in range(nb_w)],R_pos[id_signal][id_phase], width = largeur_barre,bottom =R_gamma[id_signal][id_phase]+R_h[id_signal][id_phase],color='r',label="R_pos")
        plt.xlabel('index w')
        plt.ylabel('R (bits)')
        plt.legend()
        plt.title('R mean={:.2f} bits'.format(np.mean(R_gamma[id_signal][id_phase]+R_pos[id_signal][id_phase]+R_h[id_signal][id_phase])))
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)    
        
        
        
