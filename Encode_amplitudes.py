# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:10:31 2024

@author: presvotscor
"""
import numpy as np
from Quantization import Quantizer  # Import a helper class for quantization operations

# Class to handle encoding of gamma coefficients
class encoded_gamma():
    def __init__(self):
        # Initialize the encoded_gamma object (no attributes used at initialization)
        pass

    # Perform quantization on gamma coefficients
    def quantization_gamma(self, gamma, n_gamma, N):
        """
        Quantize a list of gamma coefficients and produce encoded output.

        Parameters:
        - gamma: The list of coefficients to be quantized.
        - n_gamma: The initial number of bits available for quantization.
        - N: The size of the signal.

        Returns:
        - code: The list of quantized and encoded values as binary data.
        - gamma_q: The quantized values of gamma.
        """
        
        # Arrays to store quantized indices and quantized gamma values
        ind_gamma_q = np.zeros(len(gamma))
        gamma_q = np.zeros(len(gamma))
        
        # Set the initial quantization range (w_test) based on the signal size N
        w_test = 2 * np.sqrt(N)
        n_gamma_test = n_gamma
        
        code = []  # List to hold encoded binary data
        
        for k in range(len(gamma)):
            # Quantize the current gamma coefficient and get its index
            ind_gamma_q[k] = Quantizer().get_ind_u(gamma[k], n_gamma_test, w_test, 0)
            
            # Convert the quantized index to binary code and append it
            code.extend(Quantizer().get_code_u(ind_gamma_q[k], n_gamma_test))
            
            # Get the quantized gamma value from the index
            gamma_q[k] = Quantizer().get_q_u(ind_gamma_q[k], n_gamma_test, w_test, 0)
            
            # Dynamically adjust the number of bits used for subsequent quantizations
            j = 1
            while 2 * np.abs(gamma_q[k]) * 2**j < w_test and n_gamma_test - j + 1 > 0:
                j += 1
            
            # Reduce the available bit budget and quantization range for the next coefficient
            n_gamma_test -= (j - 1)
            w_test /= 2**(j - 1)
        
        # Return the encoded binary data and the quantized coefficients
        return code, gamma_q

# Class to handle decoding of gamma coefficients
class decoded_gamma(): 
    def __init__(self):
        # Initialize the decoded_gamma object (no attributes used at initialization)
        pass

    # Perform inverse quantization on encoded gamma coefficients
    def inv_quantization_gamma(self, code, n_gamma, nb_gamma, N):
        """
        Decode a binary-encoded sequence of gamma coefficients.

        Parameters:
        - code: The binary data to be decoded.
        - n_gamma: The initial number of bits used for quantization.
        - nb_gamma: The number of gamma coefficients to decode.
        - N: The size of the signal.

        Returns:
        - gamma_q: The reconstructed quantized gamma coefficients.
        """
        
        ind_gamma_q = []  # List to hold decoded indices
        gamma_q = []      # List to hold the reconstructed gamma values
        
        # Set the initial quantization range (w_test) based on the signal size N
        w_test = 2 * np.sqrt(N)
        n_gamma_test = n_gamma
        ptr = 0  # Pointer to track the current position in the binary code
        
        # Continue decoding until the desired number of coefficients is reached
        while len(gamma_q) < nb_gamma:
            # Decode the next quantized index from the binary code
            ind_gamma_q.append(Quantizer().get_inv_code_u(code[ptr:ptr + n_gamma_test], n_gamma_test))
            ptr += n_gamma_test
            
            # Reconstruct the quantized gamma value from the decoded index
            gamma_q.append(Quantizer().get_q_u(ind_gamma_q[-1], n_gamma_test, w_test, 0))
            
            # Dynamically adjust the number of bits and quantization range for the next coefficient
            j = 1
            while 2 * np.abs(gamma_q[-1]) * 2**j < w_test and n_gamma_test - j + 1 > 0:
                j += 1
            
            # Reduce the available bit budget and quantization range for the next coefficient
            n_gamma_test -= (j - 1)
            w_test /= 2**(j - 1)
        
        # Return the list of reconstructed gamma values
        return gamma_q

if __name__ == "__main__":
    # Import required modules and functions
    from Normalize import normalize
    import matplotlib.pyplot as plt
    from Dico import Dico_Cos_Sin_Dirac
    from OMP import OMP
    from get_test_signal import get_RTE_signal as get_signal  # Retrieve test signals
    from Measures import get_rmse, get_snr  # Import signal quality measurement functions
 
    # General parameters
    fs = 6400  # Sampling frequency in Hz
    fn = 50    # Nominal frequency in Hz

    # Signal, phase, and window selection
    id_signal = 1  # ID of the signal to process
    id_phase = 0   # Phase index to analyze
    w = 18         # Window index to use
    N = 128        # Size of each window

    # Retrieve the selected segment of the signal
    x = get_signal(id_signal)[id_phase][w * N:w * N + N]

    # Normalize the signal and obtain the scaling factor
    x_n, kx = normalize(x)
    
    # Initialize the OMP (Orthogonal Matching Pursuit) object
    O = OMP()
    
    # Define dictionary parameters
    f_max = fs / 2  # Maximum frequency in the dictionary (Nyquist frequency)
    f_factor = 1  # Frequency scaling factor
    nb_vect_max = np.infty  # Allow an unlimited number of vectors in OMP
    
    # Create a dictionary of cosine, sine, and Dirac basis vectors
    Dico = Dico_Cos_Sin_Dirac(fs, f_factor)
    D = Dico.Creat_dico(N, f_max)  # Generate the dictionary with specified parameters
    size_D = np.size(D, 1)  # Number of vectors in the dictionary
    
    # Set the RMSE threshold for OMP reconstruction
    RMSE_max = 200 * 2**(-kx)  # Set the maximum acceptable root-mean-square error (adjusted by normalization factor)
    
    # Perform OMP-based sparse coding to find the best set of dictionary vectors (L) and their coefficients (gamma)
    L, gamma = O.best_vect(x_n, D, RMSE_max, nb_vect_max)
    
    # Reconstruct the signal from the selected vectors and their coefficients
    x_rec = O.get_x_rec(D, gamma, L) * 2**kx  # Multiply by normalization factor to return to original scale
    
    # Quantization and decoding parameters
    n_gamma = 14  # Initial number of bits for gamma quantization
    Gamma_enc = encoded_gamma()  # Instantiate the gamma encoder
    Gamma_dec = decoded_gamma()  # Instantiate the gamma decoder
    
    # Quantize the gamma coefficients
    code, gamma_q_enc = Gamma_enc.quantization_gamma(gamma, n_gamma, N)
    # Decode the quantized gamma coefficients
    gamma_q_dec = Gamma_dec.inv_quantization_gamma(code, n_gamma, len(gamma), N)
    
    # Reconstruct the signal using the quantized gamma coefficients
    x_rec_q = O.get_x_rec(D, gamma_q_dec, L) * 2**kx  # Scale back to original range
    
    # Plot the original and quantized gamma values (linear scale)
    plt.figure(figsize=(8, 4), dpi=80)
    plt.plot(np.abs(gamma), lw=2, label='gamma')
    plt.plot(np.abs(gamma_q_dec), lw=2, label='gamma q dec')
    plt.xlabel('index')
    plt.ylabel("Gamma")
    plt.title('gamma vectors={}, n_gamma={}, Rgamma={} b'.format(len(L), n_gamma, len(code)))
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    
    # Plot the original and quantized gamma values (log scale)
    plt.figure(figsize=(8, 4), dpi=80)
    plt.plot(np.abs(gamma), lw=2, label='gamma')
    plt.plot(np.abs(gamma_q_dec), lw=2, label='gamma q dec')
    plt.xlabel('index')
    plt.ylabel("Gamma")
    plt.title('gamma vectors={}, n_gamma={}, Rgamma={} b'.format(len(L), n_gamma, len(code)))
    plt.legend()
    plt.yscale('log')
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    
    # Plot the error between the original and quantized gamma values
    plt.figure(figsize=(8, 4), dpi=80)
    plt.plot(gamma - gamma_q_dec, lw=2, label='gamma - gamma q_dec')
    plt.xlabel('index')
    plt.ylabel("error")
    plt.title('error gamma gamma_q vectors={}, n_gamma={}, Rgamma={} b'.format(len(L), n_gamma, len(code)))
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)

    if np.mean(gamma_q_enc)!=np.mean(gamma_q_dec):
        print("false")
    else: 
        print("True")
    x_rec_opt=O.get_x_rec(D, gamma, L)*2**kx
    x_rec_enc=O.get_x_rec(D, gamma_q_enc, L)*2**kx
    x_rec_dec=O.get_x_rec(D, gamma_q_dec, L)*2**kx
    
    print("RMSE={:.2f} V, RMSE_enc={:.2f} V, RMSE_dec={:.2f} V, n_gamma={}, R={} b".format(get_rmse(x,x_rec_opt),get_rmse(x,x_rec_enc),get_rmse(x,x_rec_dec),n_gamma,len(code)))   
    
    plt.figure(figsize=(8,4), dpi=80)
    plt.plot(x,lw=2,label='x')
    plt.plot(x_rec,lw=2,label='x_rec, SNR={:.1f} dB, RMSE={:.0f} V'.format(get_snr(x, x_rec),get_rmse(x, x_rec)))
    plt.plot(x_rec_q,lw=2,label='x_rec_q, SNR={:.1f} dB, RMSE={:.0f} V'.format(get_snr(x, x_rec_q),get_rmse(x, x_rec_q)))
    plt.xlabel('ind')
    plt.ylabel("error")
    plt.title('Reconstructed signal, vects={}, n_gamma={}, Rgamma={} b'.format(len(L),n_gamma,len(code)))
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2) 

