# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:51:12 2024

@author: presvotscor
"""
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import copy
from Measures import get_quality

# Define the Orthogonal Matching Pursuit (OMP) class
class OMP():
    def init__(self):
        # Initialization method (currently not used)
        pass

    def corr(self, vect1, vect2):
        """
        Calculate the absolute correlation coefficient between two vectors.

        Parameters:
        - vect1: The first vector (e.g., the input signal).
        - vect2: The second vector (e.g., a dictionary atom).

        Returns:
        - coefs: The absolute correlation coefficient.
        """
        coefs = np.abs(np.dot(vect1.T, vect2))
        return coefs

    def get_one_best_vect(self, x, D, L):
        """
        Identify the best matching vector from a dictionary that maximizes correlation
        with the input signal x, excluding vectors already selected (listed in L).

        Parameters:
        - x: Input signal.
        - D: Dictionary of possible basis vectors.
        - L: List of indices of already selected dictionary vectors.

        Returns:
        - id_best: Index of the dictionary vector that best matches x.
        """
        N = len(x)  # Number of samples in the signal x
        size_D = np.size(D, 1)  # Number of vectors in the dictionary
        
        x = np.array(x).reshape(N, 1)  # Ensure x is a column vector
        corr_max = 0  # Initialize maximum correlation
        
        for d in range(size_D):
            if not d in L:  # Skip vectors already in L
                test_corr = self.corr(x, D[:, d])  # Compute correlation
                if test_corr > corr_max:  # Update maximum correlation and index
                    corr_max = test_corr
                    id_best = d
        return id_best

    def get_gamma(self, x, D, L):
        """
        Compute the coefficients (gamma) for reconstructing x using the selected
        dictionary vectors indexed by L.

        Parameters:
        - x: Input signal.
        - D: Dictionary of possible basis vectors.
        - L: List of indices of selected dictionary vectors.

        Returns:
        - gamma: Coefficients for the selected dictionary vectors.
        """
        gamma = np.dot(
            np.linalg.inv(np.dot(D[:, L].T, D[:, L])),  # Inverse of the Gram matrix
            np.dot(D[:, L].T, x)  # Projection of x onto the selected dictionary vectors
        )
        return gamma

    def get_x_rec(self, D, gamma, L):
        """
        Reconstruct the signal using the selected dictionary vectors and their coefficients.

        Parameters:
        - D: Dictionary of possible basis vectors.
        - gamma: Coefficients for the selected dictionary vectors.
        - L: List of indices of selected dictionary vectors.

        Returns:
        - Reconstructed signal.
        """
        return np.dot(D[:, L], gamma)

            
    
    def best_vect(self, x, D, metric,quality, nb_vect_max):
        """
        Select the most significant vectors from a dictionary using iterative residual minimization.
        
        Parameters:
        - x: Input signal.
        - D: Dictionary of potential basis vectors.
        - metric: A metric function to evaluate the quality of the approximation.
        - quality: The target quality threshold for the approximation (e.g., a maximum acceptable RMSE).
        - nb_vect_max: Maximum number of vectors to select.
        
        Returns:
        - L_tri: Sorted indices of the selected significant vectors.
        - gamma_tri: Sorted coefficients corresponding to the selected vectors.
        """
        # Step 1: Initialization
        N = len(x)  # Number of samples in the signal
        x_copy = copy.copy(x).reshape(N, 1)  # Create a copy of the input signal
        
        L = []       # List to store indices of significant vectors
        gamma = []   # List to store amplitudes of the significant vectors
        r = x_copy   # Initial residual is the input signal
        quality_test = np.inf  # Start with an infinite quality to ensure the loop runs at least once
        
        # Step 2: Iterative selection of significant vectors
        while quality_test > quality and len(L) < nb_vect_max:
            # Identify the next best vector from the dictionary that minimizes the residual
            id_best = self.get_one_best_vect(r, D, L)
            # Add the best vector's index to the list of significant vectors
            L.append(id_best)
            # Compute the coefficients (gamma) for the selected vectors
            gamma = self.get_gamma(x_copy, D, L)
            # Reconstruct the signal from the selected vectors and their coefficients
            x_rec = self.get_x_rec(D, gamma.reshape((len(gamma), 1)), L)
            # Calculate the current RMSE between the original signal and the reconstruction
            quality_test = get_quality(x, x_rec.reshape(N),metric)
            # Update the residual
            r = x_copy - x_rec
        
        # Step 3: Sort the selected vectors and their coefficients
        gamma = gamma.reshape(len(gamma))
        ind_gamma_tri = np.flip(np.argsort(np.abs(gamma)))  # Sort indices by coefficient magnitude in descending order
        gamma_tri = np.array(gamma)[ind_gamma_tri]  # Sort coefficients
        L_tri = np.array(L)[ind_gamma_tri]          # Sort indices of significant vectors
        
        return L_tri, gamma_tri


if __name__ == "__main__":
    from Normalize import normalize  # Import normalization function
    from Dico import Dico_Cos_Sin_Dirac  # Import dictionary creation class
    from get_test_signal import get_RTE_signal as get_signal  # Import signal retrieval function

    # General parameters
    fs = 6400  # Sampling frequency in Hz
    fn = 50    # Nominal frequency in Hz
    N = 128    # Number of samples per window

    nb_vect = 200 # Maximum number of significant vectors to select

    # Time vector for plotting
    t = np.linspace(0, (N - 1) * (1 / fs), N)

    # Create an instance of the OMP class
    O = OMP()

    # Dictionary parameters
    f_max = fs/2  # Maximum frequency considered in the dictionary
    f_factor = 1     # Frequency factor for dictionary creation

    # Select a specific signal, phase, and window for processing
    id_signal = 1
    id_phase = 0
    w = 18

    # Generate the dictionary
    Dico = Dico_Cos_Sin_Dirac(fs, f_factor)
    D = Dico.Creat_dico(N, f_max)
    size_D = np.size(D, 1)

    # Retrieve the selected portion of the signal
    x = get_signal(id_signal)[id_phase][w * N:w * N + N]

    # Normalize the signal
    x_n, kx = normalize(x)

    # Set the target RMSE threshold for vector selection
    metric="SNR"
    quality = -30#200**2 # Voltage threshold after normalization
    
    if metric=="RMSE" :
        quality *=2**(-kx)  
    elif metric=="MSE":
        quality *=2**(-2*kx)  
        

    # Perform vector selection using OMP
    L, gamma = O.best_vect(x_n, D,metric,quality, nb_vect)

    # Reconstruct the signal from the selected vectors
    x_rec = O.get_x_rec(D, gamma, L)

    # Calculate reconstruction quality
    RMSE = get_quality(x_n, x_rec,"RMSE")  # Root Mean Square Error
    SNR = -get_quality(x_n, x_rec,"SNR")    # Signal-to-Noise Ratio

    # Plot the original and reconstructed signals
    plt.figure(figsize=(8, 4), dpi=80)
    plt.plot(t, x, lw=2, label='x')  # Original signal
    plt.plot(t, x_rec * 2**kx, lw=2, label='x_rec')  # Reconstructed signal
    plt.xlabel('t (s)')
    plt.ylabel("Voltage (kV)")
    plt.title(f'SNR={SNR:.1f} dB, RMSE={RMSE * 2**(kx):.2f}, nb vect={len(L)}')
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)

    # Plot the selected vector positions
    p_pos = np.zeros(size_D)
    for k in range(len(L)):
        p_pos[L[k]] = 1
    plt.figure(figsize=(8, 4), dpi=80)
    plt.plot(p_pos, '-*', lw=2, label='p position')
    plt.xlabel('Index')
    plt.ylabel("Presence (bool)")
    plt.title(f'List of positions, number of indices={len(L)}')
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)

    # Plot the selected vector amplitudes
    plt.figure(figsize=(8, 4), dpi=80)
    plt.plot(np.abs(gamma), '-*', lw=2, label='gamma')
    plt.xlabel('Index')
    plt.ylabel("Magnitude")
    plt.title(f'Amplitudes of selected vectors, number of indices={len(L)}')
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)

    
    #%% statistic of apparition source
    #"""    
    # Import functions for calculating frequency distribution and entropy of vector appearances
    from Measures import get_frequency, entropy
    
    # Configuration: Number of signals, phases, and windows to process, as well as the number of vectors to consider
    nb_signal = 1
    nb_phase = 1
    nb_w = 50
    nb_vect = 20
    
    # Initialize lists to store results
    L_p = []  # Indices of selected dictionary vectors across all windows
    L_gamma = []  # Amplitudes (gamma coefficients) of selected vectors
    L_SNR = []  # Signal-to-Noise Ratios (SNR) of the reconstructed windows
    
    # Process each signal, phase, and window
    for id_signal in range(nb_signal):
        for id_phase in range(nb_phase):
            for w in range(20, nb_w):
                # Extract the signal segment corresponding to the current window
                x = get_signal(id_signal)[id_phase][w * N : w * N + N]
                
                # Normalize the signal and determine the maximum allowable reconstruction error
                x_n, kx = normalize(x)
                RMSE_max = 200 * 2**(-kx)  # Adjust RMSE threshold based on scaling factor
                
                # Use a best-vector selection method (e.g., Orthogonal Matching Pursuit) to find significant vectors
                L, gamma = O.best_vect(x_n, D, "MSE",0, nb_vect)
                
                # Collect the indices and amplitudes of selected vectors
                L_p.extend(L)
                L_gamma.append(gamma)
                
                # Reconstruct the signal using the selected vectors and record the SNR
                x_rec = O.get_x_rec(D, gamma, L)
                L_SNR.append(-get_quality(x_n, x_rec,"SNR"))
    
    # Calculate the frequency of vector indices across all processed windows
    positions, p = get_frequency(L_p)
    
    # Create a zero-initialized array representing all possible dictionary positions
    p_pos = np.zeros(size_D)
    
    # Assign the probabilities to the corresponding indices
    for k in range(len(positions)):
        p_pos[positions[k]] = p[k]
    
    # Plot the probability of each dictionary position
    plt.figure(figsize=(8, 4), dpi=80)
    plt.plot(p_pos, lw=2, label='Position probabilities')
    plt.xlabel('Index')
    plt.ylabel('Probability')
    plt.title('Probability of positions over {} windows (Mean SNR={:.1f} V, Entropy={:.3f} bits)'.format(
        len(L_p), np.mean(L_SNR), entropy(p) * nb_vect))
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    
    # Plot the average magnitude of the selected gamma coefficients
    plt.figure(figsize=(8, 4), dpi=80)
    plt.plot(np.mean(np.abs(L_gamma), axis=0), lw=2, label='Mean gamma magnitudes')
    plt.xlabel('Index')
    plt.ylabel('Magnitude')
    plt.title('Mean gamma magnitudes over {} windows (Mean SNR={:.1f} V)'.format(len(L_p), np.mean(L_SNR)))
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)

    #"""
    