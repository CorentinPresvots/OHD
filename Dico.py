# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:46:19 2024

@author: presvotscor
"""
#  create dico
import numpy as np

class Dico_Cos_Sin_Dirac():
    def __init__(self, fs, f_factor):
        """
        Initialize the dictionary generator with sampling frequency and frequency scaling factor.

        Parameters:
        - fs: Sampling frequency of the signal.
        - f_factor: Scaling factor for the frequency resolution in the dictionary.
        """
        self.f_factor = f_factor
        self.fs = fs
        

    def Creat_dico(self, N, f_max):
        """
        Create a hybrid dictionary consisting of impulse, cosine, and sine components.

        Parameters:
        - N: Size of the dictionary (number of samples per signal window).
        - f_max: Maximum frequency (in Hz) to include in the cosine and sine basis functions.

        Returns:
        - D: A matrix where each column is a basis vector (impulse, cosine, or sine component).
        """
        # Calculate the number of frequency-based vectors (cosines and sines)
        nb_vect_freq = int(self.f_factor * f_max * 2 * N / self.fs)
        
        # Initialize the impulse dictionary
        
       
        # Construct the impulse dictionary
  
        I = np.zeros((N, N))  # Initialize the impulse dictionary
        for raw in range(0, N):
            for col in range(0, int(N / 2)):
                if raw == col:
                    I[raw, 2 * col] = 1
                    I[N - raw - 1, 2 * col + 1] = 1
    
        # or
        #I = np.eye(N)
        
        # Initialize the matrix to hold cosine and sine basis vectors
        F = np.zeros((N, 2 * nb_vect_freq))
                

        # Construct the cosine and sine dictionary
        for raw in range(0, N):
            for col in range(0, nb_vect_freq):
                # Cosine basis vectors
                F[raw, 2 * col] = np.sqrt(2 / N) * np.cos((np.pi / N) * (raw + 1 / 2) * (col) / self.f_factor)
                # Sine basis vectors
                F[raw, 2 * col+1] = np.sqrt(2 / N) * np.sin((np.pi / N) * (raw + 1 / 2) * ((col + 1) / self.f_factor))
               
                
        # Concatenate all basis vectors into a single dictionary
        D = np.concatenate((F, I), axis=1)
        return D

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N = 128  # Number of samples per window
    
    f_factor = 1  # Frequency scaling factor for the dictionary
    fs = 6400       # Sampling frequency (Hz)
    f_max = fs / 2     # Maximum frequency in the dictionary (Hz)
    
    # Initialize the dictionary generator
    Dico = Dico_Cos_Sin_Dirac(fs, f_factor)
    
    # Create the hybrid dictionary (constant, cosines, sines, and impulses)
    D = Dico.Creat_dico(N, f_max)
    size_D = np.size(D, 1)  # Total number of dictionary vectors
    
    print("Shape of D:", D.shape, size_D)
    
    # Count the number of significantly correlated vector pairs
    nb_element_correlated = 0
    total_test = 0
    for i in range(size_D):
        for j in range(i, size_D):
            if i != j:
                total_test += 1
                # Check if the correlation between two vectors exceeds 0.1
                if np.abs(np.dot(D[:, i].T, D[:, j])) > 0.1:
                    nb_element_correlated += 1
    
    # Display the percentage of correlated vectors in the dictionary
    print("% of correlated vectors in D: {:.1f}%".format(100 * nb_element_correlated / total_test))
    
    # Generate a time vector for plotting
    t = np.linspace(0, N / fs - 1 / fs, N)
    
    # Plot the constant vector
    fig = plt.figure(figsize=(8, 4), dpi=70)
    plt.plot(t, D[:, 0], lw=2, label="Constant")
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title("Visualization of the constant vector")
    plt.legend()
    plt.grid(which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)

    # Plot the cosine and sine vectors
    for j in range(1, int((size_D - N) / 2) + 1,int(N/4)):
        fig = plt.figure(figsize=(8, 4), dpi=70)
        plt.plot(t, D[:, 2 * j - 1], lw=2, label="Cosine {} Hz".format(int(j * fs / (2 * N * f_factor))))
        plt.plot(t, D[:, 2 * j], lw=2, label="Sine {} Hz".format(int(j * fs / (2 * N * f_factor))))
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title("Visualization of the {}th cosine and sine, N={}, f_max={}, fs={} Hz".format(j, N, f_max, fs))
        plt.legend()
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)

    # Plot the Dirac vectors
    for j in range(size_D - N, size_D - N + 10):
        fig = plt.figure(figsize=(8, 4), dpi=70)
        plt.plot(t, D[:, j], lw=2, label="Dirac {}".format(size_D - N - j))
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title("Visualization of the {}th Dirac vector, N={}".format(size_D - N - j, N))
        plt.legend()
        plt.grid(which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
