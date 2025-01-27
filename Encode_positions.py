# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:14:05 2024

@author: presvotscor
"""

# %% Position Coding
from Context_Arithmetic import Context_Aritmetic_Encoder, Context_Aritmetic_Decoder
from itertools import accumulate
from Measures import calculate_conditional_entropy, calculate_entropy, get_frequency, entropy
import numpy as np
import math as mt

class Golomb():
    def __init__(self):
        pass
    
    # Fixed-length encoding based on a reference paper "Denoising DCT"
    def fix_enc(self, ind, b):
        # `b` determines the threshold `T` for switching encoding patterns
        T = 2**b - 1
        
        L = []  # List to store encoded values
        
        if np.abs(ind) > T:
            # If the value exceeds the threshold, divide it by `T` and store the quotient as a series of zeros
            Nz = np.floor((np.abs(ind) - 1) / T)
            L.extend([0] * int(Nz))
            
            # Append the remainder after dividing by `T`
            L.append(int(ind - Nz * T))
        else:
            # For values within the threshold, simply append them
            L.append(int(ind))
            
        # No return value for `nb_bit` here, it is left as a comment in the original code
        return L

    def fix_dec(self, code, b):
        # Decoding logic for the fixed-length encoding
        T = 2**b - 1
        
        # Count leading zeros to determine the number of times `T` was added
        Nz = 0
        while code[Nz] == 0:
            Nz += 1
            
        # Calculate the decoded index using the remainder after the zeros
        return Nz * T + code[-1]

    # Golomb-Rice coding implementation
    def golomb_cod(self, x, m):
        """
        Implements Golomb-Rice coding:
        - `m` is the divisor parameter controlling the code length.
        - `x` is the value to be encoded.
        """
        # Calculate the binary length `c` based on `m`
        c = int(mt.ceil(mt.log(m, 2)))
        remin = x % m  # Remainder
        quo = int(mt.floor(x / m))  # Quotient
        div = int(mt.pow(2, c) - m)

        # Initialise the bit sequence
        bits = []

        # Append `1` bits for each quotient value
        for _ in range(quo):
            bits.append(1)

        # Append a `0` bit to mark the end of the quotient section
        if m != 1:
            bits.append(0)

        # Determine the binary representation for the remainder
        if remin < div:
            b = c - 1
            a = "{0:0" + str(b) + "b}"
            bi = a.format(remin)
        else:
            b = c
            a = "{0:0" + str(b) + "b}"
            bi = a.format(remin + div)

        # Convert binary string to individual bits and append
        bits.extend(int(bit) for bit in bi)

        return bits

    # Generates a Golomb-Rice code dictionary
    # `m` controls the coding parameter
    # `size_dico` specifies the size of the dictionary
    # `pos_E` specifies a special symbol "E" in the dictionary
    def creat_dico(self, m, size_dico, pos_E):
        dico = {}

        sym = 0  # Symbol index
        cpt = 0  # Code index
        while sym <= size_dico + 1:
            if sym == pos_E and cpt == sym:
                # Assign a unique Golomb code for the special "E" symbol
                dico['E'] = self.golomb_cod(cpt, m)
                cpt += 1
            else:
                # Assign a Golomb code for regular symbols
                dico[sym] = self.golomb_cod(cpt, m)
                sym += 1
                cpt += 1

        return dico


class Golomb_encoding_pos(Golomb, Context_Aritmetic_Encoder):
    def __init__(self, adaptive):
        # Initialize the Golomb coding and arithmetic encoding components
        Golomb.__init__(self)
        
        M = 12  # Precision parameter for arithmetic encoding
        Context_Aritmetic_Encoder.__init__(self, M)
        
        # Store whether the encoding should adapt probabilities dynamically
        self.adaptive = adaptive

    def reset_Golomb_encoding_pos(self):
        # Reset the arithmetic encoding state and reinitialize probability tables
        self.reset_Context_Aritmetic_Encoder()
        
        # Initialize the alphabet and probability tables for conditional entropy coding
        self.alphabet_values_p0 = [0, 1]
        self.alphabet_values_p1 = [0, 1]
        
        # Initial occurrences (frequencies) for symbols following '0' and '1'
        self.occurrence_values_p0 = [1, 1]
        self.occurrence_values_p1 = [1, 1]
        
        # Alphabet and occurrences for polarity
        self.alphabet_p = [0, 1]  # 0 for '-', 1 for '+'
        self.occurrence_p = [1, 1]
        
        # Compute cumulative frequencies
        self.cumulate_occurrence_values_p0 = list(accumulate(self.occurrence_values_p0))
        self.cumulate_occurrence_values_p1 = list(accumulate(self.occurrence_values_p1))
        self.cumulate_occurrence_p = list(accumulate(self.occurrence_p))

    def encoded_value(self, code_value):
        # Perform entropy coding on the input sequence (code_value)
        code = []
        previous_bit = 1  # Start assuming the previous bit is 1

        # Process each bit in the input sequence
        for k in range(len(code_value)):
            if previous_bit == 0:
                if code_value[k] == 0:
                    # Encode a '0' after a previous '0'
                    code_value_k = self.encode_one_symbol(0, self.occurrence_values_p0, self.cumulate_occurrence_values_p0)
                    code.extend(code_value_k)
                    if self.adaptive:
                        self.occurrence_values_p0[0] += 1
                        self.cumulate_occurrence_values_p0 = list(accumulate(self.occurrence_values_p0))
                    previous_bit = 0
                elif code_value[k] == 1:
                    # Encode a '1' after a previous '0'
                    code_value_k = self.encode_one_symbol(1, self.occurrence_values_p0, self.cumulate_occurrence_values_p0)
                    code.extend(code_value_k)
                    if self.adaptive:
                        self.occurrence_values_p0[1] += 1
                        self.cumulate_occurrence_values_p0 = list(accumulate(self.occurrence_values_p0))
                    previous_bit = 1
            
            elif previous_bit == 1:
                if code_value[k] == 0:
                    # Encode a '0' after a previous '1'
                    code_value_k = self.encode_one_symbol(0, self.occurrence_values_p1, self.cumulate_occurrence_values_p1)
                    code.extend(code_value_k)
                    if self.adaptive:
                        self.occurrence_values_p1[0] += 1
                        self.cumulate_occurrence_values_p1 = list(accumulate(self.occurrence_values_p1))
                    previous_bit = 0
                elif code_value[k] == 1:
                    # Encode a '1' after a previous '1'
                    code_value_k = self.encode_one_symbol(1, self.occurrence_values_p1, self.cumulate_occurrence_values_p1)
                    code.extend(code_value_k)
                    if self.adaptive:
                        self.occurrence_values_p1[1] += 1
                        self.cumulate_occurrence_values_p1 = list(accumulate(self.occurrence_values_p1))
                    previous_bit = 1
        
        return code


    
    def encoded_polarity(self, p):  
        """
        Perform entropy coding on the polarity (sign) information.

        Parameters:
        - p: Polarity value (0 or 1) to encode.

        Returns:
        - code: The encoded polarity as a sequence of bits.
        """
        code = []
        if p == 0:
            # Encode a polarity of 0 using the current probability model
            code_p = self.encode_one_symbol(0, self.occurrence_p, self.cumulate_occurrence_p)
            code.extend(code_p)
            if self.adaptive:
                # Update probabilities if adaptive mode is enabled
                self.occurrence_p[0] += 1
                self.cumulate_occurrence_p = list(accumulate(self.occurrence_p))
        elif p == 1:
            # Encode a polarity of 1 using the current probability model
            code_p = self.encode_one_symbol(1, self.occurrence_p, self.cumulate_occurrence_p)
            code.extend(code_p)
            if self.adaptive:
                # Update probabilities if adaptive mode is enabled
                self.occurrence_p[1] += 1
                self.cumulate_occurrence_p = list(accumulate(self.occurrence_p))                   
        
        return code     
    
    def binarisation(self, data, n_value_max):
        """
        Convert a sequence of values into a list of binary components and polarity bits.

        Parameters:
        - data: The input sequence of values to binarize.
        - n_value_max: The maximum number of bits used to represent each value.

        Returns:
        - source: A list of tuples where each tuple contains:
            1. The binary representation of the value's magnitude.
            2. The sign of the value (0 for negative, 1 for positive).
        """
        source = []
        for k in range(len(data)):
            value = data[k]
            # Convert the absolute value to binary representation
            v = self.fix_enc(np.abs(value), n_value_max)
            # Store the binary representation and the sign bit
            source.append((v, int((np.sign(value) + 1) / 2)))
        
        return source

    def golomb_encoding(self, data, dico_value, n_value_max, polarity):
        """
        Perform Golomb coding on the input data, including encoding of magnitudes and optional polarity.

        Parameters:
        - data: Input sequence of values to encode.
        - dico_value: Dictionary of Golomb codes for each symbol.
        - n_value_max: Maximum bit length for value representation.
        - polarity: Whether to encode polarity separately.

        Returns:
        - binarisation: The raw binary representation before entropy coding.
        - code: The fully encoded sequence.
        - H: The entropy (in bits) of the encoded values.
        - H1: The conditional entropy (in bits) of the encoded values.
        - R_opt: The theoretical optimal rate based on entropy.
        """
        self.reset_Golomb_encoding_pos()
        # Convert values to binary form and polarity bits
        source_bin = self.binarisation(data, n_value_max)
        
        # Initialize lists to store intermediate results
        binarisation_value = []
        binarisation = []
        code = []
        nb_s = 0

        # Iterate over the source binary representation
        for value, sign in source_bin:
            # Encode the magnitude using the dictionary
            for k in range(len(value)):
                code_value = dico_value[value[k]]
                binarisation.extend(code_value)
                code.extend(self.encoded_value(code_value))
                binarisation_value.extend(code_value)
            
            # If polarity is enabled, encode it separately
            if polarity:
                binarisation.append(sign)
                code.extend(self.encoded_polarity(sign))
            
            nb_s += 1
        
        # Finalize arithmetic encoding
        code.extend(self.finish(self.l, self.follow))
        
        # Calculate entropy and other rate metrics
        _, proba_data = get_frequency(np.abs(data))
        R_opt = entropy(proba_data) * len(data) + len(data) - data.count(0.)
        H = calculate_entropy(binarisation_value) * len(binarisation_value) + nb_s
        H1 = calculate_conditional_entropy(binarisation_value) * len(binarisation_value) + nb_s

        return binarisation, code, H, H1, R_opt




class Golomb_decoding_pos(Golomb, Context_Aritmetic_Decoder):
    def __init__(self, adaptive):
        # Initialize the Golomb coding and arithmetic decoding components
        Golomb.__init__(self)
        
        M = 12  # Precision parameter for arithmetic decoding
        Context_Aritmetic_Decoder.__init__(self, M)
        
        # Store whether the decoding should adapt probabilities dynamically
        self.adaptive = adaptive

    def reset_Golomb_decoding_pos(self):
        # Reset the arithmetic decoding state and reinitialize probability tables
        self.reset_Context_Aritmetic_Decoder()
        
        # Initialize the alphabet and probability tables for conditional entropy decoding
        self.alphabet_values_p0 = [0, 1]
        self.alphabet_values_p1 = [0, 1]
        
        # Initial occurrences (frequencies) for symbols following '0' and '1'
        self.occurrence_values_p0 = [1, 1]
        self.occurrence_values_p1 = [1, 1]
        
        # Alphabet and occurrences for polarity
        self.alphabet_p = [0, 1]  # 0 for '-', 1 for '+'
        self.occurrence_p = [1, 1]
        
        # Compute cumulative frequencies
        self.cumulate_occurrence_values_p0 = list(accumulate(self.occurrence_values_p0))
        self.cumulate_occurrence_values_p1 = list(accumulate(self.occurrence_values_p1))
        self.cumulate_occurrence_p = list(accumulate(self.occurrence_p))

    def decoded_value(self, code, previous_bit):
        # Decode a single bit from the input sequence, using conditional probabilities
        if previous_bit == 0:
            bin_value = self.decode_one_symbol(code, self.alphabet_values_p0, self.occurrence_values_p0, self.cumulate_occurrence_values_p0)
            if self.adaptive:
                if bin_value == 0:
                    self.occurrence_values_p0[0] += 1
                elif bin_value == 1:
                    self.occurrence_values_p0[1] += 1
                self.cumulate_occurrence_values_p0 = list(accumulate(self.occurrence_values_p0))
        elif previous_bit == 1:
            bin_value = self.decode_one_symbol(code, self.alphabet_values_p1, self.occurrence_values_p1, self.cumulate_occurrence_values_p1)
            if self.adaptive:
                if bin_value == 0:
                    self.occurrence_values_p1[0] += 1
                elif bin_value == 1:
                    self.occurrence_values_p1[1] += 1
                self.cumulate_occurrence_values_p1 = list(accumulate(self.occurrence_values_p1))
        
        return bin_value

    def decoded_polarity(self, code):
        # Decode the polarity (sign) bit from the input sequence
        bin_p = self.decode_one_symbol(code, self.alphabet_p, self.occurrence_p, self.cumulate_occurrence_p)
        if self.adaptive:
            if bin_p == 0:
                self.occurrence_p[0] += 1
            elif bin_p == 1:
                self.occurrence_p[1] += 1
            self.cumulate_occurrence_p = list(accumulate(self.occurrence_p))
        return bin_p

    def golomb_decoding(self, code, dico_value, N, n_value_max, polarity):
        """
        Perform Golomb decoding on a given encoded sequence.
        
        Parameters:
        - code: The encoded bit sequence to be decoded.
        - dico_value: A dictionary mapping symbol values to their encoded representations.
        - N: The number of symbols to decode.
        - n_value_max: The maximum value of a symbol, used for decoding the fixed code.
        - polarity: A flag indicating whether polarity (sign) decoding is enabled.
        
        Returns:
        - sym: A list of decoded symbols.
        """
        self.code = code
        self.reset_Golomb_decoding_pos()
        self.ini_codeword(self.code)  # Initialize the arithmetic decoder with the provided code
        
        sym = []  # List to store the decoded symbols
        code_value_fix = []  # Temporary buffer for decoded fixed-length values
        code_value = []  # Temporary buffer for currently decoded value bits
        previous_bit_value = 1  # Initialize the previous bit value to 1
        
        # Decode until the desired number of symbols (N) are obtained
        while len(sym) < N:
            # Decode the next bit using the current state and update the previous bit value
            bin_value = self.decoded_value(self.code, previous_bit_value)
            previous_bit_value = bin_value
            code_value.append(bin_value)  # Accumulate the decoded bit
            
            # Check if the accumulated bits match a known value in the dictionary
            for v in dico_value:
                if dico_value[v] == code_value:
                    if v == 0:
                        # If the matched value is 0, reset the code value buffers
                        code_value_fix.append(0)
                        code_value = []
                        previous_bit_value = 1
                    else:
                        # Decode the polarity/sign if enabled
                        if polarity:
                            bin_p = self.decoded_polarity(self.code)
                            sign = bin_p * 2 - 1  # Convert binary polarity to +/- 1
                        else:
                            sign = 1
                        
                        # Append the fixed-length decoded value and compute the symbol
                        code_value_fix.append(v)
                        ind = self.fix_dec(code_value_fix, n_value_max)
                        sym.append(ind * sign)  # Store the decoded symbol
                        
                        # Reset the buffers for the next symbol
                        code_value_fix = []
                        code_value = []
                        previous_bit_value = 1
                    break  # Exit the loop after finding a match
        
        return sym  # Return the list of decoded symbols

  
if __name__ == "__main__":
    # Import the necessary modules and classes
    from Normalize import normalize
    from Dico import Dico_Cos_Sin_Dirac
    from OMP import OMP
    from get_test_signal import get_RTE_signal as get_signal  # Import signal retrieval function

    # General parameters
    fs = 6400  # Sampling frequency in Hz
    fn = 50    # Nominal frequency in Hz

    # Signal, phase, and window selection
    id_signal = 1  # The ID of the signal to process
    id_phase = 0   # The phase to analyze
    w = 18         # The index of the window to use
    N = 128        # The size of each window

    # Retrieve a segment of the signal (specific phase and window)
    x = get_signal(id_signal)[id_phase][w * N:w * N + N]

    # Normalize the signal and obtain a scaling factor
    x_n, kx = normalize(x)

    # Create a time vector for plotting or analysis
    t = np.linspace(0, (N-1)*(1/fs), N)

    # Initialize the OMP (Orthogonal Matching Pursuit) object
    O = OMP()

    # Dictionary parameters
    f_max = fs / 2  # Maximum frequency (Hz)
    f_factor = 1    # Factor affecting dictionary frequencies
    nb_vect_max = np.infty  # Maximum number of vectors (infinite here)

    # Create a dictionary using cosine, sine, and Dirac components
    Dico = Dico_Cos_Sin_Dirac(fs, f_factor)
    D = Dico.Creat_dico(N, f_max)
    size_D = np.size(D, 1)

    # Set a maximum root-mean-square error for the reconstruction
    RMSE_max = 200 * 2**(-kx)  # in volts

    # Use OMP to find the sparse representation indices (L)
    L, _ = O.best_vect(x_n, D, RMSE_max, nb_vect_max)

    # Golomb coding setup
    adaptive = True    # Enable adaptive coding
    polarity = True    # Enable differential coding
    G = Golomb()       # Instantiate a Golomb coder
    G_enc = Golomb_encoding_pos(adaptive)  # Initialize Golomb encoder
    G_dec = Golomb_decoding_pos(adaptive)  # Initialize Golomb decoder

    # Golomb parameter settings
    m_value = 8            # Parameter for Golomb coding
    dico_values = G.creat_dico(m_value, size_D, None)  # Create a Golomb dictionary
    n_value_max = 8         # Maximum value for Golomb coding

    # Perform Golomb encoding and decoding
    if polarity:
        # If differential coding is enabled, compute differences between consecutive indices
        x = [L[0]]  # Start with the first index
        for k in range(len(L)-1):
            x.append(L[k+1] - L[k])  # Store differences

        # Encode the differences using Golomb coding
        binarisation_x, code_x, R_opt_binarisation, R1_opt_binarisation, R_opt_data = G_enc.golomb_encoding(x, dico_values, n_value_max, polarity)
        # Decode the encoded differences
        x_dec = G_dec.golomb_decoding(code_x, dico_values, len(L), n_value_max, polarity)

        # Reconstruct the indices from the differences
        L_dec = [x_dec[0]]  # Start with the first value
        for k in range(len(x_dec)-1):
            L_dec.append(x_dec[k+1] + L_dec[k])  # Add successive differences
    else:
        # If no differential coding, simply encode and decode the indices
        L += 1  # Increment all indices by 1
        binarisation_x, code_x, R_opt_binarisation, R1_opt_binarisation, R_opt_data = G_enc.golomb_encoding(list(L), dico_values, n_value_max, polarity)
        L_dec = G_dec.golomb_decoding(code_x, dico_values, len(L), n_value_max, polarity)

    # Verify that the original and reconstructed indices match
    if np.mean(L) == np.mean(L_dec):
        print("true")
    else:
        print("false")
        print("L    ", list(L), np.mean(L), len(L))
        print("L_dec", L_dec, np.mean(L_dec), len(L_dec))

    # Report the encoding rates
    print("R binarisation   = {:.0f} bits".format(len(binarisation_x)))
    print("R1 binarisation  = {:.0f} bits".format(len(code_x)))


  