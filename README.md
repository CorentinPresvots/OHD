## Overcomplete Hybrid Dictionaries

This repository provides a Python-based implementation of the method described in the article:  
Sabarimalai Manikandan, M., Kamwa, I., and Samantaray, S. R. (2015). "Simultaneous denoising and compression of power system disturbances using sparse representation on overcomplete hybrid dictionaries." IET Generation, Transmission & Distribution, 9(11), 1077–1088. [doi:10.1049/iet-gtd.2014.0806](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-gtd.2014.0806)

The code is provided as an open-source implementation of the method since the original article did not include a code release. This repository serves as a starting point for researchers and engineers who wish to experiment with the method or extend its capabilities.

---

## Overview of the Method

The approach simultaneously denoises and compresses power system disturbances by leveraging sparse representation over a hybrid dictionary that combines impulse, discrete cosine, and discrete sine bases. By using overcomplete dictionaries, the method reduces block boundary artifacts and facilitates direct estimation of power quantities from the coefficients associated with sinusoidal components.

### Dictionary Construction
The dictionary, \(\boldsymbol{\Psi}\), is a concatenation of three matrices:
\[
\boldsymbol{\Psi} = \left[\begin{array}{lll}
\boldsymbol{I} & \mid & \boldsymbol{C} \mid \boldsymbol{S}
\end{array}\right]_{N \times M}
\]
- **Impulse Matrix (\(\boldsymbol{I}\))**: An identity matrix (\(N \times N\)) representing discrete impulses.
- **Cosine Matrix (\(\boldsymbol{C}\))**: A set of sampled discrete cosine waveforms (\(N \times L\)), where:
  \[
  \left[\boldsymbol{C}^{L}\right]_{ij} = \sqrt{\frac{2}{L}} \cdot \epsilon_i \cdot \cos\left(\frac{\pi(2j+1)i}{2L}\right)
  \]
  \(\epsilon_i = \frac{1}{\sqrt{2}}\) for \(i = 0\), otherwise \(\epsilon_i = 1\).
- **Sine Matrix (\(\boldsymbol{S}\))**: A set of sampled discrete sine waveforms (\(N \times L\)), where:
  \[
  \left[\boldsymbol{S}^{L}\right]_{ij} = \sqrt{\frac{2}{L}} \cdot \epsilon_i \cdot \sin\left(\frac{\pi(2j+1)(i+1)}{2L}\right)
  \]
  \(\epsilon_i = \frac{1}{\sqrt{2}}\) for \(i = L-1\), otherwise \(\epsilon_i = 1\).

### Sparse Approximation and Matching Pursuit
The sparse representation is achieved using a matching pursuit algorithm that iteratively selects the dictionary elements (atoms) that best reduce the approximation error. This can be expressed as:
\[
MSE = \frac{1}{N} \left\| \mathbf{x} - \sum_{j=1}^{K} \widehat{\alpha}_j \mathbf{d}_{\widehat{i}_j} \right\|^2
\]
where \((\widehat{\alpha}_1, \ldots, \widehat{\alpha}_K, \widehat{i}_1, \ldots, \widehat{i}_K)\) is the solution that minimizes the error:
\[
\arg \min_{\alpha, i} \frac{1}{N} \left\| \mathbf{x} - \sum_{j=1}^{K} \alpha_j \mathbf{d}_{i_j} \right\|^2
\]
The process continues until \(MSE \leq MSE_{\text{max}}\).

### Coefficient Reordering
Since the dictionary is not orthogonal, the coefficients \(\widehat{\alpha}_1, \ldots, \widehat{\alpha}_K\) are sorted in descending order:
\[
\pi : \{1, \ldots, K\} \to \{\widehat{\alpha}_1, \ldots, \widehat{\alpha}_K\}
\]
so that \(\pi(1) < \pi(2) < \ldots < \pi(K)\). The corresponding reordered coefficients are:
\[
\{\alpha_{\pi(1)}, \alpha_{\pi(2)}, \ldots, \alpha_{\pi(K)}\}
\]
and the corresponding reordered dictionary atoms are:
\[
\{\mathbf{d}_{\pi(1)}, \mathbf{d}_{\pi(2)}, \ldots, \mathbf{d}_{\pi(K)}\}
\]

### Coefficient Quantization
The reordered coefficients are then quantized using a Jayant quantizer:
1. Initialize quantization step size \(\Delta\) based on the dynamic range:
   \[
   \Delta_j = \frac{w_j}{2^{n_\alpha}}
   \]
   where \(w_j\) is the range of the coefficient \(j\) (e.g., \(w_j = 2\sqrt{\frac{n}{2}}\) or \(w_j = 2\) for cosines or impulses).
2. Quantize each coefficient:
   \[
   \widetilde{\alpha}_j = \Delta_j \left\lfloor \frac{\alpha_j}{\Delta_j} \right\rfloor + \frac{\Delta_j}{2}
   \]
3. Determine the next scaling factor \(w_{j+1}\):
   \[
   w_{j+1} =
   \begin{cases}
   \frac{w_j}{2^k}, & \text{if } k > 0 \\
   w_j, & \text{otherwise}
   \end{cases}
   \]
   where \(k\) is chosen to minimize the difference between \(w_j/2\) and \(|\widetilde{\alpha}_j| \cdot 2^k\), subject to \(w_j/2 - |\widetilde{\alpha}_j| \cdot 2^k > 0\).

### Position Encoding
The positions of the selected coefficients are encoded using:
1. Differential coding:
   \[
   \delta_i = i_{k+1} - i_k
   \]
2. Exponential-Golomb coding:
   Each difference is represented using a combination of unary and binary coding.
3. CABAC (Context Adaptive Binary Arithmetic Coding):
   Each bit of the Exponential-Golomb-coded differences is encoded with a distinct context for higher compression efficiency.

---

## Pseudocode

**Initialization**  
- Build dictionary \(\boldsymbol{\Psi} = [\boldsymbol{I} \mid \boldsymbol{C} \mid \boldsymbol{S}]\)  
- Set \(MSE_{\text{max}}\)  
- Initialize \(MSE = \infty\)

**Sparse Approximation**  
- While \(MSE > MSE_{\text{max}}\):  
  1. Select the most significant coefficient and its index.  
  2. Update the reconstructed signal and the residual.  
  3. Compute the new \(MSE\).

**Coefficient Sorting**  
- Sort the coefficients \(\alpha\) in descending order.  
- Reorder the indices accordingly.

**Quantization**  
- Initialize Jayant quantizer.  
- For each coefficient:  
  - Quantize \(\alpha\) using a uniform quantizer.  
  - Compute the scaling factor \(k\) and adjust the quantizer step size.  
  - Update the dynamic range for the next coefficient.

**Position Encoding**  
- Apply differential coding to the indices.  
- Use Exponential-Golomb coding for the differences.  
- Apply CABAC to encode the resulting binary values.

---

## Note
This implementation faithfully replicates the algorithm described in the referenced article. While this code provides a framework for experimentation, readers are encouraged to further optimize the implementation and explore different parameter configurations.


# Prerequisites

- numpy


- matplotlib.pyplot
