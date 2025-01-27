## Overcomplete Hybrid Dictionaries

This repository provides a Python-based implementation of the method described in the article:  
Sabarimalai Manikandan, M., Kamwa, I., and Samantaray, S. R. (2015). "Simultaneous denoising and compression of power system disturbances using sparse representation on overcomplete hybrid dictionaries." IET Generation, Transmission & Distribution, 9(11), 1077–1088. [doi:10.1049/iet-gtd.2014.0806](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-gtd.2014.0806)

The code is provided as an open-source implementation of the method since the original article did not include a code release. This repository serves as a starting point for researchers and engineers who wish to experiment with the method or extend its capabilities.

---

## Overview of the Method

The approach simultaneously denoises and compresses power system disturbances by leveraging sparse representation over a hybrid dictionary that combines impulse, discrete cosine, and discrete sine bases. By using overcomplete dictionaries, the method reduces block boundary artifacts and facilitates direct estimation of power quantities from the coefficients associated with sinusoidal components.

### Dictionary Construction
The dictionary, $\boldsymbol{\Psi}$, is a concatenation of three matrices:
```math
\boldsymbol{\Psi} = \left[\begin{array}{lll}
\boldsymbol{I} & \mid & \boldsymbol{C} \mid \boldsymbol{S}
\end{array}\right]_{N \times 3N}
```
- **Impulse Matrix ($\boldsymbol{I}$)**: An identity matrix ($N \times N$) representing discrete impulses.
- **Cosine Matrix ($\boldsymbol{C}$)**: A set of sampled discrete cosine waveforms ($N \times N$), where
```math
\left[\boldsymbol{C}\right]_{ij} = \sqrt{\frac{2}{N}} \cdot \varepsilon_{i} \cdot \cos\left(\frac{\pi(2j+1)i}{2N}\right),\quad i=0,\dots,N-1, \ j=0,\dots,N-1
```

  $\varepsilon_i = \frac{1}{\sqrt{2}}$ for $i = 0$, otherwise $\varepsilon_i = 1$.
- **Sine Matrix ($\boldsymbol{S}$)**: A set of sampled discrete sine waveforms ($N \times N$), where
```math
\left[\boldsymbol{S}^{L}\right]_{ij} = \sqrt{\frac{2}{L}} \cdot \varepsilon_{i} \cdot \sin\left(\frac{\pi(2j+1)(i+1)}{2N}\right),\quad i=0,\dots,N-1.
```
$\varepsilon_i = \frac{1}{\sqrt{2}}$ for $i = 0$, otherwise $\varepsilon_i = 1$.

### Sparse Approximation and Matching Pursuit
The sparse representation is achieved using a matching pursuit algorithm that iteratively selects the dictionary elements (atoms) that best reduce the approximation error. This can be expressed as:
```math
MSE = \frac{1}{N} \left\| \mathbf{x} - \sum_{j=1}^{K} \widehat{\alpha}_j \mathbf{d}_{\widehat{i}_j} \right\|^2
```
where 

$(\widehat{\alpha}_1, \ldots, \widehat{\alpha}_K, \widehat{i}_1, \ldots, \widehat{i}_K)$ is the solution that minimizes the error:

```math
\arg \min_{\alpha, i} \frac{1}{N} \left\| \mathbf{x} - \sum_{j=1}^{K} \alpha_j \mathbf{d}_{i_j} \right\|^2
```
The process continues until $MSE \leq MSE_{\text{max}}$.

### Coefficient Reordering
Since the dictionary is not orthogonal, the coefficients $\widehat{\alpha}_1, \ldots, \widehat{\alpha}_K$ are sorted in descending order:
```math
\pi : \{1, \ldots, K\} \to \{\widehat{\alpha}_1, \ldots, \widehat{\alpha}_K\}
```

so that $\pi(1) < \pi(2) < \ldots < \pi(K)$. The corresponding reordered coefficients are:
```math
\{\alpha_{\pi(1)}, \alpha_{\pi(2)}, \ldots, \alpha_{\pi(K)}\}
```
and the corresponding reordered dictionary atoms are:
```math
\{\mathbf{d}_{\pi(1)}, \mathbf{d}_{\pi(2)}, \ldots, \mathbf{d}_{\pi(K)}\}
```

### Coefficient Quantization
The reordered coefficients are quantized using a Jayant quantizer. The goal is to find the minimum value of $n_\alpha$ such that
```math
MSE = \frac{1}{N} \left\| \mathbf{x} - \sum_{j=1}^{K} \widetilde{\alpha}_{\pi(j)} \mathbf{d}_{\widehat{i}_{\pi(j)}} \right\|^2,
```
where
```
\widetilde{\alpha}_{\pi(j)} = \Delta_{\pi(j)} \left\lfloor \frac{\alpha_{\pi(j)}}{\Delta_{\pi(j)}} \right\rfloor + \frac{\Delta_{\pi(j)}}{2},
```
and
```math
\Delta_{\pi(j)} = \frac{w_{\pi(j)}}{2^{n_\alpha}}.
```
Here, $w_{\pi(j)}$ is the dynamic range of coefficient ${\pi(j)}$. For example, $w_{\pi(1)} = 2\sqrt{\frac{n}{2}}$ or $w_1 = 2$ for cosine or impulse coefficients.

For subsequent coefficients
```math
w_{{\pi(j+1)}} =
\begin{cases}
\frac{w_{\pi(j)}}{2^k}, & \text{if } k > 0, \\
w_{\pi(j)}, & \text{otherwise}.
\end{cases}
```
The parameter $k$ is selected to minimize the difference between $w_{\pi(j)}/2$ and $|\widetilde{\alpha}_j{\pi(j)}| \cdot 2^k$, subject to the constraint $w_{\pi(j)}/2 - |\widetilde{\alpha}_{\pi(j)}| \cdot 2^k > 0$.


### Position Encoding

The positions of the selected coefficients are encoded as follows:  
Given the set of selected basis vectors $\{\mathbf{d}_{\pi(1)}, \mathbf{d}_{\pi(2)}, \ldots, \mathbf{d}_{\pi(K)}\}$, the differences between consecutive indices are computed as  
```math
\delta_k = \pi(i_{k+1}) - \pi(i_k), \quad k > 0.  
``` 
This results in the sequence  
```math
\{\pi(1), \delta_2, \ldots, \delta_K\}.  
```
Each difference $\delta_k$ is encoded using an exponential Golomb code, where one bit is used to indicate the sign.

Finally, Context-based Binary Arithmetic Coding (CABAC) is applied to improve compression efficiency. Each group of two bits—(0,0), (0,1), (1,0), and (1,1)—from the Exponential-Golomb-coded differences is encoded using distinct contexts. This approach captures the correlation between consecutive bits, further enhancing compression performance.


## Note
This implementation faithfully replicates the algorithm described in the referenced article. While this code provides a framework for experimentation, readers are encouraged to further optimize the implementation and explore different parameter configurations.


# Prerequisites

- numpy


- matplotlib.pyplot
