# Dimensionality Reduction and Deep Operator Learning

**MSc Mathematics Project | IIT Bhubaneswar | 2026**

- **Author:** Ajay Yadav (Roll No. 24MA05017)
- **Supervisor:** Dr. Amar Deep Sarkar, Assistant Professor, School of Basic Sciences, IIT Bhubaneswar
- **Course:** MA5D002 — Project

---

## Abstract

This project studies dimensionality reduction and deep operator learning, bridging classical statistical methods with modern neural network theory. It covers Principal Component Analysis (via perturbation theory), the Hebbian-based Maximum Eigenfilter, the Generalized Hebbian Algorithm (GHA), Kernel PCA, the Kernel Hebbian Algorithm (KHA), and the Universal Approximation Theorem for functions, functionals, and nonlinear operators (Chen & Chen). 

To overcome the optimization bottleneck and computational burden of training unconstrained continuous trunk networks on high-dimensional output fields, we introduce **PCA-DeepONet (POD-DeepONet)**. By embedding empirical orthonormal spatial basis functions extracted via Singular Value Decomposition (SVD) directly into the operator learning pipeline, the trunk optimization is replaced by a verified physical coordinate representation. Evaluated on a 1D steady-state heat conduction problem along a laptop CPU heat pipe ($N = 500$ spatial sensors, reduced to $r = 100$ PCA modes), PCA-DeepONet achieves the lowest test relative $L^2$ error ($3.20\%$) and Mean Absolute Error ($0.8199^\circ\text{C}$), outperforming both standard FNN ($4.87\%$) and standard DeepONet ($3.41\%$), while delivering a $1.57\times$ speedup in training runtime and an inference latency of $1.05\text{ ms}$ per 100 samples.

---

## Repository Structure

```
├── Chapter1_PCA/
│   ├── 01_image_compression_PCA.py       # PCA via Perturbation Theory — image compression
│   ├── 02_image_coding_GHA.py            # Image coding using Generalized Hebbian Algorithm
│   └── 03_image_denoising_KHA.py         # Image denoising: PCA vs Kernel Hebbian Algorithm
│
├── Chapter3_DeepONet/
│   ├── 01_FNN_architecture_comparison_linear_ODE.py    # FNN depth/width comparison (linear ODE)
│   ├── 02_DeepONet_train_test_error_linear_ODE.py      # DeepONet train vs test error (linear ODE)
│   ├── 03_FNN_vs_Stacked_Unstacked_DeepONet.py         # FNN vs Stacked/Unstacked DeepONet ± bias
│   └── 04_FNN_vs_DeepONet_nonlinear_ODE.py             # FNN vs DeepONet (nonlinear ODE, TensorFlow)
│
├── Chapter4_HeatEquation/
│   └── 01_heat_equation_PCA_DeepONet.py  # Laptop Heat Pipe: FNN, DeepONet, PCA-FNN, PCA-DeepONet
│
├── README.md
└── requirements.txt
```

---

## Chapter Overview

| Chapter | Topic | Key Method | Notebook |
|---------|-------|-----------|---------|
| 1 | PCA & Image Processing | Perturbation Theory, GHA, KHA | `Chapter1_PCA/` |
| 2 | Universal Approximation Theorem | UAT for Functions, Functionals, Operators | (Theory only) |
| 3 | Deep Operator Networks | DeepONet vs FNN, Linear & Nonlinear ODE | `Chapter3_DeepONet/` |
| 4 | Heat Pipe Thermal Problem | PCA + DeepONet on 1D Steady-State Heat PDE | `Chapter4_HeatEquation/` |

---

## Key Results

### Chapter 1 — Image Compression (PCA)
| Components (l) | Compression Ratio |
|---|---|
| 20 | ~40x |
| 50 | ~16x |
| 100 | ~8x |

### Chapter 3 — DeepONet vs FNN (Linear ODE)
| Model | Test MSE |
|---|---|
| FNN (Best: D=2, W=2560) | ~10⁻⁴ |
| Unstacked DeepONet + Bias | **Best overall** |

### Chapter 4 — Laptop Heat Pipe Thermal Benchmark (Poisson BVP)
Discretization: $N = 500$ spatial sensors, $r = 100$ retained PCA modes, 400 training epochs.
| Model Architecture | Test MAE (°C)| Test MAE (°C) | Train L2 Err (%) | Test L2 Err (%) | Train Time (s) | Infer Time / 100 Samples (ms) | Status
|---|---|---|---|---|---|---|---|
| Standard FNN | 1.2275 |	1.1934 | 4.96% | 4.87% |	4.05 | 	1.52 |	 |
| Standard DeepONet | 0.9357 |	0.8685 |	3.69% |	3.41% |	3.90 |	1.92 |	 |
| PCA-FNN | 0.6993 |	0.8408 |	2.74% |	3.28% |	3.38 |	1.13 |	 |
|**PCA_DeepONet**| 0.7284 | 0.8199 | 2.86% | 3.20% | 2.58 | 1.05 | 🏆 Winner (Top) |
---

## Requirements

Install all dependencies with:
```bash
pip install -r requirements.txt
```

### Running on Google Colab
All scripts are written for Google Colab. Open any `.py` file in Colab as a notebook cell, or use:
```python
# In Colab, paste the code directly into a cell and run
```

---

## References

1. Haykin, S. *Neural Networks and Learning Machines*, 3rd ed. Pearson, 2009.
2. Hebb, D.O. *The Organization of Behavior*. Wiley, 1949.
3. Jolliffe, I.T. *Principal Component Analysis*. Springer, 2002.
4. Chen & Chen. *Universal Approximation to Nonlinear Operators by Neural Networks*. IEEE Trans. Neural Networks.
5. Lu, Jin, Karniadakis. *DeepONet: Learning Nonlinear Operators*. Nature Machine Intelligence, 2021.
