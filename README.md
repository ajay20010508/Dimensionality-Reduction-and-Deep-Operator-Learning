# Dimensionality Reduction and Operator Learning

**MSc Mathematics Project | IIT Bhubaneswar | 2026**

- **Author:** Ajay Yadav (Roll No. 24MA05017)
- **Supervisor:** Dr. Amar Deep Sarkar, Assistant Professor, School of Basic Sciences, IIT Bhubaneswar
- **Course:** MA5D002 — Project

---

## Abstract

This project studies dimensionality reduction and deep operator learning. It covers PCA (via perturbation theory), Generalized Hebbian Algorithm (GHA), Kernel Hebbian Algorithm (KHA), the Universal Approximation Theorem (for functions, functionals, and operators), and Deep Operator Networks (DeepONet). A benchmark experiment on heat transfer through a 1D rod demonstrates that combining PCA (85% reduction) with DeepONet achieves competitive accuracy (MSE ≈ 1.10×10⁻¹) against a full-sensor DeepONet (1.04×10⁻¹), while significantly outperforming a standard FNN (2.10×10⁻¹).

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
│   └── 01_heat_equation_PCA_DeepONet.py  # (Coming soon) Heat equation: FNN / DeepONet / PCA-DeepONet
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
| 4 | Heat Equation | PCA + DeepONet on 1D Heat PDE | `Chapter4_HeatEquation/` |

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

### Chapter 4 — Heat Equation
| Model | Input Dim | MSE |
|---|---|---|
| Standard FNN | 100 sensors | 2.10×10⁻¹ |
| Standard DeepONet | 100 sensors | 1.04×10⁻¹ |
| **PCA-DeepONet** | **15 components** | **1.10×10⁻¹** |

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
