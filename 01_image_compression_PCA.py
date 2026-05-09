"""
Chapter 1 - Case Study 1: Image Compression using PCA (Perturbation Theory)
=============================================================================
Project : Dimensionality Reduction and Operator Learning
Author  : Ajay Yadav (Roll No. 24MA05017)
Institute: IIT Bhubaneswar
Supervisor: Dr. Amar Deep Sarkar

Description:
    Implements PCA via perturbation theory for grayscale image compression.
    The covariance matrix is perturbed to study eigenvalue corrections.
    Reconstructed images are shown for l = 20, 50, 100 principal components.

Libraries: numpy, matplotlib, skimage, imageio
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import imageio
from google.colab import files

# ── Step 1: Upload image ─────────────────────────────────────────────────────
uploaded = files.upload()
filename = list(uploaded.keys())[0]
print("Uploaded file:", filename)

# ── Step 2: Load image as grayscale ──────────────────────────────────────────
img = imageio.imread(filename)
if img.ndim == 3:
    img_gray = color.rgb2gray(img)
else:
    img_gray = img.astype(float) / 255.0
X = img_gray

# ── Step 3: Dimensions ───────────────────────────────────────────────────────
W, L = X.shape
original_size = W * L
print(f"Original image size: {W}x{L} = {original_size}")

# ── Step 4: Center data and compute covariance ───────────────────────────────
X_flat = X - np.mean(X, axis=0)
cov_matrix = np.cov(X_flat.T)

# ── Step 5: Eigen decomposition ──────────────────────────────────────────────
eigvals, eigvecs = np.linalg.eigh(cov_matrix)
idx = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

# ── Step 6: Perturbation (eigenvalue correction) ─────────────────────────────
epsilon = 0.01
perturbation = epsilon * np.random.randn(*cov_matrix.shape)
cov_perturbed = cov_matrix + perturbation
eigvals_corr = eigvals + np.array(
    [eigvecs[:, i].T @ perturbation @ eigvecs[:, i] for i in range(len(eigvals))]
)

# ── Helper: Compress and reconstruct ─────────────────────────────────────────
def compress_reconstruct(X_flat, eigvecs, k):
    eigvecs_k = eigvecs[:, :k]
    X_proj = X_flat @ eigvecs_k
    X_recon = X_proj @ eigvecs_k.T + np.mean(X, axis=0)
    return X_recon

# ── Step 7: Compare Original + compressed at l = 20, 50, 100 ─────────────────
ks = [20, 50, 100]
labels = ['A', 'B', 'C', 'D']

plt.figure(figsize=(20, 6))

plt.subplot(1, 4, 1)
plt.imshow(X, cmap='gray')
plt.axis('off')
plt.title(f"A: Original\nSize={W}*{L}")

for i, (k, label) in enumerate(zip(ks, labels[1:])):
    X_recon = compress_reconstruct(X_flat, eigvecs, k)
    compressed_size = W * k
    ratio = original_size / compressed_size
    plt.subplot(1, 4, i + 2)
    plt.imshow(X_recon, cmap='gray')
    plt.axis('off')
    plt.title(f"{label}: l={k}\nRatio={ratio:.2f}%")

plt.tight_layout()
plt.show()

# ── Step 8: Print compression statistics ─────────────────────────────────────
print(f"A: Original image size = {original_size}")
for k, label in zip(ks, labels[1:]):
    compressed_size = W * k
    ratio = original_size / compressed_size
    print(f"{label}: l={k}, Compressed size={compressed_size}, Compression ratio={ratio:.2f}%")
