"""
Chapter 1 - Case Study 3: Image Denoising using Kernel Hebbian Algorithm (KHA)
================================================================================
Project : Dimensionality Reduction and Deep Operator Learning
Author  : Ajay Yadav (Roll No. 24MA05017)
Institute: IIT Bhubaneswar
Supervisor: Dr. Amar Deep Sarkar

Description:
    Compares PCA (via perturbation theory / power iteration) and KHA (Kernel
    Hebbian Algorithm with RBF kernel) for denoising images corrupted by
    White Gaussian Noise (sigma=20). Uses 7x7 overlapping patches.

Libraries: numpy, matplotlib, PIL, math
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from google.colab import files
import io, math

# ── Upload & parameters ───────────────────────────────────────────────────────
uploaded = files.upload()
img_name = list(uploaded.keys())[0]
img      = Image.open(io.BytesIO(uploaded[img_name])).convert("L")
clean    = np.asarray(img, dtype=np.float32) / 255.0

noise_sigma      = 20.0
patch_size       = 7
pca_components   = 80
pca_power_iters  = 60
kha_components   = 60
max_train_patches= 1200
kha_neighbors    = 15
kha_iters        = 40
kha_tol          = 1e-5
seed             = 0
rng              = np.random.default_rng(seed)


# ── Utilities ─────────────────────────────────────────────────────────────────
def add_gaussian_noise(image, sigma, rng):
    return np.clip(image + rng.normal(0.0, sigma / 255.0, image.shape), 0.0, 1.0).astype(np.float32)

def psnr(ref, est):
    mse = np.mean((ref - est) ** 2)
    return float("inf") if mse < 1e-12 else 10.0 * math.log10(1.0 / mse)

def extract_patches(image, patch_size):
    h, w = image.shape
    patches = [image[y:y+patch_size, x:x+patch_size].reshape(-1)
               for y in range(h - patch_size + 1)
               for x in range(w - patch_size + 1)]
    return np.asarray(patches, dtype=np.float32), (h, w)

def aggregate_patches(patches, image_shape, patch_size):
    h, w = image_shape
    acc = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    idx = 0
    for y in range(h - patch_size + 1):
        for x in range(w - patch_size + 1):
            acc[y:y+patch_size, x:x+patch_size] += patches[idx].reshape(patch_size, patch_size)
            cnt[y:y+patch_size, x:x+patch_size] += 1.0
            idx += 1
    return acc / np.maximum(cnt, 1e-8)


# ── PCA via perturbation / power iteration ────────────────────────────────────
def dominant_eigenpairs_perturbation(C, n_components, n_iter=60, tol=1e-6, seed=0):
    """Iterative deflation without SVD — perturbation-style eigen-solver."""
    rng = np.random.default_rng(seed)
    d, B = C.shape[0], C.copy().astype(np.float64)
    eigvals, eigvecs = [], []
    for _ in range(min(n_components, d)):
        v = rng.normal(size=d); v /= np.linalg.norm(v) + 1e-12
        prev = None
        for _ in range(n_iter):
            v_new = B @ v
            if eigvecs:
                U = np.column_stack(eigvecs)
                v_new -= U @ (U.T @ v_new)
            nrm = np.linalg.norm(v_new)
            if nrm < 1e-12: break
            v_new /= nrm
            if prev is not None and np.linalg.norm(v_new - prev) < tol:
                v = v_new; break
            prev, v = v.copy(), v_new
        lam = float(v.T @ B @ v)
        if lam <= 1e-10: break
        eigvals.append(lam); eigvecs.append(v.copy())
        B -= lam * np.outer(v, v)
    if not eigvecs:
        return np.zeros((d, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.column_stack(eigvecs).astype(np.float32), np.asarray(eigvals, dtype=np.float32)

def pca_denoise_perturbation(noisy, patch_size=7, n_components=80, n_iter=60, seed=0):
    patches, image_shape = extract_patches(noisy, patch_size)
    mean_patch = patches.mean(axis=0, keepdims=True)
    X = patches - mean_patch
    C = (X.T @ X) / max(X.shape[0] - 1, 1)
    V, _ = dominant_eigenpairs_perturbation(C, n_components, n_iter, seed=seed)
    recon = (X @ V) @ V.T + mean_patch
    return np.clip(aggregate_patches(recon, image_shape, patch_size), 0.0, 1.0)


# ── KHA helpers ───────────────────────────────────────────────────────────────
def choose_training_subset(patches, max_n, rng):
    if patches.shape[0] <= max_n: return patches.copy()
    idx = rng.choice(patches.shape[0], size=max_n, replace=False); idx.sort()
    return patches[idx]

def estimate_rbf_sigma(patches, rng, sample_size=300):
    n = min(sample_size, patches.shape[0])
    sample = patches[rng.choice(patches.shape[0], n, replace=False)]
    d2 = np.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=2)
    d2 = d2[np.triu_indices(n, k=1)]; d2 = d2[d2 > 0]
    return 1.0 if len(d2) == 0 else float(np.sqrt(np.median(d2)))

def rbf_kernel(X, Y, sigma):
    Xn = np.sum(X * X, axis=1, keepdims=True)
    Yn = np.sum(Y * Y, axis=1, keepdims=True).T
    return np.exp(-np.maximum(Xn + Yn - 2.0 * X @ Y.T, 0.0) / (2.0 * sigma**2 + 1e-12)).astype(np.float32)

def center_kernel_train(K):
    rm, cm, tm = K.mean(axis=0), K.mean(axis=1, keepdims=True), float(K.mean())
    return (K - rm - cm + tm).astype(np.float32), rm, tm

def center_kernel_test(Ktest, trm, ttm):
    return (Ktest - trm - Ktest.mean(axis=1, keepdims=True) + ttm).astype(np.float32)

def iterative_kernel_components(Kc, n_components, n_iter, tol, rng):
    n, q = Kc.shape[0], min(n_components, Kc.shape[0])
    W, _ = np.linalg.qr(rng.normal(size=(n, q)).astype(np.float32))
    prev = None
    for _ in range(n_iter):
        W, _ = np.linalg.qr(Kc @ W)
        if prev is not None and np.all(1.0 - np.abs(np.diag(W.T @ prev)) < tol): break
        prev = W.copy()
    eigvals = np.sum(W * (Kc @ W), axis=0)
    order = np.argsort(eigvals)[::-1]; eigvals, W = eigvals[order], W[:, order]
    keep = eigvals > 1e-8
    return W[:, keep].astype(np.float32), eigvals[keep].astype(np.float32)

def reconstruct_from_kernel_space(query_coords, train_coords, train_patches, neighbors=15, batch_size=256):
    out, neighbors = np.empty((query_coords.shape[0], train_patches.shape[1]), dtype=np.float32), min(neighbors, train_coords.shape[0])
    for s in range(0, query_coords.shape[0], batch_size):
        Z = query_coords[s:s+batch_size]
        d2 = np.sum((Z[:, None, :] - train_coords[None, :, :]) ** 2, axis=2)
        nn_idx  = np.argpartition(d2, kth=neighbors-1, axis=1)[:, :neighbors]
        nn_dist = np.take_along_axis(d2, nn_idx, axis=1)
        scale   = np.median(nn_dist, axis=1, keepdims=True) + 1e-8
        w = np.exp(-nn_dist / (2.0 * scale)); w /= w.sum(axis=1, keepdims=True)
        out[s:s+batch_size] = np.sum(train_patches[nn_idx] * w[:, :, None], axis=1)
    return out

def kha_denoise(noisy, patch_size=7, n_components=40, max_train_patches=1200,
                neighbors=15, n_iter=40, tol=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    all_patches, image_shape = extract_patches(noisy, patch_size)
    train_patches = choose_training_subset(all_patches, max_train_patches, rng)
    mean_patch = train_patches.mean(axis=0, keepdims=True)
    Xtrain, Xall = train_patches - mean_patch, all_patches - mean_patch
    sigma  = estimate_rbf_sigma(Xtrain, rng)
    Ktrain = rbf_kernel(Xtrain, Xtrain, sigma)
    Kc, trm, ttm = center_kernel_train(Ktrain)
    eigvecs, eigvals = iterative_kernel_components(Kc, n_components, n_iter, tol, rng)
    A = eigvecs / np.sqrt(eigvals[None, :] + 1e-12)
    train_coords = Kc @ A
    recon_patches = np.empty_like(all_patches)
    batch_size = 256
    for s in range(0, Xall.shape[0], batch_size):
        Ktest   = rbf_kernel(Xall[s:s+batch_size], Xtrain, sigma)
        Ktestc  = center_kernel_test(Ktest, trm, ttm)
        q_coords = Ktestc @ A
        recon_patches[s:s+batch_size] = reconstruct_from_kernel_space(q_coords, train_coords, train_patches, neighbors, batch_size)
    return np.clip(aggregate_patches(recon_patches, image_shape, patch_size), 0.0, 1.0)


# ── Run denoising ─────────────────────────────────────────────────────────────
noisy   = add_gaussian_noise(clean, noise_sigma, rng)
pca_img = pca_denoise_perturbation(noisy, patch_size, pca_components, pca_power_iters, seed)
kha_img = kha_denoise(noisy, patch_size, kha_components, max_train_patches, kha_neighbors, kha_iters, kha_tol, seed)

print(f"Noisy  PSNR : {psnr(clean, noisy):.2f} dB")
print(f"PCA    PSNR : {psnr(clean, pca_img):.2f} dB")
print(f"KHA    PSNR : {psnr(clean, kha_img):.2f} dB")

# ── Display ───────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 10))
for i, (im, title) in enumerate([(clean, "(A). Clean Image"),
                                  (noisy, f"(B). Noisy (sigma={noise_sigma})"),
                                  (pca_img, "(C). PCA Denoised (Perturbation Theory)"),
                                  (kha_img, "(D). KHA Denoised")]):
    plt.subplot(2, 2, i+1); plt.imshow(im, cmap="gray"); plt.title(title); plt.axis("off")
plt.tight_layout(); plt.show()
