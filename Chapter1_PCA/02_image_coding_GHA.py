"""
Chapter 1 - Case Study 2: Image Coding using Generalized Hebbian Algorithm (GHA)
==================================================================================
Project : Dimensionality Reduction and Deep Operator Learning
Author  : Ajay Yadav (Roll No. 24MA05017)
Institute: IIT Bhubaneswar
Supervisor: Dr. Amar Deep Sarkar

Description:
    Implements the Generalized Hebbian Algorithm (Sanger's rule) on 8x8 image
    blocks to extract principal components. Reconstructs the image with and
    without quantization to demonstrate compression (~11:1 ratio).

Libraries: numpy, matplotlib, PIL
"""

import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    from google.colab import files
except ImportError:
    files = None


# ── Image loading ─────────────────────────────────────────────────────────────
def load_user_image():
    if files is not None:
        print("Upload an image file")
        uploaded = files.upload()
        if not uploaded:
            raise ValueError("No image uploaded")
        name = next(iter(uploaded))
        image = Image.open(io.BytesIO(uploaded[name])).convert("L")
        return np.array(image, dtype=np.float32), name
    else:
        path = input("Enter image path: ").strip()
        image = Image.open(path).convert("L")
        return np.array(image, dtype=np.float32), path


# ── Image <-> block utilities ─────────────────────────────────────────────────
def crop_to_block_grid(img, block_size=8):
    h, w = img.shape
    return img[:h - (h % block_size), :w - (w % block_size)]


def image_to_blocks(img, block_size=8):
    h, w = img.shape
    blocks = (img.reshape(h // block_size, block_size, w // block_size, block_size)
                 .transpose(0, 2, 1, 3)
                 .reshape(-1, block_size * block_size))
    means = blocks.mean(axis=1, keepdims=True)
    return blocks - means, means


def blocks_to_image(blocks, image_shape, block_size=8):
    h, w = image_shape
    return (blocks.reshape(h // block_size, w // block_size, block_size, block_size)
                  .transpose(0, 2, 1, 3)
                  .reshape(h, w))


# ── GHA training ─────────────────────────────────────────────────────────────
def initialize_weights(n_components, n_features, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, size=(n_components, n_features)).astype(np.float32)
    for i in range(n_components):
        W[i] /= np.linalg.norm(W[i]) + 1e-8
    return W


def train_gha(data, n_components=8, lr=0.0005, epochs=25, seed=0):
    """Train using Sanger's (Generalized Hebbian) rule."""
    n_samples, n_features = data.shape
    W = initialize_weights(n_components, n_features, seed=seed)
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        indices = rng.permutation(n_samples)
        eta = lr / (1.0 + 0.02 * epoch)
        max_change = 0.0

        for idx in indices:
            x = data[idx]
            y = W @ x
            W_old = W.copy()
            for i in range(n_components):
                proj = sum(y[k] * W[k] for k in range(i + 1))
                delta = np.clip(eta * y[i] * (x - proj), -0.1, 0.1)
                W[i] += delta
                W[i] /= np.linalg.norm(W[i]) + 1e-8
            max_change = max(max_change, np.linalg.norm(W - W_old))

        print(f"Epoch {epoch + 1:02d}, max weight change = {max_change:.6e}")
        if not np.isfinite(W).all():
            raise ValueError("Training became unstable: weights contain NaN or Inf")

    return W


# ── Reconstruction and quantization ──────────────────────────────────────────
def reconstruct_blocks(centered_blocks, means, W):
    coeffs = centered_blocks @ W.T
    return coeffs @ W + means, coeffs


def quantize_uniform(values, bits):
    levels = (2 ** bits) - 1
    vmin, vmax = values.min(), values.max()
    if np.isclose(vmin, vmax):
        return np.zeros_like(values, dtype=np.int32), np.full_like(values, vmin, dtype=np.float32)
    scaled = (values - vmin) / (vmax - vmin)
    q = np.round(scaled * levels).astype(np.int32)
    restored = vmin + (q.astype(np.float32) / levels) * (vmax - vmin)
    return q, restored


def make_weight_mosaic(W, block_size=8):
    masks = []
    for i in range(W.shape[0]):
        m = W[i].reshape(block_size, block_size)
        m = m - m.min()
        if m.max() > 1e-8:
            m = m / m.max()
        masks.append(m)
    return np.vstack([np.hstack(masks[:4]), np.hstack(masks[4:8])])


def compression_ratio(num_blocks, coeff_bits=5, mean_bits=6):
    return (num_blocks * 64 * 8) / (num_blocks * (8 * coeff_bits + mean_bits))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    block_size, n_components, coeff_bits, mean_bits = 8, 8, 5, 6

    img, name = load_user_image()
    img = crop_to_block_grid(img, block_size)
    centered_blocks, means = image_to_blocks(img, block_size)

    W = train_gha(centered_blocks, n_components=n_components, lr=0.0005, epochs=25, seed=0)

    recon_blocks, coeffs = reconstruct_blocks(centered_blocks, means, W)
    recon_img = blocks_to_image(np.clip(recon_blocks, 0, 255), img.shape, block_size)

    _, coeffs_q = quantize_uniform(coeffs, coeff_bits)
    _, means_q  = quantize_uniform(means,  mean_bits)
    recon_q_img = blocks_to_image(np.clip(coeffs_q @ W + means_q, 0, 255), img.shape, block_size)

    mosaic = make_weight_mosaic(W, block_size)
    ratio  = compression_ratio(centered_blocks.shape[0], coeff_bits, mean_bits)
    mse1   = np.mean((img - recon_img) ** 2)
    mse2   = np.mean((img - recon_q_img) ** 2)

    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    ax[0, 0].imshow(img,        cmap="gray", vmin=0, vmax=255); ax[0, 0].set_title(f"(A). Original\n{name}");             ax[0, 0].axis("off")
    ax[0, 1].imshow(mosaic,     cmap="gray");                   ax[0, 1].set_title("(B). Learned 8x8 Weight Masks");       ax[0, 1].axis("off")
    ax[1, 0].imshow(recon_img,  cmap="gray", vmin=0, vmax=255); ax[1, 0].set_title("(C). Reconstructed (8 Components)");  ax[1, 0].axis("off")
    ax[1, 1].imshow(recon_q_img,cmap="gray", vmin=0, vmax=255); ax[1, 1].set_title(f"(D). Quantized\nRatio≈{ratio:.2f}:1"); ax[1, 1].axis("off")
    plt.tight_layout(); plt.show()

    print(f"Reconstruction MSE        : {mse1:.4f}")
    print(f"Quantized Reconstruction MSE: {mse2:.4f}")
    print(f"Compression Ratio         : {ratio:.2f}:1")


if __name__ == "__main__":
    main()
