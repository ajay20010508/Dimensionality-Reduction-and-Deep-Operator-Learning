"""
Chapter 3 - Experiment 1: FNN Architecture Comparison (Linear ODE)
====================================================================
Project : Dimensionality Reduction and Deep Operator Learning
Author  : Ajay Yadav (Roll No. 24MA05017)
Institute: IIT Bhubaneswar
Supervisor: Dr. Amar Deep Sarkar

Description:
    Compares FNN architectures (depth x width) for learning the antiderivative
    operator on a linear 1D dynamic system. Training and test MSE curves are
    plotted on a log scale for lr = 0.001.

    Configs: (D=2,W=10), (D=3,W=150), (D=4,W=300)

Libraries: numpy, torch, matplotlib
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ── Data generation ───────────────────────────────────────────────────────────
def generate_grf(num_samples, m=100, length=0.2):
    """Gaussian Random Field samples via Cholesky."""
    x   = np.linspace(0, 1, m)
    cov = np.exp(-np.abs(np.subtract.outer(x, x)) / length)
    L   = np.linalg.cholesky(cov + 1e-6 * np.eye(m))
    u   = np.dot(np.random.randn(num_samples, m), L.T)
    return x, u

def antiderivative(u, x):
    return np.cumsum(u, axis=1) * (x[1] - x[0])

x, u_train = generate_grf(10000)
s_train    = antiderivative(u_train, x)
x, u_test  = generate_grf(1000)
s_test     = antiderivative(u_test, x)

u_train_t = torch.tensor(u_train, dtype=torch.float32)
s_train_t = torch.tensor(s_train, dtype=torch.float32)
u_test_t  = torch.tensor(u_test,  dtype=torch.float32)
s_test_t  = torch.tensor(s_test,  dtype=torch.float32)


# ── FNN model ─────────────────────────────────────────────────────────────────
class FNN(nn.Module):
    def __init__(self, depth, width, m=100):
        super().__init__()
        layers = [nn.Linear(m, width)]
        for _ in range(depth - 1):
            layers += [nn.ReLU(), nn.Linear(width, width)]
        layers += [nn.ReLU(), nn.Linear(width, m)]
        self.net = nn.Sequential(*layers)

    def forward(self, u):
        return self.net(u)


# ── Experiment loop ───────────────────────────────────────────────────────────
configs      = [(2, 10), (3, 150), (4, 300)]
epochs       = 500
log_interval = 50
colors       = ['tab:blue', 'tab:red', 'tab:green']

plt.figure(figsize=(12, 7))

for i, (d, w) in enumerate(configs):
    print(f"Training: Depth={d}, Width={w}...")
    model     = FNN(depth=d, width=w)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn   = nn.MSELoss()
    train_hist, test_hist = [], []

    for epoch in range(epochs):
        model.train(); optimizer.zero_grad()
        loss = loss_fn(model(u_train_t), s_train_t)
        loss.backward(); optimizer.step()
        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                test_loss = loss_fn(model(u_test_t), s_test_t)
            train_hist.append(loss.item())
            test_hist.append(test_loss.item())

    epoch_axis = np.arange(0, epochs, log_interval)
    plt.plot(epoch_axis, train_hist, label=f"D{d}W{w} Train", color=colors[i], linestyle='-',  linewidth=1.5)
    plt.plot(epoch_axis, test_hist,  label=f"D{d}W{w} Test",  color=colors[i], linestyle='--', linewidth=1.5)

plt.yscale("log"); plt.xlabel("Epochs"); plt.ylabel("MSE Loss")
plt.title("FNN Architecture Comparison (Train vs Test), lr=0.001")
plt.legend(loc='upper right'); plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout(); plt.show()

