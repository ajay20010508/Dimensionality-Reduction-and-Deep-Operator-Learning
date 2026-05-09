"""
Chapter 3 - Experiment 2: DeepONet Training/Test Error (Linear ODE)
=====================================================================
Project : Dimensionality Reduction and Deep Operator Learning
Author  : Ajay Yadav (Roll No. 24MA05017)
Institute: IIT Bhubaneswar
Supervisor: Dr. Amar Deep Sarkar

Description:
    Trains a DeepONet (branch: depth-2 width-40, trunk: depth-3 width-40)
    to learn the antiderivative operator of a linear ODE using GRF inputs.
    Plots training vs test MSE error over 500 epochs.

ODE: u'(x) + u(x) = f(x),  u(0) = 0

Libraries: numpy, torch, scipy, matplotlib
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

np.random.seed(0); torch.manual_seed(0)

# ── GRF data ──────────────────────────────────────────────────────────────────
def grf_samples(n_samples, n_points, length_scale=0.2):
    x = np.linspace(0, 1, n_points)
    X1, X2 = np.meshgrid(x, x)
    K = np.exp(-((X1 - X2) ** 2) / (2 * length_scale ** 2)) + 1e-6 * np.eye(n_points)
    return x, np.random.multivariate_normal(np.zeros(n_points), K, n_samples)

def solve_ode(f_values, x_grid):
    sol = solve_ivp(lambda x, u: -u + np.interp(x, x_grid, f_values),
                    [0, 1], [0], t_eval=x_grid, method="RK45")
    return sol.y[0]

n_train, n_test, n_sensor = 1000, 200, 100
x_sensor, f_train = grf_samples(n_train, n_sensor)
_,        f_test  = grf_samples(n_test,  n_sensor)
u_train = np.array([solve_ode(f, x_sensor) for f in f_train])
u_test  = np.array([solve_ode(f, x_sensor) for f in f_test])

def make_tensors(f, u):
    Xb, Xt, Y = [], [], []
    for i in range(len(f)):
        for j in range(n_sensor):
            Xb.append(f[i]); Xt.append([x_sensor[j]]); Y.append([u[i, j]])
    return (torch.tensor(np.array(v), dtype=torch.float32) for v in (Xb, Xt, Y))

X_branch, X_trunk, Y           = make_tensors(f_train, u_train)
X_branch_test, X_trunk_test, Y_test = make_tensors(f_test, u_test)

# ── DeepONet ──────────────────────────────────────────────────────────────────
trunk_width = branch_width = 40; p = 40; learning_rate = 0.001

class BranchNet(nn.Module):
    def __init__(self): super().__init__(); self.net = nn.Sequential(nn.Linear(n_sensor,40),nn.Tanh(),nn.Linear(40,40),nn.Tanh(),nn.Linear(40,p))
    def forward(self,x): return self.net(x)

class TrunkNet(nn.Module):
    def __init__(self): super().__init__(); self.net = nn.Sequential(nn.Linear(1,40),nn.Tanh(),nn.Linear(40,40),nn.Tanh(),nn.Linear(40,40),nn.Tanh(),nn.Linear(40,p))
    def forward(self,x): return self.net(x)

class DeepONet(nn.Module):
    def __init__(self): super().__init__(); self.branch=BranchNet(); self.trunk=TrunkNet()
    def forward(self,xb,xt): return torch.sum(self.branch(xb)*self.trunk(xt),dim=1,keepdim=True)

model     = DeepONet()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn   = nn.MSELoss()

# ── Training ──────────────────────────────────────────────────────────────────
train_errors, test_errors = [], []
for epoch in range(500):
    model.train(); optimizer.zero_grad()
    loss = loss_fn(model(X_branch, X_trunk), Y)
    loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        test_loss = loss_fn(model(X_branch_test, X_trunk_test), Y_test)
    train_errors.append(loss.item()); test_errors.append(test_loss.item())
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Train: {loss.item():.6f} | Test: {test_loss.item():.6f}")

plt.figure(figsize=(9, 5))
plt.plot(train_errors, label="Training Error"); plt.plot(test_errors, label="Test Error")
plt.xlabel("Epoch"); plt.ylabel("MSE Error"); plt.title("DeepONet: Train vs Test Error (Linear ODE)")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
