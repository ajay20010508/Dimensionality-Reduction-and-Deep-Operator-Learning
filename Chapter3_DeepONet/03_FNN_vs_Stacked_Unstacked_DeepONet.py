"""
Chapter 3 - Experiment 3: FNN vs Stacked/Unstacked DeepONet with/without Bias (Linear ODE)
============================================================================================
Project : Dimensionality Reduction and Deep Operator Learning
Author  : Ajay Yadav (Roll No. 24MA05017)
Institute: IIT Bhubaneswar
Supervisor: Dr. Amar Deep Sarkar

Description:
    Compares final MSE (train & test) of six architectures:
      - FNN with/without bias
      - Unstacked DeepONet with/without bias
      - Stacked DeepONet with/without bias
    All trained on the antiderivative operator (linear ODE).
    Results shown as a grouped bar chart.

Libraries: numpy, torch, scipy, matplotlib
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

np.random.seed(0); torch.manual_seed(0)

n_train, n_test, n_sensor = 100, 20, 50
epochs, lr = 300, 0.001

# ── GRF + ODE data ────────────────────────────────────────────────────────────
def grf_samples(n_samples, n_points, length_scale=0.2):
    x = np.linspace(0, 1, n_points); X1, X2 = np.meshgrid(x, x)
    K = np.exp(-((X1-X2)**2)/(2*length_scale**2)) + 1e-6*np.eye(n_points)
    return x, np.random.multivariate_normal(np.zeros(n_points), K, n_samples)

def solve_ode(f_values, x_grid):
    sol = solve_ivp(lambda x, u: -u + np.interp(x, x_grid, f_values), [0,1], [0], t_eval=x_grid, method="RK45")
    return sol.y[0]

x_sensor, f_train = grf_samples(n_train, n_sensor)
_,        f_test  = grf_samples(n_test,  n_sensor)
u_train = np.array([solve_ode(f, x_sensor) for f in f_train])
u_test  = np.array([solve_ode(f, x_sensor) for f in f_test])

def make_fnn_data(f, u):
    X, Y = [], []
    for i in range(len(f)):
        for j in range(n_sensor):
            X.append(np.concatenate([f[i],[x_sensor[j]]])); Y.append([u[i,j]])
    return torch.tensor(np.array(X),dtype=torch.float32), torch.tensor(np.array(Y),dtype=torch.float32)

def make_don_data(f, u):
    Xb, Xt, Y = [], [], []
    for i in range(len(f)):
        for j in range(n_sensor):
            Xb.append(f[i]); Xt.append([x_sensor[j]]); Y.append([u[i,j]])
    return (torch.tensor(np.array(v),dtype=torch.float32) for v in (Xb,Xt,Y))

X_tr_fnn, Y_tr_fnn = make_fnn_data(f_train, u_train)
X_te_fnn, Y_te_fnn = make_fnn_data(f_test,  u_test)
Xb_tr, Xt_tr, Y_tr_don = make_don_data(f_train, u_train)
Xb_te, Xt_te, Y_te_don = make_don_data(f_test,  u_test)

# ── Model definitions ─────────────────────────────────────────────────────────
class FNN(nn.Module):
    def __init__(self, in_dim, use_bias=True):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,20,bias=use_bias),nn.Tanh(),nn.Linear(20,20,bias=use_bias),nn.Tanh(),nn.Linear(20,20,bias=use_bias),nn.Tanh(),nn.Linear(20,1,bias=use_bias))
    def forward(self,x): return self.net(x)

class BranchNet(nn.Module):
    def __init__(self,in_dim,out_dim,use_bias=True):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,20,bias=use_bias),nn.Tanh(),nn.Linear(20,20,bias=use_bias),nn.Tanh(),nn.Linear(20,out_dim,bias=use_bias))
    def forward(self,x): return self.net(x)

class TrunkNet(nn.Module):
    def __init__(self,out_dim,use_bias=True):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1,20,bias=use_bias),nn.Tanh(),nn.Linear(20,20,bias=use_bias),nn.Tanh(),nn.Linear(20,20,bias=use_bias),nn.Tanh(),nn.Linear(20,out_dim,bias=use_bias))
    def forward(self,x): return self.net(x)

class UnstackedDeepONet(nn.Module):
    def __init__(self,m,p=20,use_bias=True):
        super().__init__()
        self.branch=BranchNet(m,p,use_bias); self.trunk=TrunkNet(p,use_bias)
        self.use_bias_term=use_bias
        if use_bias: self.bias=nn.Parameter(torch.zeros(1))
    def forward(self,xb,xt):
        y=torch.sum(self.branch(xb)*self.trunk(xt),dim=1,keepdim=True)
        return y+self.bias if self.use_bias_term else y

class SmallBranch(nn.Module):
    def __init__(self,in_dim,use_bias=True):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,20,bias=use_bias),nn.Tanh(),nn.Linear(20,20,bias=use_bias),nn.Tanh(),nn.Linear(20,1,bias=use_bias))
    def forward(self,x): return self.net(x)

class StackedDeepONet(nn.Module):
    def __init__(self,m,p=20,use_bias=True):
        super().__init__()
        self.branches=nn.ModuleList([SmallBranch(m,use_bias) for _ in range(p)])
        self.trunk=TrunkNet(p,use_bias); self.use_bias_term=use_bias
        if use_bias: self.bias=nn.Parameter(torch.zeros(1))
    def forward(self,xb,xt):
        b=torch.cat([br(xb) for br in self.branches],dim=1)
        y=torch.sum(b*self.trunk(xt),dim=1,keepdim=True)
        return y+self.bias if self.use_bias_term else y

# ── Training function ─────────────────────────────────────────────────────────
def train_model(model, X1_tr, X2_tr, Y_tr, X1_te, X2_te, Y_te, model_type="fnn"):
    opt=torch.optim.Adam(model.parameters(),lr=lr); loss_fn=nn.MSELoss()
    for epoch in range(epochs):
        model.train(); opt.zero_grad()
        pred=model(X1_tr) if model_type=="fnn" else model(X1_tr,X2_tr)
        loss_fn(pred,Y_tr).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        tr_pred = model(X1_tr) if model_type=="fnn" else model(X1_tr,X2_tr)
        te_pred = model(X1_te) if model_type=="fnn" else model(X1_te,X2_te)
    return loss_fn(tr_pred,Y_tr).item(), loss_fn(te_pred,Y_te).item()

# ── Run all models ────────────────────────────────────────────────────────────
results_train, results_test = {}, {}
for label, model, mtype, X1t, X2t, Y_t, X1e, X2e, Y_e in [
    ("FNN + Bias",       FNN(n_sensor+1,True),             "fnn", X_tr_fnn, None,  Y_tr_fnn, X_te_fnn, None,  Y_te_fnn),
    ("FNN - Bias",       FNN(n_sensor+1,False),            "fnn", X_tr_fnn, None,  Y_tr_fnn, X_te_fnn, None,  Y_te_fnn),
    ("Unstacked + Bias", UnstackedDeepONet(n_sensor,20,True), "don", Xb_tr, Xt_tr, Y_tr_don, Xb_te, Xt_te, Y_te_don),
    ("Unstacked - Bias", UnstackedDeepONet(n_sensor,20,False),"don", Xb_tr, Xt_tr, Y_tr_don, Xb_te, Xt_te, Y_te_don),
    ("Stacked + Bias",   StackedDeepONet(n_sensor,20,True),  "don", Xb_tr, Xt_tr, Y_tr_don, Xb_te, Xt_te, Y_te_don),
    ("Stacked - Bias",   StackedDeepONet(n_sensor,20,False), "don", Xb_tr, Xt_tr, Y_tr_don, Xb_te, Xt_te, Y_te_don),
]:
    print(f"Training {label}...")
    tr, te = train_model(model, X1t, X2t, Y_t, X1e, X2e, Y_e, mtype)
    results_train[label]=tr; results_test[label]=te

# ── Print results ─────────────────────────────────────────────────────────────
print("\nTraining Errors:"); [print(f"  {k}: {v:.6f}") for k,v in results_train.items()]
print("\nTest Errors:");     [print(f"  {k}: {v:.6f}") for k,v in results_test.items()]

# ── Bar chart ─────────────────────────────────────────────────────────────────
labels   = list(results_train.keys())
x        = np.arange(len(labels)); width = 0.35
plt.figure(figsize=(12, 6))
plt.bar(x-width/2, list(results_train.values()), width, label="Train Error")
plt.bar(x+width/2, list(results_test.values()),  width, label="Test Error")
plt.xticks(x, labels, rotation=20); plt.ylabel("MSE Error")
plt.title("Error Comparison: FNN, Stacked and Unstacked DeepONet")
plt.legend(); plt.tight_layout(); plt.show()
