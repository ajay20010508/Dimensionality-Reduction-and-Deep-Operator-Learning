"""
Chapter 3 - Experiment 4: FNN vs DeepONet for Nonlinear ODE
=============================================================
Project : Dimensionality Reduction and Operator Learning
Author  : Ajay Yadav (Roll No. 24MA05017)
Institute: IIT Bhubaneswar
Supervisor: Dr. Amar Deep Sarkar

Description:
    Compares FNN and DeepONet for learning the solution operator of a
    nonlinear ODE: s'(x) = -s(x)^2 + u(x), s(0) = 0.
    Input functions u(x) are random Fourier series. Training/test MSE
    curves are plotted on a log scale over 500 epochs.

Libraries: numpy, tensorflow/keras, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models

np.random.seed(42); tf.random.set_seed(42)

# ── Problem setup ─────────────────────────────────────────────────────────────
x0, x1          = 0.0, 1.0
m                = 100
x_sensor         = np.linspace(x0, x1, m)
n_y              = 100
y_grid           = np.linspace(x0, x1, n_y)
n_train_funcs    = 1000
n_test_funcs     = 200

# ── Data generation ───────────────────────────────────────────────────────────
def sample_u(x, n_modes=5):
    coeffs = np.random.uniform(-1.0, 1.0, size=n_modes)
    return sum(c * np.sin((k+1) * np.pi * x) for k, c in enumerate(coeffs))

def solve_ode(u_vals, x_sensor, y_grid):
    sol = solve_ivp(lambda x, s: -s[0]**2 + np.interp(x, x_sensor, u_vals),
                    (y_grid[0], y_grid[-1]), [0.0], t_eval=y_grid, method="RK45")
    return sol.y[0]

def generate_dataset(n_funcs):
    U, S = [], []
    for _ in range(n_funcs):
        u = sample_u(x_sensor); s = solve_ode(u, x_sensor, y_grid)
        U.append(u); S.append(s)
    return np.array(U, dtype=np.float32), np.array(S, dtype=np.float32)

print("Generating datasets...")
U_train, S_train = generate_dataset(n_train_funcs)
U_test,  S_test  = generate_dataset(n_test_funcs)

def build_pointwise(U, S):
    X, Y = [], []
    for i in range(len(U)):
        for j, y in enumerate(y_grid):
            X.append(np.concatenate([U[i], [y]])); Y.append(S[i, j])
    return np.array(X), np.array(Y).reshape(-1, 1)

def build_don(U, S):
    b, t, targets = [], [], []
    for i in range(len(U)):
        for j, y in enumerate(y_grid):
            b.append(U[i]); t.append([y]); targets.append(S[i, j])
    return np.array(b), np.array(t), np.array(targets).reshape(-1, 1)

X_train_fnn, y_train_fnn = build_pointwise(U_train, S_train)
X_test_fnn,  y_test_fnn  = build_pointwise(U_test,  S_test)
b_train, t_train, y_train_don = build_don(U_train, S_train)
b_test,  t_test,  y_test_don  = build_don(U_test,  S_test)

# ── Model architectures ───────────────────────────────────────────────────────
fnn = models.Sequential([
    layers.Input(shape=(m+1,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(1)
])
fnn.compile(optimizer='adam', loss="mse", metrics=["mse"])

def build_deeponet(p=64):
    bi = layers.Input(shape=(m,)); ti = layers.Input(shape=(1,))
    b  = layers.Dense(128, activation="relu")(bi); b = layers.Dense(p)(b)
    t  = layers.Dense(256, activation="relu")(ti); t = layers.Dense(p)(t)
    res = layers.Dot(axes=1)([b, t])
    out = layers.Dense(1, use_bias=True)(layers.Reshape((1,))(res))
    return models.Model(inputs=[bi, ti], outputs=out)

deeponet = build_deeponet()
deeponet.compile(optimizer='adam', loss="mse", metrics=["mse"])

# ── Training ──────────────────────────────────────────────────────────────────
epochs = 500
print("Training FNN...")
history_fnn = fnn.fit(X_train_fnn, y_train_fnn,
                      validation_data=(X_test_fnn, y_test_fnn),
                      epochs=epochs, batch_size=1024, verbose=0)

print("Training DeepONet...")
history_don = deeponet.fit([b_train, t_train], y_train_don,
                            validation_data=([b_test, t_test], y_test_don),
                            epochs=epochs, batch_size=1024, verbose=0)

# ── Unified visualization ─────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.plot(history_fnn.history['mse'],     color='blue',    label='FNN Training Error')
plt.plot(history_fnn.history['val_mse'], color='cyan',    linestyle='--', label='FNN Test Error')
plt.plot(history_don.history['mse'],     color='darkred', label='DeepONet Training Error')
plt.plot(history_don.history['val_mse'], color='orange',  linestyle='--', label='DeepONet Test Error')
plt.title('Error Comparison: FNN vs DeepONet (Nonlinear ODE)')
plt.ylabel('MSE'); plt.xlabel('Epoch'); plt.yscale('log')
plt.legend(loc='upper right'); plt.grid(True, which="both", alpha=0.3)
plt.tight_layout(); plt.show()

print(f"FNN Generalization Gap    : {abs(history_fnn.history['mse'][-1] - history_fnn.history['val_mse'][-1]):.4e}")
print(f"DeepONet Generalization Gap: {abs(history_don.history['mse'][-1] - history_don.history['val_mse'][-1]):.4e}")
