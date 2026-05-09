"""
Chapter 4 - Heat Transfer through a Rod: Numerical Experiment with PCA and DeepONet
======================================================================================
Project : Dimensionality Reduction and Deep Operator Learning
Author  : Ajay Yadav (Roll No. 24MA05017)
Institute: IIT Bhubaneswar
Supervisor: Dr. Amar Deep Sarkar

Description:
    Compares three models for solving the 1D heat equation with Dirichlet BCs:
        1. Optimized FNN
        2. Standard DeepONet (full 150-sensor input)
        3. PCA-DeepONet (20 principal components)

PDE  : du/dt = alpha * d2u/dx2,   alpha = 0.01
BCs  : u(0,t) = u(1,t) = 0  (Dirichlet)
Grid : nx=150, dt=0.0005, t_final=0.1
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 2. PARAMETERS
# ============================================================
num_train_samples = 1500
num_test_samples  = 100
nx                = 150
points_per_sample = 100
x                 = np.linspace(0, 1, nx)

alpha   = 0.01
t_final = 0.1
dt      = 0.0005

epochs        = 300
batch_size    = 256
learning_rate = 1e-3
n_pca         = 20
latent_dim    = 128


# ============================================================
# 3. DATA GENERATION
# ============================================================
def generate_u0(num_samples, x):
    U0_list = []
    for _ in range(num_samples):
        u0  = np.random.uniform(0.5, 2.0) * np.sin(np.random.randint(1, 4) * np.pi * x)
        u0 += np.random.uniform(0.2, 1.0) * np.exp(-((x - np.random.uniform(0.3, 0.7))**2) / 0.01)
        U0_list.append(u0)
    return np.array(U0_list)


def solve_heat(u0_list, x, dt, t_final, k):
    dx  = x[1] - x[0]
    cfl = k * dt / (dx ** 2)
    if cfl > 0.5:
        print(f"Warning: CFL = {cfl:.4f} > 0.5, explicit scheme may be unstable.")
    U_targets = []
    for u_init in u0_list:
        u = u_init.copy()
        for _ in range(int(t_final / dt)):
            u_new = np.zeros_like(u)
            u_new[1:-1] = u[1:-1] + cfl * (u[2:] - 2*u[1:-1] + u[:-2])
            u = u_new.copy()
        U_targets.append(u)
    return np.array(U_targets)


# ============================================================
# 4. GENERATE TRAIN AND TEST DATA
# ============================================================
print("Generating training data...")
U0_train       = generate_u0(num_train_samples, x)
U_train_target = solve_heat(U0_train, x, dt, t_final, alpha)

print("Generating test data...")
U0_test       = generate_u0(num_test_samples, x)
U_test_target = solve_heat(U0_test, x, dt, t_final, alpha)


# ============================================================
# 5. PCA DIMENSIONALITY REDUCTION
# ============================================================
pca          = PCA(n_components=n_pca)
U0_train_pca = pca.fit_transform(U0_train)
U0_test_pca  = pca.transform(U0_test)
print(f"PCA: {nx} sensors reduced to {n_pca} components ({100*(1 - n_pca/nx):.1f}% reduction)")


# ============================================================
# 6. PREPARE POINTWISE DATA
# ============================================================
def prepare_pointwise_data(U0_input, U_target, x, points_per_sample, U0_pca=None):
    X_fnn, X_branch_std, X_branch_pca, X_trunk, Y = [], [], [], [], []
    for i in range(len(U0_input)):
        idx = np.random.choice(len(x), points_per_sample, replace=False)
        for j in idx:
            X_fnn.append(np.append(U0_input[i], x[j]))
            X_branch_std.append(U0_input[i])
            if U0_pca is not None:
                X_branch_pca.append(U0_pca[i])
            X_trunk.append([x[j]])
            Y.append(U_target[i, j])
    X_branch_pca = np.array(X_branch_pca) if U0_pca is not None else None
    return (np.array(X_fnn), np.array(X_branch_std),
            X_branch_pca, np.array(X_trunk), np.array(Y))


print("Preparing training points...")
X_fnn_train, X_branch_std_train, X_branch_pca_train, X_trunk_train, Y_train = \
    prepare_pointwise_data(U0_train, U_train_target, x, points_per_sample, U0_train_pca)

print("Preparing test points...")
X_fnn_test, X_branch_std_test, X_branch_pca_test, X_trunk_test, Y_test = \
    prepare_pointwise_data(U0_test, U_test_target, x, points_per_sample, U0_test_pca)

print(f"Training points = {len(Y_train)}")
print(f"Test points     = {len(Y_test)}")


# ============================================================
# 7. NORMALIZATION
# ============================================================
scaler_fnn        = StandardScaler()
scaler_branch_std = StandardScaler()
scaler_branch_pca = StandardScaler()
scaler_trunk      = StandardScaler()
scaler_y          = StandardScaler()

X_fnn_train_scaled        = scaler_fnn.fit_transform(X_fnn_train)
X_fnn_test_scaled         = scaler_fnn.transform(X_fnn_test)
X_branch_std_train_scaled = scaler_branch_std.fit_transform(X_branch_std_train)
X_branch_std_test_scaled  = scaler_branch_std.transform(X_branch_std_test)
X_branch_pca_train_scaled = scaler_branch_pca.fit_transform(X_branch_pca_train)
X_branch_pca_test_scaled  = scaler_branch_pca.transform(X_branch_pca_test)
X_trunk_train_scaled      = scaler_trunk.fit_transform(X_trunk_train)
X_trunk_test_scaled       = scaler_trunk.transform(X_trunk_test)
Y_train_scaled = scaler_y.fit_transform(Y_train.reshape(-1, 1)).ravel()
Y_test_scaled  = scaler_y.transform(Y_test.reshape(-1, 1)).ravel()


# ============================================================
# 8. MODEL DEFINITIONS
# ============================================================
def build_fnn(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    z = layers.Dense(64, activation="tanh")(inputs)
    z = layers.Dense(64, activation="tanh")(z)
    z = layers.Dense(32, activation="tanh")(z)
    z = layers.Dense(32, activation="tanh")(z)
    z = layers.Dense(16, activation="tanh")(z)
    outputs = layers.Dense(1)(z)
    model = Model(inputs, outputs, name="Optimized_FNN")
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def build_deeponet(branch_dim, latent_dim=128, name="DeepONet"):
    branch_input = layers.Input(shape=(branch_dim,))
    b = layers.Dense(128, activation="relu")(branch_input)
    b = layers.Dense(128, activation="relu")(b)
    b = layers.Dense(64,  activation="relu")(b)
    b = layers.Dense(latent_dim)(b)

    trunk_input = layers.Input(shape=(1,))
    t = layers.Dense(128, activation="relu")(trunk_input)
    t = layers.Dense(128, activation="relu")(t)
    t = layers.Dense(64,  activation="relu")(t)
    t = layers.Dense(latent_dim)(t)

    output = layers.Dot(axes=1)([b, t])
    model  = Model(inputs=[branch_input, trunk_input], outputs=output, name=name)
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


model_fnn    = build_fnn(nx + 1)
model_std_do = build_deeponet(nx,    latent_dim=latent_dim, name="Optimized_Std_DeepONet")
model_pca_do = build_deeponet(n_pca, latent_dim=latent_dim, name="Optimized_PCA_DeepONet")


# ============================================================
# 9. CUSTOM CALLBACK FOR TEST MSE
# ============================================================
class TestMSECallback(callbacks.Callback):
    def __init__(self, test_inputs, y_test_scaled):
        super().__init__()
        self.test_inputs   = test_inputs
        self.y_test_scaled = y_test_scaled
        self.test_mse      = []

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.test_inputs, verbose=0).flatten()
        self.test_mse.append(np.mean((self.y_test_scaled - pred) ** 2))


# ============================================================
# 10. CALLBACKS
# ============================================================
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1
)


# ============================================================
# 11. TRAIN FNN
# ============================================================
print("\nTraining Optimized FNN...")
fnn_test_cb = TestMSECallback(X_fnn_test_scaled, Y_test_scaled)
history_fnn = model_fnn.fit(
    X_fnn_train_scaled, Y_train_scaled,
    validation_data=(X_fnn_test_scaled, Y_test_scaled),
    epochs=epochs, batch_size=batch_size, verbose=1,
    callbacks=[fnn_test_cb, reduce_lr]
)


# ============================================================
# 12. TRAIN STANDARD DEEPONET
# ============================================================
print("\nTraining Optimized Standard DeepONet...")
std_test_cb = TestMSECallback([X_branch_std_test_scaled, X_trunk_test_scaled], Y_test_scaled)
history_std = model_std_do.fit(
    [X_branch_std_train_scaled, X_trunk_train_scaled], Y_train_scaled,
    validation_data=([X_branch_std_test_scaled, X_trunk_test_scaled], Y_test_scaled),
    epochs=epochs, batch_size=batch_size, verbose=1,
    callbacks=[std_test_cb, reduce_lr]
)


# ============================================================
# 13. TRAIN PCA-DEEPONET
# ============================================================
print("\nTraining Optimized PCA-DeepONet...")
pca_test_cb = TestMSECallback([X_branch_pca_test_scaled, X_trunk_test_scaled], Y_test_scaled)
history_pca = model_pca_do.fit(
    [X_branch_pca_train_scaled, X_trunk_train_scaled], Y_train_scaled,
    validation_data=([X_branch_pca_test_scaled, X_trunk_test_scaled], Y_test_scaled),
    epochs=epochs, batch_size=batch_size, verbose=1,
    callbacks=[pca_test_cb, reduce_lr]
)


# ============================================================
# 14. PLOT: TRAINING AND TEST ERROR vs EPOCH
# ============================================================
e1 = np.arange(1, len(history_fnn.history["loss"]) + 1)
e2 = np.arange(1, len(history_std.history["loss"]) + 1)
e3 = np.arange(1, len(history_pca.history["loss"]) + 1)

plt.figure(figsize=(12, 6))
plt.plot(e1, history_fnn.history["loss"],  "r-",  label="FNN Train MSE")
plt.plot(e1, fnn_test_cb.test_mse,          "r--", label="FNN Test MSE")
plt.plot(e2, history_std.history["loss"],  "g-",  label="Std DeepONet Train MSE")
plt.plot(e2, std_test_cb.test_mse,          "g--", label="Std DeepONet Test MSE")
plt.plot(e3, history_pca.history["loss"],  "b-",  label="PCA-DeepONet Train MSE")
plt.plot(e3, pca_test_cb.test_mse,          "b--", label="PCA-DeepONet Test MSE")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training and Test Error vs Epoch")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 15. PREDICT ON ONE UNSEEN INPUT FUNCTION
# ============================================================
print("\nGenerating one unseen input function...")
u0_new     = generate_u0(1, x)[0]
u_true_new = solve_heat([u0_new], x, dt, t_final, alpha)[0]

# FNN prediction
X_new_fnn  = np.array([np.append(u0_new, xi) for xi in x])
u_pred_fnn = scaler_y.inverse_transform(
    model_fnn.predict(scaler_fnn.transform(X_new_fnn), verbose=0).reshape(-1, 1)).flatten()

# Standard DeepONet prediction
X_nb_std   = scaler_branch_std.transform(np.tile(u0_new, (nx, 1)))
X_nt       = scaler_trunk.transform(x.reshape(-1, 1))
u_pred_std = scaler_y.inverse_transform(
    model_std_do.predict([X_nb_std, X_nt], verbose=0).reshape(-1, 1)).flatten()

# PCA-DeepONet prediction
X_nb_pca   = scaler_branch_pca.transform(np.tile(pca.transform([u0_new]), (nx, 1)))
u_pred_pca = scaler_y.inverse_transform(
    model_pca_do.predict([X_nb_pca, X_nt], verbose=0).reshape(-1, 1)).flatten()


# ============================================================
# 16. PLOT: EXACT vs APPROXIMATION
# ============================================================
plt.figure(figsize=(12, 6))
plt.plot(x, u_true_new,  "k-",  linewidth=2.5, label="Exact (Numerical)")
plt.plot(x, u_pred_fnn,  "r--", linewidth=2.0, label="Optimized FNN")
plt.plot(x, u_pred_std,  "g-.", linewidth=2.0, label="Optimized Std DeepONet")
plt.plot(x, u_pred_pca,  "b:",  linewidth=2.5, label="Optimized PCA-DeepONet")
plt.xlabel("x")
plt.ylabel("u(x, t=0.1)")
plt.title("Exact vs Predicted Solution for One New Input Function")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 17. MSE FOR ONE UNSEEN INPUT
# ============================================================
print("\nMSE for one unseen input function:")
print(f"  Optimized FNN MSE             = {np.mean((u_true_new - u_pred_fnn)**2):.8e}")
print(f"  Optimized Std DeepONet MSE    = {np.mean((u_true_new - u_pred_std)**2):.8e}")
print(f"  Optimized PCA-DeepONet MSE    = {np.mean((u_true_new - u_pred_pca)**2):.8e}")


# ============================================================
# 18. FINAL TEST MSE ON FULL TEST SET
# ============================================================
def get_pred(model, inputs, scaler_y):
    return scaler_y.inverse_transform(
        model.predict(inputs, verbose=0).reshape(-1, 1)).flatten()

Y_test_true = scaler_y.inverse_transform(Y_test_scaled.reshape(-1, 1)).flatten()
pred_fnn    = get_pred(model_fnn,    X_fnn_test_scaled, scaler_y)
pred_std    = get_pred(model_std_do, [X_branch_std_test_scaled, X_trunk_test_scaled], scaler_y)
pred_pca    = get_pred(model_pca_do, [X_branch_pca_test_scaled, X_trunk_test_scaled], scaler_y)

print("\nFinal Test MSE on 100 unseen test samples:")
print(f"  Optimized FNN Test MSE            = {mean_squared_error(Y_test_true, pred_fnn):.8e}")
print(f"  Optimized Std DeepONet Test MSE   = {mean_squared_error(Y_test_true, pred_std):.8e}")
print(f"  Optimized PCA-DeepONet Test MSE   = {mean_squared_error(Y_test_true, pred_pca):.8e}")
