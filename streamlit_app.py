import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.integrate import solve_ivp

st.title("LQR Control of Two-Mass Spring-Damper System")

# --- Sliders for Q and R matrices ---
st.sidebar.header("LQR Tuning Parameters")

q1 = st.sidebar.slider("Q[0,0] (Position y₁)", 1.0, 500.0, 100.0)
q2 = st.sidebar.slider("Q[1,1] (Position y₂)", 1.0, 500.0, 100.0)
q3 = st.sidebar.slider("Q[2,2] (Velocity ẏ₁)", 0.1, 50.0, 1.0)
q4 = st.sidebar.slider("Q[3,3] (Velocity ẏ₂)", 0.1, 50.0, 1.0)

r1 = st.sidebar.slider("R[0,0] (Effort u₁)", 0.1, 10.0, 3.0)
r2 = st.sidebar.slider("R[1,1] (Effort u₂)", 0.1, 10.0, 3.0)

# --- System Parameters ---
m = 1.0
k = 0.5
b = 0.1
apply_ur = True

A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [-k/m, k/m, -b/m, b/m],
    [k/m, -k/m, b/m, -b/m]
])

B = np.array([
    [0, 0],
    [0, 0],
    [1/m, 0],
    [0, 1/m]
])

# --- LQR Matrices ---
Q = np.diag([q1, q2, q3, q4])
R = np.diag([r1, r2])

# Solve CARE and get LQR gain
P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

# --- Reference state ---
xr = np.array([0.3, 1.8, 0, 0])
ur, *_ = np.linalg.lstsq(B, -A @ xr, rcond=None)

# --- Dynamics ---
def dynamics(t, x):
    u = -K @ (x - xr) + ur * apply_ur
    return A @ x + B @ u

# --- Simulation ---
x0 = np.zeros(4)
t_eval = np.linspace(0, 10, 1000)
sol = solve_ivp(dynamics, (0, 10), x0, t_eval=t_eval)

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sol.t, sol.y[0], label=r'$y_1$')
ax.plot(sol.t, sol.y[1], label=r'$y_2$')
ax.plot(sol.t, sol.y[2], label=r'$\dot{y}_1$')
ax.plot(sol.t, sol.y[3], label=r'$\dot{y}_2$')
ax.axhline(xr[0], color='blue', linestyle='--', linewidth=1, label=r'$y_1^{\mathrm{ref}}$')
ax.axhline(xr[1], color='red', linestyle='--', linewidth=1, label=r'$y_2^{\mathrm{ref}}$')
ax.set_xlabel("Time [s]")
ax.set_ylabel("State")
ax.set_title("LQR State Response")
ax.grid(True)
ax.legend()
st.pyplot(fig)
