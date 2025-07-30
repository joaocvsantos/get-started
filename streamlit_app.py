import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
import tempfile

#region plot
st.title("LQR Control")

# --- Sliders for Q and R matrices ---
st.sidebar.header("Tuning Parameters")

apply_ur = st.sidebar.checkbox("Apply ur")

q1 = st.sidebar.slider("Q[0,0] (Position y₁)", 1.0, 1000.0, 100.0)
q2 = st.sidebar.slider("Q[1,1] (Position y₂)", 1.0, 1000.0, 100.0)
q3 = st.sidebar.slider("Q[2,2] (Velocity ẏ₁)", 0.1, 50.0, 1.0)
q4 = st.sidebar.slider("Q[3,3] (Velocity ẏ₂)", 0.1, 50.0, 1.0)

r1 = st.sidebar.slider("R[0,0] (Effort u₁)", 0.1, 100.0, 1.0)
r2 = st.sidebar.slider("R[1,1] (Effort u₂)", 0.1, 100.0, 1.0)


# --- System Parameters ---
m = 1.0
k = 1.0
b = 0.1

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
t_eval = np.linspace(0, 5, 1000)
sol = solve_ivp(dynamics, (0, 10), x0, t_eval=t_eval)

y1 = sol.y[0]
y2 = sol.y[1]
y1_dot = sol.y[2]
y2_dot = sol.y[3]
X = np.vstack([y1, y2, y1_dot, y2_dot]).T
U = -(X - xr) @ K.T + ur * apply_ur
f1_array = U[:, 0]
f2_array = U[:, 1]
t = sol.t

# --- Plotting ---
st.header('Positions')
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sol.t, sol.y[0], color='blue', label=r'$y_1$')
ax.plot(sol.t, sol.y[1], color='red', label=r'$y_2$')
ax.axhline(xr[0], color='blue', linestyle='--', linewidth=1.3, label=r'$y_1^{\mathrm{ref}}$')
ax.axhline(xr[1], color='red', linestyle='--', linewidth=1.3, label=r'$y_2^{\mathrm{ref}}$')
ax.set_xlabel("Time [s]")
ax.set_ylabel("Positions [m]")
ax.grid(True)
ax.legend()
st.pyplot(fig)

st.header('Forces')
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sol.t, f1_array, color='blue', label=r'$f_1$')
ax.plot(sol.t, f2_array, color='red', label=r'$f_2$')
ax.set_ylim(-10., 30)  # set y-axis limits
ax.set_xlabel("Time [s]")
ax.set_ylabel("Forces [N]")
ax.grid(True)
ax.legend()
st.pyplot(fig)


#endregion 

#region animation

# Extract state and compute control inputs

# Visualization setup
mass_radius = 0.1
arrow_scale = 0.3  # scale for visualizing force arrows

fig, ax = plt.subplots(figsize=(10, 3))
ax.set_xlim(-1, 3)
ax.set_ylim(-0.5, 0.5)
ax.set_aspect('equal')
ax.axis('off')

# Draw masses
mass1 = plt.Rectangle((0, -mass_radius), 2 * mass_radius, 2 * mass_radius, fc='blue')
mass2 = plt.Rectangle((0, -mass_radius), 2 * mass_radius, 2 * mass_radius, fc='red')
ax.add_patch(mass1)
ax.add_patch(mass2)

# Spring and damper lines
spring_line, = ax.plot([], [], 'k-', lw=2)
damper_line, = ax.plot([], [], 'g-', lw=4, alpha=0.4)

# Reference lines
ax.axvline(xr[0], color='blue', linestyle='--', linewidth=1, label=r'$y_1^{\mathrm{ref}}$')
ax.axvline(xr[1], color='red', linestyle='--', linewidth=1, label=r'$y_2^{\mathrm{ref}}$')
ax.legend(loc='upper center', ncol=2)

# Force arrows
f1_arrow = ax.arrow(0, 0.15, 0, 0, width=0.01, color='blue')
f2_arrow = ax.arrow(0, 0.15, 0, 0, width=0.01, color='red')

time_scale = 10
def animate(i):
    global f1_arrow, f2_arrow

    m1x = y1[i*time_scale]
    m2x = y2[i*time_scale]
    f1 = f1_array[i*time_scale]
    f2 = f2_array[i*time_scale]

    # Update mass positions
    mass1.set_xy((m1x - mass_radius, -mass_radius))
    mass2.set_xy((m2x - mass_radius, -mass_radius))

    # Update spring and damper
    spring_line.set_data([m1x + mass_radius, m2x - mass_radius], [0, 0])
    damper_line.set_data([m1x + mass_radius, m2x - mass_radius], [-0.1, -0.1])

    # Remove old arrows
    f1_arrow.remove()
    f2_arrow.remove()

    # Draw new arrows
    f1_arrow = ax.arrow(m1x, 0.15, arrow_scale * f1, 0, width=0.01, color='blue')
    f2_arrow = ax.arrow(m2x, 0.15, arrow_scale * f2, 0, width=0.01, color='red')

    return mass1, mass2, spring_line, damper_line, f1_arrow, f2_arrow

ani = animation.FuncAnimation(fig, animate, frames=round(len(t)/time_scale)-1, interval=20, blit=True)
tmpfile = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
ani.save(tmpfile.name, writer='pillow')
st.header('Animation')
st.image(tmpfile.name)

