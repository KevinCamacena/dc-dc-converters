import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def buck(x,t, L, R, C, Uav, E):
    i,v = x
    di_dt = E * Uav / L - v / L
    dv_dt = i / C - v / R / C
    return [di_dt, dv_dt]

# Differential equations
def buck_boost(x, t, L, R, C, Uav, E):
    i, v = x
    di_dt = (E * Uav / L) + (1 - Uav) * v / L
    dv_dt = - (1 - Uav) * i / C - v / (R * C)
    return [di_dt, dv_dt]


# Differential equations with non-negative current enforcement
def buck_boost_clamped(x, t, L, R, C, Uav, E):
    i, v = x

    # original derivatives
    di_dt = (E * Uav) / L + (1 - Uav) * v / L
    # use i_eff = max(i, 0) to avoid small negative numerical jitter
    i_eff = max(i, 0.0)
    dv_dt = - (1 - Uav) * i_eff / C - v / (R * C)

    # enforce diode-like clamp: if current is zero (or negative)
    # and derivative would push it negative, forbid it (set di_dt = 0)
    if i <= 0.0 and di_dt < 0.0:
        di_dt = 0.0

    return [di_dt, dv_dt]

    
def main():
    # Parameters
    L = 1e-3      # H
    R = 47         # ohms
    C = 50e-6      # F
    Uav = 0.5      # duty ratio
    E = 12         # input voltage [V]

    # Time vector (0 to 3 seconds, 0.01 s step)
    t_ini = 0
    t_stop = 5e-2
    t_step = 1e-5
    t = np.arange(t_ini, t_stop, t_step)
    
    # Initial conditions
    x0 = [0, 0]

    
    # Solve system
    sol = odeint(buck, x0, t, args=(L, R, C, Uav, E))
    i = sol[:, 0]
    v = sol[:, 1]
    
    # Plot
    plt.figure(figsize=(10,5))
    
    plt.subplot(2,1,1)
    plt.plot(t, i)
    plt.xlim(t_ini, t_stop)
    plt.ylabel("Inductor current i(t) [A]")
    plt.grid(True)
    
    plt.subplot(2,1,2)
    plt.plot(t, v)
    plt.xlim(t_ini, t_stop)
    plt.xlabel("Time [s]")
    plt.ylabel("Capacitor voltage v(t) [V]")
    plt.grid(True)
    
    plt.suptitle(f"Buck-Boost Converter Dynamics (Uav = {Uav})")
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()

