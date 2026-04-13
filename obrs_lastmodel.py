import numpy as np
import matplotlib.pyplot as plt

class CHRRModel:
    def __init__(self):
        # Simulation Settings
        self.dt = 0.01
        self.t_max = 150
        self.steps = int(self.t_max / self.dt)

        # Balanced Parameters to show the transition
        self.p = {
            "D1": 0.4,    # Oncogenic pressure
            "D2": 0.8,    # Immune effectiveness
            "e1": 0.5,    # Base surveillance rate
            "e2": 0.3,    # Surveillance decay
            "a": 0.7,     # Plasticity drive
            "b": 0.5,     # Oncogenic suppression of plasticity
            "thr": 0.4    # The "Detection Threshold" for P
        }

    def odes(self, v, S_prev, I, k):
        P, E, O, S, F = v
        p = self.p

        # 1. SURVEILLANCE AMPLIFICATION (The core of your theory)
        # We use (P/thr)**k so that if P > thr, higher k increases S.
        dS = p["e1"] * ((P / p["thr"]) ** k) - p["e2"] * S_prev

        # 2. ONCOGENIC LOAD
        # Includes your bistability and saturation terms
        dO = (
            p["D1"] * (P * E) 
            + 0.2 * O * (O - 0.3) * (1 - O) 
            - p["D2"] * (S * O)
        )

        # 3. EPIGENETIC ACCESSIBILITY
        dE = 0.4 * P * (1 - E) - 0.2 * (O * E) - 0.1 * E

        # 4. FIBROTIC CONSTRAINT
        dF = 0.5 * I * (1 - F) - 0.3 * P * F - 0.1 * F

        # 5. PROLIFERATIVE PLASTICITY
        dP = (
            p["a"] * E * (1 - F)
            - p["b"] * (O * P)
            - 0.1 * P
        )

        return np.array([dP, dE, dO, dS, dF])

    def step(self, v, S_prev, I, k):
        dt = self.dt
        k1 = self.odes(v, S_prev, I, k)
        k2 = self.odes(v + 0.5 * dt * k1, S_prev, I, k)
        k3 = self.odes(v + 0.5 * dt * k2, S_prev, I, k)
        k4 = self.odes(v + dt * k3, S_prev, I, k)
        v_next = v + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return np.clip(v_next, 0.0, 1.0)

    def simulate(self, k_val, initial_state=None):
        v = np.zeros((self.steps, 5))
        v[0] = initial_state if initial_state is not None else [0.1, 0.3, 0.01, 0.2, 0.1]
        S_prev = v[0, 3]

        for i in range(self.steps - 1):
            t = i * self.dt
            # Injury pulse at t=20
            I_t = 1.5 * np.exp(-0.5 * ((t - 20) / 2)**2)
            v[i + 1] = self.step(v[i], S_prev, I_t, k_val)
            S_prev = v[i + 1, 3]
        return v

if __name__ == "__main__":
    model = CHRRModel()
    k_values = np.linspace(0.5, 3.0, 50)
    
    P_results, O_results, k_plot, colors = [], [], [], []

    for k in k_values:
        # Simulate from healthy start
        data = model.simulate(k)
        final_P, final_O = data[-1, 0], data[-1, 2]
        
        k_plot.append(k)
        P_results.append(final_P)
        O_results.append(final_O)
        colors.append("green" if final_O < 0.2 else "red")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(k_plot, P_results, c=colors, label="Plasticity (P)", alpha=0.6)
    plt.scatter(k_plot, O_results, c=colors, marker='x', label="Oncogenesis (O)", alpha=0.6)
    plt.axvline(1.0, color='black', linestyle='--', alpha=0.3, label="Linear Threshold")
    
    plt.title("CHRR Theory: Stability via Superlinear Surveillance")
    plt.xlabel("Surveillance Scaling (k)")
    plt.ylabel("Steady State Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
