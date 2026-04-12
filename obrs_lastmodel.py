import numpy as np
import matplotlib.pyplot as plt


class OBRSModel:
    def __init__(self):

        # =========================
        # SIMULATION SETTINGS
        # =========================
        self.dt = 0.01
        self.t_max = 150
        self.steps = int(self.t_max / self.dt)

        # =========================
        # PARAMETRI (TUNED)
        # =========================
        self.p = {
            "D1": 0.65,
            "D2": 0.35,   # ↓ indebolita immunità (CRUCIALE)
            "e1": 0.65,
            "e2": 0.25,
            "a": 0.85,
            "b": 0.60
        }

    # =========================
    # SYSTEM DYNAMICS
    # =========================
    def odes(self, v, S_prev, I, k):

        P, E, O, S, F = v
        p = self.p

        # Sorveglianza
        dS = p["e1"] * (P ** k) - p["e2"] * S_prev

        # Oncogenesi (NON LINEARE FORTE)
        dO = (
            p["D1"] * (P * E + 0.25 * P**2)
            + 0.35 * O * (O - 0.3) * (1 - O)   # bistabilità
            + 0.2 * (O**2) / (1 + O**2)        # saturazione
            - p["D2"] * (S * O)
        )

        # Epigenetica
        dE = 0.5 * P * (1 - E) - 0.35 * (O * E) - 0.05 * E

        # Fibrosi
        dF = 0.6 * I * (1 - F) - 0.35 * P * F - 0.1 * F

        # Plasticità
        dP = (
            p["a"] * E * (1 - F)
            - p["b"] * (O * P)
            - 0.1 * P
        )

        return np.array([dP, dE, dO, dS, dF])

    # =========================
    # RK4
    # =========================
    def step(self, v, S_prev, I, k):

        dt = self.dt

        k1 = self.odes(v, S_prev, I, k)
        k2 = self.odes(v + 0.5 * dt * k1, S_prev, I, k)
        k3 = self.odes(v + 0.5 * dt * k2, S_prev, I, k)
        k4 = self.odes(v + dt * k3, S_prev, I, k)

        v_next = v + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return np.clip(v_next, 0.0, 1.0)

    # =========================
    # SIMULATION (FIXED)
    # =========================
    def simulate(self, k_val, initial_state=None):

        v = np.zeros((self.steps, 5))

        if initial_state is None:
            v[0] = [0.05, 0.3, 0.01, 0.2, 0.1]
        else:
            v[0] = initial_state

        S_prev = v[0, 3]

        for i in range(self.steps - 1):

            t = i * self.dt
            I_t = 2.0 * np.exp(-0.5 * ((t - 30) / 3)**2)

            v_next = self.step(v[i], S_prev, I_t, k_val)
            v[i + 1] = v_next
            S_prev = v[i + 1, 3]

        return v

    # =========================
    # CLASSIFICATION
    # =========================
    def classify(self, data):

        tail = data[int(0.8 * len(data)):]

        P = tail[:, 0]
        O = tail[:, 2]

        p_std = np.std(P)
        o_mean = np.mean(O)

        if o_mean > 0.65:
            return "Malignant", "red"

        if p_std > 0.02:
            return "Oscillatory", "orange"

        return "Stable OBRS", "green"


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    model = OBRSModel()

    k_values = np.linspace(0.5, 2.5, 60)

    k_plot = []
    P_final = []
    O_final = []
    colors = []

    # =========================
    # TEST BISTABILITY
    # =========================
    k_test = 1.5

    data_low = model.simulate(k_test)
    data_high = model.simulate(k_test, initial_state=[0.05, 0.3, 0.7, 0.2, 0.1])

    print("LOW INIT:", data_low[-1])
    print("HIGH INIT:", data_high[-1])

    # =========================
    # BIFURCATION
    # =========================
    for k in k_values:

        # LOW INIT
        data1 = model.simulate(k)
        label1, color1 = model.classify(data1)

        k_plot.append(k)
        P_final.append(data1[-1, 0])
        O_final.append(data1[-1, 2])
        colors.append(color1)

        # HIGH INIT
        data2 = model.simulate(k, initial_state=[0.05, 0.3, 0.7, 0.2, 0.1])
        label2, color2 = model.classify(data2)

        k_plot.append(k)
        P_final.append(data2[-1, 0])
        O_final.append(data2[-1, 2])
        colors.append(color2)

    # =========================
    # BIFURCATION PLOT
    # =========================
    plt.figure(figsize=(12, 6))

    plt.scatter(k_plot, P_final, c=colors, s=25, label="Plasticity (P)")
    plt.scatter(k_plot, O_final, c=colors, marker="x", s=25, label="Oncogenesis (O)")

    plt.axvline(1.0, linestyle="--", color="black", alpha=0.4)

    plt.title("OBRS Bifurcation Diagram (Bistability Regime)")
    plt.xlabel("k (Surveillance Scaling)")
    plt.ylabel("Final State")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.show()

    # =========================
    # PHASE SPACE
    # =========================
    plt.figure(figsize=(8, 6))

    for k in [0.7, 1.2, 2.0]:

        data = model.simulate(k)

        P = data[:, 0]
        O = data[:, 2]

        label, color = model.classify(data)

        plt.plot(P, O, color=color, lw=1.5, label=f"k={k} ({label})")
        plt.scatter(P[-1], O[-1], color=color, s=50)

    plt.title("Phase Space OBRS (P vs O)")
    plt.xlabel("Plasticity (P)")
    plt.ylabel("Oncogenesis (O)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.show()