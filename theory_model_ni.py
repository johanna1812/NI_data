#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.special import binom

# =========================================================
# EXACT QUANTUM FOCK-SPACE SIMULATOR FOR SU(1,1)
# =========================================================

def simulate_su11_pn_curves(r, eta, n_in=0, max_n_plot=5, dim=40):
    """
    Simulates a Single-Mode SU(1,1) Interferometer using exact Fock space operators.
    
    r: Squeezing parameter of each individual squeezer
    eta: Overall transmission efficiency (1 = no loss, 0 = all lost)
    n_in: Input heralded state (0 = Vacuum, 1 = Single Photon, etc.)
    max_n_plot: Highest P(n) to plot
    dim: Truncation dimension of the Fock space (needs to be large enough to hold all photons)
    """
    
    # 1. Define Creation/Annihilation Operators up to 'dim'
    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    adag = a.T
    n_op = adag @ a

    # 2. Build Squeezing Matrices (S1 and S2)
    # S(r) = exp((r/2) * (a^2 - a_dag^2))
    squeezing_generator = 0.5 * r * (a @ a - adag @ adag)
    S1 = expm(squeezing_generator)
    S2 = expm(squeezing_generator)

    # 3. Phase Scan Array
    phases = np.linspace(0, 2 * np.pi, 200)
    P_lossy_results = {k: [] for k in range(max_n_plot + 1)}
    P_click_results = []

    for phi in phases:
        # 4. Phase Shift Operator R(phi)
        R = expm(-1j * phi * n_op)
        
        # 5. Total Interferometer Unitary: U = S2 * R * S1
        U = S2 @ R @ S1
        
        # 6. Prepare Input State (e.g. |0> or |1>)
        psi_in = np.zeros(dim, dtype=complex)
        psi_in[n_in] = 1.0
        
        # 7. Apply Unitary
        psi_out = U @ psi_in
        P_ideal = np.abs(psi_out)**2
        
        # 8. Apply Binomial Loss
        P_lossy = np.zeros(dim)
        for n in range(dim):
            for m in range(n, dim):
                # Probability of having m photons, and exactly n surviving the loss
                P_lossy[n] += binom(m, n) * (eta**n) * ((1 - eta)**(m - n)) * P_ideal[m]
                
        # Store results for plotting
        for k in range(max_n_plot + 1):
            P_lossy_results[k].append(P_lossy[k])
            
        # Click detector probability (1 - P(0 photons))
        P_click_results.append(1.0 - P_lossy[0])
            
    # =========================================================
    # PLOTTING
    # =========================================================
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14})
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cmap = plt.get_cmap('tab10')
    
    for k in range(max_n_plot + 1):
        ax.plot(phases, P_lossy_results[k], lw=2.5, color=cmap(k), label=f"$P({k})$")

    # Plot the binary Click Detector curve
    ax.plot(phases, P_click_results, lw=3, color='black', linestyle='--', label="Click ($n \geq 1$)")

    title_state = "Vacuum ($N_{in}=0$)" if n_in == 0 else f"Heralded ($N_{{in}}={n_in}$)"
    ax.set_title(f"Theoretical SU(1,1) Interference Fringes\nInput: {title_state} | Squeezing: $r={r}$ | Loss: {round((1-eta)*100)}%")
    
    ax.set_xlabel("Internal Phase Shift $\\phi$ (radians)")
    ax.set_ylabel("Probability $P(n)$")
    
    # Format X-axis in multiples of Pi
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['$0$', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$'])
    
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- Experiment 1: The Ideal Case ---
    # r=0.8 (Moderate Squeezing), eta=1.0 (No Loss), input=|0>
    print("Simulating Ideal Case...")
    simulate_su11_pn_curves(r=0.4, eta=1.0, n_in=0, max_n_plot=4)
    
    # --- Experiment 2: The Realistic Case (Your Lab) ---
    # r=0.8, eta=0.4 (60% Loss filling in the valleys), input=|0>
    print("Simulating Lossy Case...")
    simulate_su11_pn_curves(r=0.4, eta=0.4, n_in=0, max_n_plot=4)
    
    # --- Experiment 3: Heralded Single Photon Input ---
    # r=0.8, eta=0.6, input=|1>
    print("Simulating Heralded Input...")
    simulate_su11_pn_curves(r=0.4, eta=0.6, n_in=1, max_n_plot=4)
# %%
