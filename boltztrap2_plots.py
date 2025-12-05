import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TAU = 1e-14  # relaxation time in seconds

df = pd.read_csv("interpolation.trace", sep='\s+', header=0)

fermi_energy = None
with open("OUTCAR") as f:
    for line in f:
        if "E-fermi" in line:
            parts = line.split()
            fermi_energy = float(parts[2])
            break

if fermi_energy is None:
    raise ValueError("Fermi energy not found in OUTCAR")

temperatures = [300, 400, 500]

fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

for T_target in temperatures:
    subset = df[df['T[K]'] == T_target]
    mu_diff_Ef = subset['Ef[Ry]'] * 13.605693 - fermi_energy
    mask = (mu_diff_Ef >= -1) & (mu_diff_Ef <= 1)

    sigma = subset.loc[mask, 'sigma/tau0[1/(ohm*m*s)]'] * TAU
    kappae = subset.loc[mask, 'kappae/tau0[W/(m*K*s)]'] * TAU
    S = subset.loc[mask, 'S[V/K]']

    PF = (S**2) * sigma

    axs[0, 0].plot(mu_diff_Ef[mask], PF, label=f'{T_target} K')
    axs[0, 1].plot(mu_diff_Ef[mask], sigma, label=f'{T_target} K')
    axs[1, 0].plot(mu_diff_Ef[mask], kappae, label=f'{T_target} K')
    axs[1, 1].plot(mu_diff_Ef[mask], S * 1e6, label=f'{T_target} K')  # µV/K

axs[0, 0].set_ylabel('Power Factor (W/(m·K²))')
axs[0, 1].set_ylabel(r'σ (1/(Ω·m))')

axs[1, 0].set_ylabel(r'κₑ (W/(m·K))')
axs[1, 0].set_xlabel(r'$\mu - E_F$ (eV)')

axs[1, 1].set_ylabel('Seebeck (µV/K)')
axs[1, 1].set_xlabel(r'$\mu - E_F$ (eV)')

for ax in axs.flat:
    ax.grid(True)
    ax.legend()

plt.tight_layout()
# plt.show()
plt.savefig('therm.tif', dpi=1080)
