# vi /home/kanwal/.config/matplotlib/matplotlibrc
# sed -i '1s/^# //' interpolation.trace

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TAU = 1e-14

df = pd.read_csv("interpolation.trace", sep='\s+', header=0)

kl_data = np.loadtxt("kappa_2d.dat")     # T(K)  kappa_l(W/mK)
kl_T = kl_data[:, 0]
kl_val = kl_data[:, 1]

from scipy.interpolate import interp1d
kl_interp = interp1d(kl_T, kl_val, kind="linear", fill_value="extrapolate")

# Read fermi energy from OUTCAR
fermi_energy = None
with open("OUTCAR") as f:
    for line in f:
        if "E-fermi" in line:
            parts = line.split()
            fermi_energy = float(parts[2])  # 3rd token is Fermi energy in eV
            break

if fermi_energy is None:
    raise ValueError("Fermi energy not found in OUTCAR")

temperatures = [300, 400, 500]

plt.figure(figsize=(8,6))
for T_target in temperatures:
    subset = df[df['T[K]'] == T_target]

    # Convert Ef from Ry to eV and subtract Fermi energy
    mu_diff_Ef = subset['Ef[Ry]'] * 13.605693 - fermi_energy
    print(fermi_energy)
    # Limit mu_diff_Ef to between -1 and 1 eV
    mask = (mu_diff_Ef >= -1) & (mu_diff_Ef <= 1)

    sigma = subset.loc[mask, 'sigma/tau0[1/(ohm*m*s)]'] * TAU
    kappae = subset.loc[mask, 'kappae/tau0[W/(m*K*s)]'] * TAU
    S = subset.loc[mask, 'S[V/K]']
    kappa_l = float(kl_interp(T_target))
    ZT = (S**2) * sigma * T_target / (kappae + kappa_l)

    plt.plot(mu_diff_Ef[mask], ZT, label=f'T = {T_target} K')

plt.xlabel(r'$\mu - E_F$ (eV)')
plt.ylabel('ZT')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('ZT_corrected.tif', dpi=1080)
plt.show()