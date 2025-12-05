# phono3py --fc3 --fc2 --dim 2 2 1 --br --mesh 17 17 1 --tmin 100 --tmax 500 --tstep 50 -c POSCAR-unit

import h5py
import numpy as np
import matplotlib.pyplot as plt

fname = "kappa-m17171.hdf5"

with h5py.File(fname, "r") as f:
    T = f["temperature"][:]
    kappa = f["kappa"][:]

print("T:", T)
print("kappa shape:", kappa.shape)

kxx = kappa[:, 0]
kyy = kappa[:, 1]

# thickness correction
c_ang = 19.359   # out-of-plane lattice constant (Å)
d_ang = 3.424    # layer thickness (Å)
factor = c_ang / d_ang

k_inplane = 0.5 * (kxx + kyy) * factor # isotropic

data = np.column_stack((T, k_inplane))
np.savetxt("kappa_2d.dat", data,
           header="T(K)  kappa_l(W/mK)", fmt="%.3f")

plt.figure(figsize=(5,4))
plt.plot(T, k_inplane, 'o-', color='r', label=r'$\kappa_{\mathrm{lat}}$')
plt.xlabel('T (K)')
plt.ylabel(r'$\kappa_{\mathrm{lat}}$ (W m$^{-1}$ K$^{-1}$)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('kl_corrected.tif', dpi=1080)
plt.show()
