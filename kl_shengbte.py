#!/usr/bin/env python3

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def extract_kappa_data():
    """Extract thermal conductivity values from all temperature directories."""
    kappa_data = []
    
    temp_dirs = glob.glob('T*K')
    
    for dir_path in temp_dirs:
        # Extract temperature from directory name
        temp = int(dir_path.split('T')[1].split('K')[0])
        
        kappa_file = os.path.join(dir_path, 'BTE.kappa_scalar')
        
        if os.path.exists(kappa_file):
            try:
                # Read last line and get converged kappa value
                with open(kappa_file, 'r') as f:
                    last_line = f.readlines()[-1].strip()
                kappa = float(last_line.split()[1])  # Second column
                kappa_data.append((temp, kappa))
            except Exception as e:
                print(f"Error processing {kappa_file}: {str(e)}")
    
    kappa_data.sort(key=lambda x: x[0])
    return np.array(kappa_data)

def plot_thermal_conductivity(data):
    """Plot thermal conductivity vs temperature."""
    plt.figure(figsize=(8, 6))
    
    temperatures = data[:, 0]
    kappa_values = data[:, 1] * 1e5  # Original values are in 1e-5 W/m-K
    
    # Lattice constant c (Å) used in ShengBTE cell
    c_ang = 22.96472 # z-coordinate (Å)
    d_ang = 2.991  # layer thickness (Å)
    factor = c_ang / d_ang

    kappa_values = data[:, 1] * 1e5 * factor

    corrected = np.column_stack((temperatures, kappa_values))
    np.savetxt(
        'kappa_2d.dat',
        corrected,
        header='Temperature(K) kappa_l(W/m-K) with thickness correction (d = 2.991 Å)',
        fmt=['%d', '%.8e']
    )
    
    plt.plot(temperatures, kappa_values, 
             marker='o', linestyle='-', 
             color='#2ecc71', linewidth=2, 
             markersize=8, markeredgecolor='black')
    
    plt.xlabel('Temperature (K)', fontsize=16)
    plt.ylabel('Lattice Thermal Conductivity (W/m-K)', fontsize=16)
    
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    if np.max(kappa_values) < 1:
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig('thermal_conductivity.png', dpi=1080)
    plt.show()

if __name__ == '__main__':
    data = extract_kappa_data()
    
    if len(data) > 0:
        np.savetxt('kappa_3d.dat', data, 
                   header='Temperature(K) Thermal_Conductivity(1e-5_W/m-K)', 
                   fmt=['%d', '%.8e'])

        plot_thermal_conductivity(data)
    else:
        print("No valid data found. Check directory structure and data files.")
