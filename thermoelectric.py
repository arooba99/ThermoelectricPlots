import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib as mpl
import numpy as np

# ================= NEW CONSTANT =================
SHENGBTE_DIR = "/mnt/d/Thermoelectric/WSTe/shengbte"
TAU = 1e-14  # Relaxation time for CRTA (in seconds)
# ================================================

def print_contact_info():
    print("============================================")
    print("       Dr. Christian Tantardini")
    print(" Email: christiantantardini@ymail.com")
    print(" https://scholar.google.com/citations?user=QCZdlUQAAAAJ")
    print("============================================\n")

def load_data(input_file):
    """Load data from the input file into a pandas DataFrame."""
    try:
        data = pd.read_csv(
            input_file, sep=r'\s+', comment='#', header=None,
            names=[
                'Ef[unit]', 'T[K]', 'N[e/uc]', 'DOS(ef)[1/(Ha*uc)]', 'S[V/K]',
                'sigma/tau0[1/(ohm*m*s)]', 'RH[m**3/C]', 'kappae/tau0[W/(m*K*s)]',
                'cv[J/(mol*K)]', 'chi[m**3/mol]'
            ]
        )
        # ========== CRTA MODIFICATION: Multiply by tau ==========
        data['sigma[1/(ohm*m)]'] = data['sigma/tau0[1/(ohm*m*s)]'] * TAU
        data['kappae[W/(m*K)]'] = data['kappae/tau0[W/(m*K*s)]'] * TAU
        # ========================================================
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def detect_ef_unit(input_file):
    """Detect the unit of Ef by inspecting the header or first few lines."""
    unit_conversion = None
    try:
        with open(input_file, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    if 'Ef[Ry]' in line:
                        unit_conversion = 'Ef[Ry]'
                        break
                    elif 'Ef[Ha]' in line:
                        unit_conversion = 'Ef[Ha]'
                        break
                    elif 'Ef[eV]' in line:
                        unit_conversion = 'Ef[eV]'
                        break
        if unit_conversion is None:
            print("Ef unit not specified in the header. Assuming 'Ef[eV]'.\n")
            unit_conversion = 'Ef[eV]'
        else:
            print(f"Detected Ef unit: {unit_conversion}\n")
        return unit_conversion
    except Exception as e:
        print(f"Error detecting Ef unit: {e}")
        sys.exit(1)

def convert_ef_to_eV(data, unit_conversion, rydberg_to_ev, hartree_to_ev):
    """Convert Ef to eV based on the detected unit."""
    try:
        if unit_conversion == 'Ef[Ry]':
            data['Ef[eV]'] = data['Ef[unit]'] * rydberg_to_ev
            print("Converted Ef from Rydberg to eV.\n")
        elif unit_conversion == 'Ef[Ha]':
            data['Ef[eV]'] = data['Ef[unit]'] * hartree_to_ev
            print("Converted Ef from Hartree to eV.\n")
        elif unit_conversion == 'Ef[eV]':
            data['Ef[eV]'] = data['Ef[unit]']
            print("Ef is already in eV. No conversion needed.\n")
        else:
            print("Unknown Ef unit. Cannot convert to eV.")
            sys.exit(1)
        return data
    except Exception as e:
        print(f"Error converting Ef to eV: {e}")
        sys.exit(1)

def subtract_fermi_energy(data, fermi_energy):
    """Subtract Fermi energy from Ef[eV]."""
    try:
        data['Ef[eV]'] -= fermi_energy
        print(f"Fermi energy of {fermi_energy} eV subtracted from Ef[eV].\n")
        print("Ef[eV] after subtraction:", data['Ef[eV]'].min(), data['Ef[eV]'].max())
        return data
    except Exception as e:
        print(f"Error subtracting Fermi energy: {e}")
        sys.exit(1)

def convert_S(data, s_scale_factor):
    """Convert S from V/K to µV/K."""
    try:
        data['S[µV/K]'] = data['S[V/K]'] * s_scale_factor
        print("Converted Seebeck coefficient from V/K to µV/K.\n")
        return data
    except Exception as e:
        print(f"Error converting S to µV/K: {e}")
        sys.exit(1)

def get_available_mu_Ef(data):
    """Retrieve and return sorted unique (mu - E_F) values."""
    unique_mu_Ef = np.sort(data['Ef[eV]'].unique())
    return unique_mu_Ef

def find_closest_mu_Ef(input_value, available_values, tolerance=0.005):
    """Find the closest (mu - E_F) value within the specified tolerance."""
    idx = (np.abs(available_values - input_value)).argmin()
    closest_value = available_values[idx]
    if np.abs(closest_value - input_value) <= tolerance:
        return closest_value
    else:
        return None

def plot_power_factor_vs_mu_Ef(data, temperatures, output_dir, palette):
    """Plot power factor vs (mu - E_F) for chosen T and save data to CSV."""
    plt.figure(figsize=(10, 7))
    
    for idx, T in enumerate(temperatures):
        subset = data[data['T[K]'] == T].copy()
        if subset.empty:
            print(f"Warning: No data found for T = {T} K.")
            continue
        
        # Calculate power factor S²σ (using CRTA-adjusted sigma)
        subset.loc[:, 'Power_Factor[W/m*K²]'] = (subset['S[µV/K]'] * 1e-6)**2 * subset['sigma[1/(ohm*m)]']
        
        # Save data
        csv_filename = os.path.join(output_dir, f'PowerFactor_vs_mu_Ef_T_{T}_K.csv')
        subset.to_csv(csv_filename, index=False)
        print(f"Power factor data for T = {T} K saved to '{csv_filename}'.")

        plt.plot(
            subset['Ef[eV]'], 
            subset['Power_Factor[W/m*K²]'], 
            linestyle='-', 
            linewidth=2, 
            color=palette[idx % len(palette)],
            label=rf'T = {T} K'
        )
    
    if plt.gca().has_data():
        plt.xlabel(r'$\mu - E_F$ (eV)', fontsize=16)
        plt.ylabel(r'Power Factor (W/m·K²)', fontsize=16)
        plt.title(r'Power Factor ($S^2\sigma$) vs $\mu - E_F$', fontsize=18)
        plt.legend(title='Temperature (K)', fontsize=14, title_fontsize=16, 
                 loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plot_filename_png = os.path.join(output_dir, 'PowerFactor_vs_mu_Ef.png')
        plot_filename_pdf = os.path.join(output_dir, 'PowerFactor_vs_mu_Ef.pdf')
        plt.savefig(plot_filename_png, dpi=300, bbox_inches='tight')
        plt.savefig(plot_filename_pdf, bbox_inches='tight')
        plt.close()
        print(f"Power factor plot saved as PNG and PDF in '{output_dir}'.\n")
    else:
        plt.close()
        print("No power factor plots generated due to missing data.\n")

def plot_power_factor_vs_T(data, mu_Ef_values, output_dir, palette, available_mu_Ef, tolerance=0.002):
    """Plot power factor vs T for chosen (mu - E_F) values."""
    plt.figure(figsize=(10, 7))
    for idx, mu_Ef in enumerate(mu_Ef_values):
        subset = data[np.isclose(data['Ef[eV]'], mu_Ef, atol=tolerance)].copy()
        if subset.empty:
            print(f"Warning: No data found for μ - E_F = {mu_Ef:.3f} eV within tolerance.")
            continue
        
        # Calculate power factor
        subset.loc[:, 'Power_Factor[W/m*K²]'] = (subset['S[µV/K]'] * 1e-6)**2 * subset['sigma[1/(ohm*m)]']
        
        # Save data
        csv_filename = os.path.join(output_dir, f'PowerFactor_vs_T_mu_Ef_{mu_Ef:.3f}_eV.csv')
        subset.to_csv(csv_filename, index=False)
        print(f"Power factor data for μ - E_F = {mu_Ef:.3f} eV saved to '{csv_filename}'.")

        plt.plot(
            subset['T[K]'], 
            subset['Power_Factor[W/m*K²]'], 
            linestyle='-', 
            linewidth=2, 
            color=palette[idx % len(palette)],
            label=rf'$\mu - E_F$ = {mu_Ef:.3f} eV'
        )
    
    if plt.gca().has_data():
        plt.xlabel('Temperature (K)', fontsize=16)
        plt.ylabel(r'Power Factor (W/m·K²)', fontsize=16)
        plt.title('Power Factor vs Temperature', fontsize=18)
        plt.legend(title=r'$\mu - E_F$ (eV)', fontsize=14, title_fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plot_filename_png = os.path.join(output_dir, 'PowerFactor_vs_T.png')
        plot_filename_pdf = os.path.join(output_dir, 'PowerFactor_vs_T.pdf')
        plt.savefig(plot_filename_png, dpi=300, bbox_inches='tight')
        plt.savefig(plot_filename_pdf, bbox_inches='tight')
        plt.close()
        print(f"Plot 'PowerFactor_vs_T' saved as PNG and PDF in '{output_dir}'.\n")
    else:
        plt.close()
        print("No plots were generated for Power Factor vs T.\n")

def plot_S_vs_mu_Ef(data, temperatures, output_dir, palette):
    """Plot S vs (mu - E_F) for each chosen T and save data to CSV."""
    plt.figure(figsize=(10, 7))
    for idx, T in enumerate(temperatures):
        subset = data[data['T[K]'] == T].copy()
        if subset.empty:
            print(f"Warning: No data found for T = {T} K.")
            continue

        # Save the data used for this plot
        csv_filename = os.path.join(output_dir, f'S_vs_mu_Ef_T_{T}_K.csv')
        subset.to_csv(csv_filename, index=False)
        print(f"Data for T = {T} K saved to '{csv_filename}'.")

        plt.plot(
            subset['Ef[eV]'], 
            subset['S[µV/K]'], 
            linestyle='-', 
            linewidth=2, 
            color=palette[idx % len(palette)],
            label=rf'T = {T} K'
        )
    if plt.gca().has_data():
        plt.xlabel(r'$\mu - E_F$ (eV)', fontsize=16)
        plt.ylabel(r'$S$ ($\mu$V/K)', fontsize=16)
        plt.title(r'Seebeck Coefficient vs $\mu - E_F$', fontsize=18)
        plt.legend(title='Temperature (K)', fontsize=14, title_fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plot_filename_png = os.path.join(output_dir, 'S_vs_mu_Ef.png')
        plot_filename_pdf = os.path.join(output_dir, 'S_vs_mu_Ef.pdf')
        plt.savefig(plot_filename_png, dpi=300, bbox_inches='tight')
        plt.savefig(plot_filename_pdf, bbox_inches='tight')
        plt.close()
        print(f"Plot 'S_vs_mu_Ef' saved as PNG and PDF in '{output_dir}'.\n")
    else:
        plt.close()
        print("No plots were generated for S vs μ - E_F due to missing data.\n")

def plot_S_vs_T(data, mu_Ef_values, output_dir, palette, available_mu_Ef, tolerance=0.002):
    """Plot S vs T for each chosen (mu - E_F) and save data to CSV."""
    plt.figure(figsize=(10, 7))
    for idx, mu_Ef in enumerate(mu_Ef_values):
        subset = data[np.isclose(data['Ef[eV]'], mu_Ef, atol=tolerance)].copy()
        if subset.empty:
            print(f"Warning: No data found for μ - E_F = {mu_Ef:.3f} eV within tolerance.")
            continue
        s_check = subset[np.isin(subset['T[K]'], [300, 1200])]
        if not s_check.empty:
            # print(f"\nReported values: 322.15 µV/K at 300K, 219.44 µV/K at 1200K for WSTe")
            print(f"\nChecking S values at 300K and 1200K for μ - E_F = {mu_Ef:.3f} eV:")
            print(s_check[['T[K]', 'S[µV/K]']])
            print()

        # Save the data used for this plot
        csv_filename = os.path.join(output_dir, f'S_vs_T_mu_Ef_{mu_Ef:.3f}_eV.csv')
        subset.to_csv(csv_filename, index=False)
        print(f"Data for μ - E_F = {mu_Ef:.3f} eV saved to '{csv_filename}'.")

        plt.plot(
            subset['T[K]'], 
            subset['S[µV/K]'], 
            linestyle='-', 
            linewidth=2, 
            color=palette[idx % len(palette)],
            label=rf'$\mu - E_F$ = {mu_Ef:.3f} eV'
        )
    if plt.gca().has_data():
        plt.xlabel('Temperature (K)', fontsize=16)
        plt.ylabel(r'$S$ ($\mu$V/K)', fontsize=16)
        plt.title('Seebeck Coefficient vs Temperature', fontsize=18)
        plt.legend(title=r'$\mu - E_F$ (eV)', fontsize=14, title_fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plot_filename_png = os.path.join(output_dir, 'S_vs_T.png')
        plot_filename_pdf = os.path.join(output_dir, 'S_vs_T.pdf')
        plt.savefig(plot_filename_png, dpi=300, bbox_inches='tight')
        plt.savefig(plot_filename_pdf, bbox_inches='tight')
        plt.close()
        print(f"Plot 'S_vs_T' saved as PNG and PDF in '{output_dir}'.\n")
    else:
        plt.close()
        print("No plots were generated for S vs T due to missing data.\n")

def plot_sigma_vs_T(data, mu_Ef_values, output_dir, palette, available_mu_Ef, tolerance=0.002):
    """Plot sigma vs T using CRTA-adjusted values."""
    plt.figure(figsize=(10, 7))
    for idx, mu_Ef in enumerate(mu_Ef_values):
        subset = data[np.isclose(data['Ef[eV]'], mu_Ef, atol=tolerance)].copy()
        if subset.empty:
            print(f"Warning: No data found for μ - E_F = {mu_Ef:.3f} eV within tolerance.")
            continue

        # Save the data used for this plot
        csv_filename = os.path.join(output_dir, f'sigma_vs_T_mu_Ef_{mu_Ef:.3f}_eV.csv')
        subset.to_csv(csv_filename, index=False)
        print(f"Data for μ - E_F = {mu_Ef:.3f} eV saved to '{csv_filename}'.")

        plt.plot(
            subset['T[K]'], 
            subset['sigma[1/(ohm*m)]'], 
            linestyle='-', 
            linewidth=2, 
            color=palette[idx % len(palette)],
            label=rf'$\mu - E_F$ = {mu_Ef:.3f} eV'
        )
    if plt.gca().has_data():
        plt.xlabel('Temperature (K)', fontsize=16)
        plt.ylabel(r'$\sigma$ [1/(Ohm·m)]', fontsize=16)
        plt.title('Electrical Conductivity vs Temperature', fontsize=18)
        plt.legend(title=r'$\mu - E_F$ (eV)', fontsize=14, title_fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plot_filename_png = os.path.join(output_dir, 'sigma_vs_T.png')
        plot_filename_pdf = os.path.join(output_dir, 'sigma_vs_T.pdf')
        plt.savefig(plot_filename_png, dpi=300, bbox_inches='tight')
        plt.savefig(plot_filename_pdf, bbox_inches='tight')
        plt.close()
        print(f"Plot 'sigma_vs_T' saved as PNG and PDF in '{output_dir}'.\n")
    else:
        plt.close()
        print("No plots were generated for σ/τ₀ vs T due to missing data.\n")

def plot_kappae_vs_T(data, mu_Ef_values, output_dir, palette, available_mu_Ef, tolerance=0.002):
    """Plot kappae vs T using CRTA-adjusted values."""
    plt.figure(figsize=(10, 7))
    for idx, mu_Ef in enumerate(mu_Ef_values):
        subset = data[np.isclose(data['Ef[eV]'], mu_Ef, atol=tolerance)].copy()
        if subset.empty:
            print(f"Warning: No data found for μ - E_F = {mu_Ef:.3f} eV within tolerance.")
            continue

        # Save the data used for this plot
        csv_filename = os.path.join(output_dir, f'kappae_vs_T_mu_Ef_{mu_Ef:.3f}_eV.csv')
        subset.to_csv(csv_filename, index=False)
        print(f"Data for μ - E_F = {mu_Ef:.3f} eV saved to '{csv_filename}'.")

        plt.plot(
            subset['T[K]'], 
            subset['kappae[W/(m*K)]'], 
            linestyle='-', 
            linewidth=2, 
            color=palette[idx % len(palette)],
            label=rf'$\mu - E_F$ = {mu_Ef:.3f} eV'
        )
    if plt.gca().has_data():
        plt.xlabel('Temperature (K)', fontsize=16)
        plt.ylabel(r'$\kappa_e$ [W/(m·K)]', fontsize=16)
        plt.title('Electronic Thermal Conductivity vs Temperature', fontsize=18)
        plt.legend(title=r'$\mu - E_F$ (eV)', fontsize=14, title_fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plot_filename_png = os.path.join(output_dir, 'kappae_vs_T.png')
        plot_filename_pdf = os.path.join(output_dir, 'kappae_vs_T.pdf')
        plt.savefig(plot_filename_png, dpi=300, bbox_inches='tight')
        plt.savefig(plot_filename_pdf, bbox_inches='tight')
        plt.close()
        print(f"Plot 'kappae_vs_T' saved as PNG and PDF in '{output_dir}'.\n")
    else:
        plt.close()
        print("No plots were generated for κₑ vs T due to missing data.\n")

def get_shengbte_kl():
    """Get K_l values from ShengBTE output files with exact temperature matching."""
    kl_data = {}
    if not os.path.exists(SHENGBTE_DIR):
        print(f"Warning: ShengBTE directory '{SHENGBTE_DIR}' not found. Using κ_l = 0.")
        return kl_data
        
    for t in range(300, 1201, 50):  # Match 300-1200 K step 50
        dir_path = os.path.join(SHENGBTE_DIR, f"T{t}K")
        kappa_file = os.path.join(dir_path, "BTE.kappa_scalar")
        try:
            with open(kappa_file, "r") as f:
                last_line = f.readlines()[-1].strip().split()
                kl_data[t] = float(last_line[1])
        except (FileNotFoundError, IndexError):
            print(f"Warning: No ShengBTE data for T = {t} K. Using κ_l = 0.")
            kl_data[t] = 0.0
    return kl_data

def plot_ZT_vs_T(data, mu_Ef_values, output_dir, palette, available_mu_Ef, kl_data, temps_combined, kl_values, tolerance=0.002):
    """Plot ZT vs T for chosen (μ - E_F) values."""
    plt.figure(figsize=(10, 7))
    
    for idx, mu_Ef in enumerate(mu_Ef_values):
        subset = data[np.isclose(data['Ef[eV]'], mu_Ef, atol=tolerance)].copy()
        if subset.empty:
            print(f"Warning: No data found for μ - E_F = {mu_Ef:.3f} eV within tolerance.")
            continue
        
        # Merge BTP2 and ShengBTE data
        merged = subset.merge(pd.DataFrame({'T[K]': temps_combined, 'κ_l': kl_values}), on='T[K]', how='left')
        merged['κ_l'] = merged['κ_l'].fillna(0.0)  # Use 0 if no ShengBTE data
        
        # Calculate ZT = (S²σT)/(κ_e + κ_l)
        merged['ZT'] = ((merged['S[µV/K]'] * 1e-6)**2 * 
                       merged['sigma/tau0[1/(ohm*m*s)]'] * TAU * 
                       merged['T[K]']) / \
                      (merged['kappae/tau0[W/(m*K*s)]'] * TAU + merged['κ_l'])
        
        # Save data
        csv_filename = os.path.join(output_dir, f'ZT_vs_T_mu_Ef_{mu_Ef:.3f}_eV.csv')
        merged.to_csv(csv_filename, index=False)
        print(f"Data for μ - E_F = {mu_Ef:.3f} eV saved to '{csv_filename}'.")

        plt.plot(merged['T[K]'], merged['ZT'], 
                linestyle='-', linewidth=2,
                color=palette[idx % len(palette)],
                label=rf'$\mu - E_F$ = {mu_Ef:.3f} eV')
    
    if plt.gca().has_data():
        plt.xlabel('Temperature (K)', fontsize=16)
        plt.ylabel(r'$ZT$', fontsize=16)
        plt.title('Thermoelectric Figure of Merit (ZT) vs Temperature', fontsize=18)
        plt.legend(title=r'$\mu - E_F$ (eV)', fontsize=14, title_fontsize=16, 
                 loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ZT_vs_T.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot 'ZT_vs_T' saved as PNG in '{output_dir}'.\n")
    else:
        plt.close()
        print("No ZT plots generated.\n")

def plot_ZT_vs_mu_Ef(data, temperatures, output_dir, palette, kl_data, temps_combined, kl_values):
    """Plot ZT vs μ-E_F for chosen temperatures."""
    plt.figure(figsize=(10, 7))
    
    # Create interpolation function for κ_l
    kl_interp = np.interp(temperatures, temps_combined, kl_values, left=0.0, right=0.0)
    
    for idx, T in enumerate(temperatures):
        subset = data[data['T[K]'] == T].copy()
        if subset.empty:
            print(f"Warning: No data found for T = {T} K.")
            continue
        
        # Get κ_l for this temperature
        current_kl = kl_interp[idx] if idx < len(kl_interp) else 0.0
        
        # Calculate ZT
        subset['ZT'] = ((subset['S[µV/K]'] * 1e-6)**2 * 
                       subset['sigma/tau0[1/(ohm*m*s)]'] * TAU * T) / \
                      (subset['kappae/tau0[W/(m*K*s)]'] * TAU + current_kl)
        
        # Save data
        csv_filename = os.path.join(output_dir, f'ZT_vs_mu_Ef_T_{T}_K.csv')
        subset.to_csv(csv_filename, index=False)
        print(f"Data for T = {T} K saved to '{csv_filename}'.")

        plt.plot(
            subset['Ef[eV]'], 
            subset['ZT'], 
            linestyle='-', 
            linewidth=2, 
            color=palette[idx % len(palette)],
            label=rf'T = {T} K'
        )
    
    if plt.gca().has_data():
        plt.xlabel(r'$\mu - E_F$ (eV)', fontsize=16)
        plt.ylabel(r'$ZT$', fontsize=16)
        plt.title(r'Thermoelectric Figure of Merit (ZT) vs $\mu - E_F$', fontsize=18)
        plt.legend(title='Temperature (K)', fontsize=14, title_fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plot_filename_png = os.path.join(output_dir, 'ZT_vs_mu_Ef.png')
        plot_filename_pdf = os.path.join(output_dir, 'ZT_vs_mu_Ef.pdf')
        plt.savefig(plot_filename_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot 'ZT_vs_mu_Ef' saved as PNG in '{output_dir}'.\n")
    else:
        plt.close()
        print("No ZT vs μ-E_F plots generated.\n")

def get_user_choice(prompt, choices):
    """Utility function to get a valid user choice."""
    while True:
        choice = input(prompt).strip().lower()
        if choice in choices:
            return choice
        else:
            print(f"Invalid choice. Please choose from {choices}.\n")

def get_multiple_values(prompt, available_values, value_type=float, tolerance=0.005):
    """Utility function to get multiple values from the user."""
    print(f"Available (μ - E_F) values: {np.around(available_values, decimals=3)}")
    while True:
        user_input = input(prompt).strip()
        try:
            input_values = [value_type(val) for val in user_input.split(',') if val.strip()]
            matched_values = []
            for val in input_values:
                closest_val = find_closest_mu_Ef(val, available_values, tolerance)
                if closest_val is not None:
                    matched_values.append(closest_val)
                else:
                    print(f"No (μ - E_F) value found within ±{tolerance} eV of {val:.3f} eV.")
            if matched_values:
                return matched_values
            else:
                print("None of the entered values matched within the tolerance. Please try again.\n")
        except ValueError:
            print(f"Invalid input. Please enter comma-separated {value_type.__name__} values.\n")

def main():
    print_contact_info()

    # Conversion factors
    rydberg_to_ev = 13.605698  # Rydberg to eV
    hartree_to_ev = 27.211386  # Hartree to eV
    s_scale_factor = 1e6  # To display S in units of µV/K

    # Set up argument parser for initial data processing
    parser = argparse.ArgumentParser(
        description=r"""
**BTP2 Data Extraction and Plotting Script**

This script processes a space-separated data file containing various physical properties and generates high-quality plots suitable for publication. It offers functionalities such as unit conversion, Fermi energy adjustment, and multiple plotting options.

**Functionalities:**
1. **Unit Conversion:** Automatically detects the unit of Ef (Rydberg, Hartree, or eV) and converts it to eV if necessary.
2. **Fermi Energy Adjustment:** Optionally subtracts the provided Fermi energy from Ef[eV].
3. **Plotting Capabilities:**
   - **Option 1:** Plot S vs (μ - E_F) for chosen temperature(s).
   - **Option 2:** Plot S vs T for chosen (μ - E_F) value(s).
   - **Option 3:** Plot σ vs T for chosen (μ - E_F) value(s).
   - **Option 4:** Plot κₑ vs T for chosen (μ - E_F) value(s).
   - **Option 5:** Plot PF vs (μ - E_F) for chosen temperature(s).
   - **Option 6:** Plot PF vs T for chosen (μ - E_F) value(s).
   - **Option 7:** Plot ZT vs T for chosen (μ - E_F) value(s).
   - **Option 8:** Plot ZT vs (μ - E_F) for chosen temperature(s).
   - **Option 9:** Exit.

**Input File Format:**
- Space-separated text file containing columns:
  ['Ef[unit]', 'T[K]', 'N[e/uc]', 'DOS(ef)[1/(Ha*uc)]', 'S[V/K]',
  'sigma/tau0[1/(ohm*m*s)]', 'RH[m**3/C]', 'kappae/tau0[W/(m*K*s)]',
  'cv[J/(mol*K)]', 'chi[m**3/mol]']
- Lines beginning with '#' are treated as comments and ignored.

**Usage:**
python3 thermoelectric.py -i interpolation.trace
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required argument
    parser.add_argument('-i', '--input', required=True, help='Path to the input file containing the data (e.g., data.txt)')
    args = parser.parse_args()

    # Load and process data
    data = load_data(args.input)
    unit_conversion = detect_ef_unit(args.input)
    data = convert_ef_to_eV(data, unit_conversion, rydberg_to_ev, hartree_to_ev)

    shengbte_kl = get_shengbte_kl()
    temps_btp2     = data['T[K]'].unique()
    temps_combined = np.array(sorted(set(temps_btp2).union(shengbte_kl.keys())))
    kl_values      = np.array([shengbte_kl.get(t, 0.0) for t in temps_combined])

    # Interactive prompt for subtracting Fermi energy
    subtract_choice = get_user_choice("Do you want to subtract the Fermi energy from μ? (yes/no): ", ['yes', 'no'])
    if subtract_choice == 'yes':
        fermi_energy = None
        while fermi_energy is None:
            try:
                fermi_input = input("Enter the Fermi energy in eV to subtract from μ (e.g., 5.0): ").strip()
                fermi_energy = float(fermi_input)
            except ValueError:
                print("Invalid input. Please enter a numerical value for the Fermi energy.\n")
        data = subtract_fermi_energy(data, fermi_energy)
    else:
        print("Proceeding without subtracting Fermi energy from μ.\n")

    # Convert S from V/K to µV/K
    data = convert_S(data, s_scale_factor)

    # Ensure plots directory exists
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)

    # Apply a professional matplotlib style
    sns.set_style("whitegrid")
    palette = sns.color_palette("tab10")
    mpl.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'figure.figsize': (10, 7),
        'lines.linewidth': 2,
        'grid.color': '0.8',
        'grid.linestyle': '--',
        'legend.frameon': False
    })

    # Get available (mu - E_F) values
    available_mu_Ef = get_available_mu_Ef(data)

    # Interactive plotting options
    while True:
        print("Select a plotting option:")
        print("1. Plot S vs (μ - E_F) for chosen T(s)")
        print("2. Plot S vs T for chosen (μ - E_F) value(s)")
        print("3. Plot σ vs T (CRTA)")
        print("4. Plot κₑ vs T (CRTA)")
        print("5. Plot Power Factor vs (μ - E_F)")
        print("6. Plot Power Factor vs T")
        print("7. Plot ZT vs T")
        print("8. Plot ZT vs μ - E_F")
        print("9. Exit")

        choice = get_user_choice("Enter the number corresponding to your choice (1-9): ", ['1', '2', '3', '4', '5', '6', '7', '8', '9'])
        print()

        if choice == '1':
            temperatures = []
            while not temperatures:
                try:
                    temp_input = input("Enter temperature(s) in K separated by commas (e.g., 300,400,500): ").strip()
                    temperatures = [float(val) for val in temp_input.split(',') if val.strip()]
                    if not temperatures:
                        print("No temperatures entered. Please try again.\n")
                except ValueError:
                    print("Invalid input. Please enter numerical values for temperatures.\n")
            plot_S_vs_mu_Ef(data, temperatures, plots_dir, palette)

        elif choice == '2':
            mu_Ef_values = get_multiple_values(
                "Enter (μ - E_F) value(s) in eV separated by commas (e.g., 1.000,2.000,3.000): ",
                available_mu_Ef,
                float,
                tolerance=0.002
            )
            plot_S_vs_T(data, mu_Ef_values, plots_dir, palette, available_mu_Ef, tolerance=0.002)

        elif choice == '3':
            mu_Ef_values = get_multiple_values(
                "Enter (μ - E_F) value(s) in eV separated by commas (e.g., 1.000,2.000,3.000): ",
                available_mu_Ef,
                float,
                tolerance=0.002
            )
            plot_sigma_vs_T(data, mu_Ef_values, plots_dir, palette, available_mu_Ef, tolerance=0.002)

        elif choice == '4':
            mu_Ef_values = get_multiple_values(
                "Enter (μ - E_F) value(s) in eV separated by commas (e.g., 1.000,2.000,3.000): ",
                available_mu_Ef,
                float,
                tolerance=0.002
            )
            plot_kappae_vs_T(data, mu_Ef_values, plots_dir, palette, available_mu_Ef, tolerance=0.002)

        elif choice == '5':
            temperatures = []
            while not temperatures:
                try:
                    temp_input = input("Enter temperature(s) in K (e.g., 300,600,900): ").strip()
                    temperatures = [float(val) for val in temp_input.split(',') if val.strip()]
                except ValueError:
                    print("Invalid input. Please enter numerical values.\n")
            plot_power_factor_vs_mu_Ef(data, temperatures, plots_dir, palette)

        elif choice == '6':
            mu_Ef_values = get_multiple_values(
                "Enter (μ - E_F) value(s) in eV separated by commas: ",
                available_mu_Ef,
                float,
                tolerance=0.002
            )
            plot_power_factor_vs_T(data, mu_Ef_values, plots_dir, palette, available_mu_Ef, tolerance=0.002)

        elif choice == '7':
            mu_Ef_values = get_multiple_values(
                "Enter (μ - E_F) value(s) in eV separated by commas: ",
                available_mu_Ef,
                float,
                tolerance=0.002
            )
            plot_ZT_vs_T(data, mu_Ef_values, plots_dir, palette, available_mu_Ef, 
                        shengbte_kl, temps_combined, kl_values, tolerance=0.002)

        elif choice == '8':
            temperatures = []
            while not temperatures:
                try:
                    temp_input = input("Enter temperature(s) in K separated by commas (e.g., 300,400,500): ").strip()
                    temperatures = [float(val) for val in temp_input.split(',') if val.strip()]
                    if not temperatures:
                        print("No temperatures entered. Please try again.\n")
                except ValueError:
                    print("Invalid input. Please enter numerical values for temperatures.\n")
            plot_ZT_vs_mu_Ef(data, temperatures, plots_dir, palette, 
                            shengbte_kl, temps_combined, kl_values)

        elif choice == '9':
            print("Exiting the script. Goodbye!")
            sys.exit(0)

        continue_choice = get_user_choice("Do you want to perform another action? (yes/no): ", ['yes', 'no'])
        print()
        if continue_choice == 'no':
            print("All selected actions have been completed. Exiting the script. Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    main()