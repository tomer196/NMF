import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


data_path = '/Users/tomer/private/NMF/data/synth1/'
out_folder = "/Users/tomer/private/NMF/out_parameterized_nmf/synth1"
os.makedirs(out_folder, exist_ok=True)

def analyze_data():
    # --- UV-vis Data Loading ---
    df_An_full = pd.read_csv(data_path + "An.csv", header=None)
    uv_wavelengths = df_An_full.iloc[1:, 0].values.astype(float) # Skip 1st row, take 1st col for wavelengths
    An_np = df_An_full.iloc[1:, 1:].values.astype(float)          # Skip 1st row & 1st col for matrix data

    df_W_true_full = pd.read_csv(data_path + "W_true.csv", header=None)
    W_true_np = df_W_true_full.iloc[1:, 1:].values.astype(float)  # Skip 1st row & 1st col for matrix data
    # W_true_np /= 100

    df_H_true = pd.read_csv(data_path + "H_true.csv", header=None)
    try:
        H_true_np = df_H_true.iloc[1:, 1:].values.astype(float)
    except:
        H_true_np = df_H_true.iloc[1:, 2:].values.astype(float)
    

    # --- Raman Data Loading ---
    df_An_raman_full = pd.read_csv(data_path + "An_raman.csv", header=None)
    raman_wavelengths = df_An_raman_full.iloc[1:, 0].values.astype(float) # Skip 1st row, take 1st col
    An_raman_np = df_An_raman_full.iloc[1:, 1:].values.astype(float) # Skip 1st row & 1st col

    df_W_true_raman_full = pd.read_csv(data_path + "W_true_raman.csv", header=None)
    # W_true_raman_wavelengths = df_W_true_raman_full.iloc[1:, 0].values.astype(float) # If needed, skip 1st row
    W_true_raman_np = df_W_true_raman_full.iloc[1:, 1:].values.astype(float) # Skip 1st row & 1st col

    # df_H_true_raman_full = pd.read_csv(data_path + "H_true_raman.csv", header=None)
    # H_true_raman_np = df_H_true_raman_full.iloc[1:, 1:].values.astype(float) # Skip 1st row & 1st col
    H_true_raman_np = H_true_np.copy() 

    # Convert data to PyTorch tensors
    An = torch.from_numpy(An_np).float()
    W_true = torch.from_numpy(W_true_np).float()
    H_true = torch.from_numpy(H_true_np).float()

    An_raman = torch.from_numpy(An_raman_np).float()
    W_true_raman = torch.from_numpy(W_true_raman_np).float()
    H_true_raman = torch.from_numpy(H_true_raman_np).float()


    print("--- UV-vis Data Shapes (Original NumPy shapes for reference) ---")
    print(f"An_np: {An_np.shape}")
    print(f"W_true_np: {W_true_np.shape}")
    print(f"H_true_np: {H_true_np.shape}")
    print(f"UV Wavelengths: {uv_wavelengths.shape}")

    print("\n--- Raman Data Shapes (Original NumPy shapes for reference) ---")
    print(f"An_raman_np: {An_raman_np.shape}")
    print(f"W_true_raman_np: {W_true_raman_np.shape}")
    print(f"H_true_raman_np: {H_true_raman_np.shape}")
    print(f"Raman Wavelengths: {raman_wavelengths.shape}")

    print("\n--- PyTorch Tensor Shapes ---")
    print(f"An: {An.shape}")
    print(f"W_true: {W_true.shape}")
    print(f"H_true: {H_true.shape}")
    print(f"An_raman: {An_raman.shape}")
    print(f"W_true_raman: {W_true_raman.shape}")
    print(f"H_true_raman: {H_true_raman.shape}")


    # --- UV-vis Data Exploration ---
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(An.cpu().numpy(), aspect='auto', cmap='viridis') # Use .cpu().numpy() for plotting
    plt.colorbar(label='Absorbance')
    plt.title('An (UV-vis Absorbance Matrix)')
    plt.xlabel('Time Index')
    plt.ylabel('Wavelength Index')

    plt.subplot(2, 2, 2)
    for i in range(W_true.shape[1]):
        plt.plot(uv_wavelengths, W_true[:, i].cpu().numpy(), label=f'Compound {i+1}') # Use .cpu().numpy()
    plt.title('W_true (UV-vis Spectra)')
    plt.xlabel('Wavelength (eV)')
    plt.ylabel('Absorbance')
    if W_true.shape[1] <= 10: # Add legend if not too many compounds
        plt.legend()

    plt.subplot(2, 2, 3)
    # import ipdb; ipdb.set_trace()
    for i in range(H_true.shape[0]):
        plt.plot(H_true[i, :].cpu().numpy(), label=f'Compound {i+1}') # Use .cpu().numpy()
    plt.title('H_true (Concentration Profiles)')
    plt.xlabel('Time Index')
    plt.ylabel('Concentration')
    if H_true.shape[0] <= 10: # Add legend if not too many compounds
        plt.legend()

    # Verify An = W_true @ H_true
    An_calculated = W_true @ H_true # PyTorch matmul
    uv_noise = An - An_calculated

    plt.subplot(2, 2, 4)
    plt.imshow(uv_noise.cpu().numpy(), aspect='auto', cmap='coolwarm', vmin=-torch.max(torch.abs(uv_noise)).item(), vmax=torch.max(torch.abs(uv_noise)).item()) # Use .cpu().numpy() and torch utils
    plt.colorbar(label='Difference (Noise)')
    plt.title('Noise (An - W_true @ H_true)')
    plt.xlabel('Time Index')
    plt.ylabel('Wavelength Index')

    plt.tight_layout()
    plt.suptitle('UV-vis Data Exploration and Verification', fontsize=16, y=1.02)
    plt.savefig(os.path.join(out_folder, 'uv_vis_data_exploration.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot: An, W_true @ H_true, and uv_noise ---
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(An.cpu().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label='Absorbance')
    plt.title('An (UV-vis Absorbance Matrix)')
    plt.xlabel('Time Index')
    plt.ylabel('Wavelength Index')

    plt.subplot(1, 3, 2)
    plt.imshow(An_calculated.cpu().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label='Absorbance')
    plt.title('W_true @ H_true (Reconstructed)')
    plt.xlabel('Time Index')
    plt.ylabel('Wavelength Index')

    plt.subplot(1, 3, 3)
    plt.imshow(uv_noise.cpu().numpy(), aspect='auto', cmap='coolwarm', 
            vmin=-torch.max(torch.abs(uv_noise)).item(), vmax=torch.max(torch.abs(uv_noise)).item())
    plt.colorbar(label='Difference (Noise)')
    plt.title('uv_noise (An - W_true @ H_true)')
    plt.xlabel('Time Index')
    plt.ylabel('Wavelength Index')

    plt.tight_layout()
    plt.suptitle('UV-vis: An, Reconstruction, and Noise', fontsize=16, y=1.05)
    plt.savefig(os.path.join(out_folder, 'uv_vis_An_WtrueHtrue_noise.png'), dpi=300, bbox_inches='tight')
    plt.close()


    # --- Raman Data Exploration ---
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(An_raman.cpu().numpy(), aspect='auto', cmap='viridis') # Use .cpu().numpy()
    plt.colorbar(label='Intensity')
    plt.title('An_raman (Raman Intensity Matrix)')
    plt.xlabel('Time Index')
    plt.ylabel('Wavelength Index')

    plt.subplot(2, 2, 2)
    for i in range(W_true_raman.shape[1]):
        plt.plot(raman_wavelengths, W_true_raman[:, i].cpu().numpy(), label=f'Compound {i+1}') # Use .cpu().numpy()
    plt.title('W_true_raman (Raman Spectra)')
    plt.xlabel('Raman Shift (cm^-1 or similar)') # Adjust label based on actual units
    plt.ylabel('Intensity')
    if W_true_raman.shape[1] <= 10:
        plt.legend()

    plt.subplot(2, 2, 3)
    for i in range(H_true_raman.shape[0]):
        plt.plot(H_true_raman[i, :].cpu().numpy(), label=f'Compound {i+1}') # Use .cpu().numpy()
    plt.title('H_true_raman (Binarized Concentration Profiles)')
    plt.xlabel('Time Index')
    plt.ylabel('Presence (0 or 1)')
    if H_true_raman.shape[0] <= 10:
        plt.legend()

    # Verify An_raman = W_true_raman @ H_true_raman
    An_raman_calculated = W_true_raman @ H_true_raman # PyTorch matmul
    raman_noise = An_raman - An_raman_calculated

    plt.subplot(2, 2, 4)
    plt.imshow(raman_noise.cpu().numpy(), aspect='auto', cmap='coolwarm', vmin=-torch.max(torch.abs(raman_noise)).item(), vmax=torch.max(torch.abs(raman_noise)).item()) # Use .cpu().numpy() and torch utils
    plt.colorbar(label='Difference (Noise)')
    plt.title('Noise (An_raman - W_true_raman @ H_true_raman)')
    plt.xlabel('Time Index')
    plt.ylabel('Wavelength Index')

    plt.tight_layout()
    plt.suptitle('Raman Data Exploration and Verification', fontsize=16, y=1.02)
    plt.savefig(os.path.join(out_folder, 'raman_data_exploration.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot: An_raman, W_true_raman @ H_true_raman, and raman_noise ---
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(An_raman.cpu().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label='Intensity')
    plt.title('An_raman (Raman Intensity Matrix)')
    plt.xlabel('Time Index')
    plt.ylabel('Wavelength Index')

    plt.subplot(1, 3, 2)
    plt.imshow(An_raman_calculated.cpu().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar(label='Intensity')
    plt.title('W_true_raman @ H_true_raman (Reconstructed)')
    plt.xlabel('Time Index')
    plt.ylabel('Wavelength Index')

    plt.subplot(1, 3, 3)
    plt.imshow(raman_noise.cpu().numpy(), aspect='auto', cmap='coolwarm', 
            vmin=-torch.max(torch.abs(raman_noise)).item(), vmax=torch.max(torch.abs(raman_noise)).item())
    plt.colorbar(label='Difference (Noise)')
    plt.title('raman_noise (An_raman - W_true_raman @ H_true_raman)')
    plt.xlabel('Time Index')
    plt.ylabel('Wavelength Index')

    plt.tight_layout()
    plt.suptitle('Raman: An_raman, Reconstruction, and Noise', fontsize=16, y=1.05)
    plt.savefig(os.path.join(out_folder, 'raman_vis_An_WtrueHtrue_noise.png'), dpi=300, bbox_inches='tight')
    plt.close()


    # --- Compare H_true from UV-vis and H_true_raman from Raman ---
    num_compounds = H_true.shape[0] # Assuming H_true and H_true_raman have profiles for the same number of compounds
    time_uv = np.arange(H_true.shape[1]) # Time points for UV-vis H_true (e.g., 0, 1, ..., 99)

    # H_true_raman has 50 timepoints, representing every two time steps of H_true's 100 timepoints
    # So, Raman timepoints correspond to 0, 2, 4, ..., 98 of the UV-vis time scale
    time_raman = np.arange(0, H_true.shape[1], H_true.shape[1] // H_true_raman.shape[1])

    # Determine the number of rows and columns for subplots
    # Aim for a somewhat square layout, e.g., if 3 compounds, 3 rows, 1 col. If 4, 2 rows, 2 cols.
    if num_compounds <= 0:
        print("No compounds to plot for H comparison.")
    else:
        # Simple layout: one row per compound if few, or adapt for more
        if num_compounds <= 4:
            n_rows_h_compare = num_compounds
            n_cols_h_compare = 1
        else:
            n_cols_h_compare = 2 # Or 3, depending on preference
            n_rows_h_compare = (num_compounds + n_cols_h_compare - 1) // n_cols_h_compare


        plt.figure(figsize=(6 * n_cols_h_compare, 4 * n_rows_h_compare))

        for i in range(num_compounds):
            plt.subplot(n_rows_h_compare, n_cols_h_compare, i + 1)
            plt.plot(time_uv, H_true[i, :].cpu().numpy(), label=f'H_true (UV-vis) - Comp {i+1}', marker='o', linestyle='-') # Use .cpu().numpy()
            plt.plot(time_raman, H_true_raman[i, :].cpu().numpy(), label=f'H_true_raman (Raman) - Comp {i+1}', marker='x', linestyle='--') # Use .cpu().numpy()
            plt.title(f'Concentration Profiles for Compound {i+1}')
            plt.xlabel('Time Index (UV-vis Scale)')
            plt.ylabel('Concentration / Presence')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.suptitle('Comparison of H_true (UV-vis) and H_true_raman (Raman)', fontsize=16, y=1.02)
        plt.savefig(os.path.join(out_folder, 'h_true_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    analyze_data()