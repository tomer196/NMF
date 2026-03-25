import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim


data_path = "/Users/tomer/private/NMF/Li2S-main/data/"
data_analysis_folder = "/Users/tomer/private/NMF/Li2S-main/out1/data_analysis/"
opt_folder = "/Users/tomer/private/NMF/Li2S-main/out1/opt/"

# Create output folders if they don't exist
os.makedirs(data_analysis_folder, exist_ok=True)
os.makedirs(opt_folder, exist_ok=True)

# --- UV-vis Data Loading ---
df_An_full = pd.read_csv(data_path + "An.csv", header=None)
uv_wavelengths = df_An_full.iloc[1:, 0].values.astype(float) # Skip 1st row, take 1st col for wavelengths
An_np = df_An_full.iloc[1:, 1:].values.astype(float)          # Skip 1st row & 1st col for matrix data

df_W_true_full = pd.read_csv(data_path + "W_true.csv", header=None)
# Assuming first col of W_true is also wavelengths and has same number of points as An
# W_true_wavelengths = df_W_true_full.iloc[1:, 0].values.astype(float) # If needed, skip 1st row
W_true_np = df_W_true_full.iloc[1:, 1:].values.astype(float)  # Skip 1st row & 1st col for matrix data
W_true_np /= 100

df_H_true_full = pd.read_csv(data_path + "H_true.csv", header=None)
H_true_np = df_H_true_full.iloc[1:, 1:].values.astype(float)  # Skip 1st row & 1st col for matrix data

# --- Raman Data Loading ---
df_An_raman_full = pd.read_csv(data_path + "An_raman.csv", header=None)
raman_wavelengths = df_An_raman_full.iloc[1:, 0].values.astype(float) # Skip 1st row, take 1st col
An_raman_np = df_An_raman_full.iloc[1:, 1:].values.astype(float) # Skip 1st row & 1st col

df_W_true_raman_full = pd.read_csv(data_path + "W_true_raman.csv", header=None)
# W_true_raman_wavelengths = df_W_true_raman_full.iloc[1:, 0].values.astype(float) # If needed, skip 1st row
W_true_raman_np = df_W_true_raman_full.iloc[1:, 1:].values.astype(float) # Skip 1st row & 1st col

df_H_true_raman_full = pd.read_csv(data_path + "H_true_raman.csv", header=None)
H_true_raman_np = df_H_true_raman_full.iloc[1:, 1:].values.astype(float) # Skip 1st row & 1st col

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
plt.savefig(os.path.join(data_analysis_folder, 'uv_vis_data_exploration.png'), dpi=300, bbox_inches='tight')
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
plt.savefig(os.path.join(data_analysis_folder, 'uv_vis_An_WtrueHtrue_noise.png'), dpi=300, bbox_inches='tight')
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
plt.savefig(os.path.join(data_analysis_folder, 'raman_data_exploration.png'), dpi=300, bbox_inches='tight')
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
plt.savefig(os.path.join(data_analysis_folder, 'raman_vis_An_WtrueHtrue_noise.png'), dpi=300, bbox_inches='tight')
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
    plt.savefig(os.path.join(data_analysis_folder, 'h_true_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


# --- Optimization Problem ---
# Objective: ||An - W @ H_opt||^2 + ||An_raman - sigmoid(W_raman_opt) @ H_opt_subsampled||^2
# Variables to optimize:
# W_opt: UV-vis spectra (N_uv_wl, k)
# H_opt: Concentrations (k, N_time)
# W_raman_opt: Raman spectra, pre-sigmoid (N_raman_wl, k)

print("\n--- Starting Optimization (PyTorch) ---")

# Define threshold activation function
def threshold_activation(x, epsilon=0.01, sharpness=100.0):
    # Smooth approximation of step function using tanh
    # When concentration (x) > epsilon, output approaches 1
    # When concentration (x) < epsilon, output approaches 0
    return 0.5 * (1 + torch.tanh(sharpness * (x - epsilon)))

# Parameters
k = H_true.shape[0]  # Number of components
N_uv_wl, N_time = An.shape
N_raman_wl, N_time_raman = An_raman.shape

# Add parameters for threshold activation
epsilon = 0.01  # Threshold for concentration
sharpness = 100.0  # Controls how sharp the transition is

# Initialization
# Small positive random values for W_opt and H_opt
# Random values for W_raman_opt
torch.manual_seed(42) # for reproducibility with PyTorch
W_opt = torch.rand(N_uv_wl, k, dtype=torch.float32) * 0.01
W_opt.requires_grad_(True)
H_opt = torch.rand(k, N_time, dtype=torch.float32) * 0.01
H_opt.requires_grad_(True)
# Initialize W_raman_opt with small random values
W_raman_opt = torch.randn(N_raman_wl, k, dtype=torch.float32) * 0.01
W_raman_opt.requires_grad_(True)

# Subsampling indices for H_opt
# H_opt_subsampled should have N_time_raman columns
subsample_factor = N_time // N_time_raman
subsample_indices = np.arange(0, N_time, subsample_factor) # Kept as numpy array for indexing
if len(subsample_indices) > N_time_raman:
    subsample_indices = subsample_indices[:N_time_raman]
elif len(subsample_indices) < N_time_raman:
    # This case should ideally not happen if data is structured as assumed
    # For robustness, one might pad or adjust, but here we assume exact or near exact match
    print(f"Warning: Subsample indices length {len(subsample_indices)} vs N_time_raman {N_time_raman}")
    # Fallback if subsampling logic is imperfect for some N_time, N_time_raman combinations
    if N_time_raman <= len(subsample_indices):
         subsample_indices = subsample_indices[:N_time_raman]
    else: # N_time_raman > len(subsample_indices) - this is problematic
        raise ValueError("Subsampling results in too few time points for Raman data.")


# Gradient Descent Parameters
learning_rate = 0.1 # Adjusted learning rate; may need further tuning for SGD or Adam
iterations = 500000 # Increased iterations
loss_history = []

# Optimizer
# Using SGD, Adam could be an alternative e.g. optim.Adam([W_opt, H_opt, W_raman_opt], lr=0.001)
optimizer = optim.SGD([W_opt, H_opt, W_raman_opt], lr=learning_rate)


print(f"Optimizing for k={k} components.")
print(f"Shapes: W_opt={W_opt.shape}, H_opt={H_opt.shape}, W_raman_opt={W_raman_opt.shape}")
print(f"An={An.shape}, An_raman={An_raman.shape}")
print(f"H_opt will be subsampled to ({H_opt.shape[0]}, {N_time_raman}) for the Raman term.")


for i in range(iterations):
    optimizer.zero_grad() # Clear previous gradients

    # 1. Subsample H_opt
    H_opt_sub = H_opt[:, subsample_indices]

    # 2. Forward pass
    # UV-vis term
    An_pred = W_opt @ H_opt
    E1 = An - An_pred
    loss1 = torch.sum(E1**2)

    # Raman term
    # Apply threshold activation instead of sigmoid
    W_raman_activated = threshold_activation(W_raman_opt, epsilon=epsilon, sharpness=sharpness)
    An_raman_pred = W_raman_activated @ H_opt_sub
    E2 = An_raman - An_raman_pred
    loss2 = torch.sum(E2**2)

    total_loss = loss1 + loss2
    loss_history.append(total_loss.item())

    if (i + 1) % 500 == 0:
        print(f"Iteration {i+1}/{iterations}, Loss: {total_loss.item():.4f} (L1: {loss1.item():.4f}, L2: {loss2.item():.4f})")

    # 3. Gradients (Automatic differentiation)
    total_loss.backward()

    # 4. Update parameters
    optimizer.step()

    # 5. Enforce positivity for W_opt and H_opt and normalization for H_opt
    with torch.no_grad():
        W_opt.data.clamp_(min=0)
        H_opt.data.clamp_(min=0)
        H_opt.data = H_opt.data / (torch.sum(H_opt.data, dim=0, keepdim=True) + 1e-9)

    # W_raman_opt is not constrained as the threshold_activation handles the output range

print("Optimization finished.")

# --- Plot Loss ---
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.grid(True)
plt.savefig(os.path.join(opt_folder, 'loss_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# --- Compare Optimized H_opt with H_true ---
plt.figure(figsize=(12, num_compounds * 2))
time_uv = np.arange(H_opt.shape[1]) # H_opt is a tensor, .shape[1] gives integer
H_opt_plot = H_opt.detach().cpu().numpy() # Convert to NumPy for plotting
H_true_plot = H_true.cpu().numpy()       # Convert H_true tensor to NumPy for plotting

for i in range(k):
    plt.subplot(k, 1, i + 1)
    plt.plot(time_uv, H_true_plot[i, :], label=f'H_true Comp {i+1}', linestyle='--')
    # Normalize H_opt_plot for comparison if scales are different
    h_opt_normalized = H_opt_plot[i, :] * (np.max(H_true_plot[i, :]) / (np.max(H_opt_plot[i, :]) + 1e-9) ) # Basic scaling
    plt.plot(time_uv, h_opt_normalized, label=f'H_opt Comp {i+1} (scaled)')
    plt.title(f'Concentration Profile for Compound {i+1}')
    plt.xlabel('Time Index')
    plt.ylabel('Concentration (scaled)')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.suptitle('Optimized H_opt vs H_true', fontsize=16, y=1.02)
plt.savefig(os.path.join(opt_folder, 'h_opt_vs_h_true.png'), dpi=300, bbox_inches='tight')
plt.close()


# --- Compare Optimized W_opt with W_true ---
plt.figure(figsize=(12, num_compounds * 3)) # Adjusted figure size
W_opt_plot = W_opt.detach().cpu().numpy() # Convert to NumPy for plotting
W_true_plot = W_true.cpu().numpy()       # Convert W_true tensor to NumPy for plotting

for i in range(k):
    plt.subplot(k, 1, i + 1) # One plot per row for each component
    plt.plot(uv_wavelengths, W_true_plot[:, i], label=f'W_true Comp {i+1}', linestyle='--')
    # Normalize W_opt_plot for comparison
    w_opt_normalized = W_opt_plot[:, i] * (np.max(W_true_plot[:, i]) / (np.max(W_opt_plot[:, i]) + 1e-9)) # Basic scaling
    plt.plot(uv_wavelengths, w_opt_normalized, label=f'W_opt Comp {i+1} (scaled)')
    plt.title(f'UV-vis Spectrum for Compound {i+1}')
    plt.xlabel('Wavelength (eV)')
    plt.ylabel('Absorbance (scaled)')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.suptitle('Optimized W_opt vs W_true', fontsize=16, y=1.02) # Adjusted y for suptitle
plt.savefig(os.path.join(opt_folder, 'w_opt_vs_w_true.png'), dpi=300, bbox_inches='tight')
plt.close()

# --- Compare Optimized threshold(W_raman_opt) with W_true_raman ---
W_raman_final_plot = threshold_activation(W_raman_opt, epsilon=epsilon, sharpness=sharpness).detach().cpu().numpy()
W_true_raman_plot = W_true_raman.cpu().numpy()

plt.figure(figsize=(12, num_compounds * 3))
for i in range(k):
    plt.subplot(k, 1, i + 1)
    plt.plot(raman_wavelengths, W_true_raman_plot[:, i], label=f'W_true_raman Comp {i+1}', linestyle='--')
    # Normalize W_raman_final_plot for comparison
    w_raman_normalized = W_raman_final_plot[:, i] * (np.max(W_true_raman_plot[:, i]) / (np.max(W_raman_final_plot[:, i]) + 1e-9))
    plt.plot(raman_wavelengths, w_raman_normalized, label=f'threshold(W_raman_opt) Comp {i+1} (scaled)')
    plt.title(f'Raman Spectrum for Compound {i+1}')
    plt.xlabel('Raman Shift (cm^-1 or similar)')
    plt.ylabel('Intensity (scaled)')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.suptitle('Optimized threshold(W_raman_opt) vs W_true_raman', fontsize=16, y=1.02)
plt.savefig(os.path.join(opt_folder, 'w_raman_opt_vs_w_true_raman.png'), dpi=300, bbox_inches='tight')
plt.close()


