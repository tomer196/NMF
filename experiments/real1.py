import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nmf.parameterized_nmf_v2 import *

data_path = '/Users/tomer/private/NMF/data/real1/'
out_folder = "/Users/tomer/private/NMF/out_parameterized_nmf/real1"
os.makedirs(out_folder, exist_ok=True)


def plot_wh(w, h, prefix=""):
    uv_wavelengths = np.arange(w.shape[0])
    k = w.shape[1]
    # Plot UV-vis components
    plt.figure(figsize=(15, 5*k))
    for i in range(k):
        plt.subplot(k, 1, i+1)
        plt.plot(uv_wavelengths, w[:, i], 'r--', 
                label='Estimated', linewidth=2)
        plt.title(f'UV-vis Component {i+1}')
        plt.xlabel('Wavelength (eV)')
        plt.ylabel('Absorbance')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f'{prefix}_uv_vis_components.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot concentration profiles
    plt.figure(figsize=(15, 5*k))
    for i in range(k):
        plt.subplot(k, 1, i+1)
        plt.plot(h[i], 'r--', 
                label='Estimated', linewidth=2)
        plt.title(f'Concentration Profile {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.ylim([0, 1])
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f'{prefix}_concentration_profiles.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_results(Areal, Aapprox, prefix=""):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(Areal.cpu().numpy(), aspect='auto', cmap='viridis')
    plt.title('Original Data (A)')
    plt.xlabel('Time')
    plt.ylabel('Wavelength')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(Aapprox.cpu().numpy(), aspect='auto', cmap='viridis')
    plt.title('NMF Approximation (W @ H)')
    plt.xlabel('Time')
    plt.ylabel('Wavelength')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    err = torch.abs(Areal - Aapprox)
    plt.imshow(err.cpu().numpy(), aspect='auto', cmap='inferno')
    plt.title('Absolute Error')
    plt.xlabel('Time')
    plt.ylabel('Wavelength')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f'{prefix}_data_approximation.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    # Load UV-vis data
    df_An = pd.read_csv(data_path + "An_UV_real.csv", header=None)
    uv_wavelengths = df_An.iloc[1:, 0].values.astype(float)
    An = torch.from_numpy(df_An.iloc[:, 2:].values.astype(float)).to(torch.float32)
    An = An.to(device)
    # An += 1e-2 * torch.randn_like(An)  # Add small noise  - robustness test
    # An /= 100  # Scale down An to improve optimization stability
    
    # Print data statistics
    print("\nData Statistics:")
    print(f"UV-vis range: [{torch.min(An):.3f}, {torch.max(An):.3f}]")
    print(f"UV-vis shape: {An.shape}")
    
    # Create parameterizations
    k_components = 3  # Set the number of components manually
    n_wavelengths, n_timepoints = An.shape
    print(f"Number of components: {k_components}")
    
    W_param = MixtureOfGeneralizedGaussiansParameterization(
        shape=(n_wavelengths, k_components), 
        n_gaussians=2,
        axis=1,  # Column-wise for W matrix
        mean_bounds=(0, n_wavelengths),
        std_bounds=(0.1, n_wavelengths/10),
        scale_bounds=(0, 1.0),
        beta_bounds=(1.5, 2.5)
    )
    W_init = torch.zeros((n_wavelengths, k_components), device=device)
    W_init[:, 0] = An[:, 0]  # Initialize first component with first column of An
    W_param = PartiallyFixedMixtureOfGeneralizedGaussiansParameterization(
        shape=(n_wavelengths, k_components), 
        n_gaussians=2,
        mean_bounds=(0, n_wavelengths),
        std_bounds=(0.1, n_wavelengths/10),
        scale_bounds=(0, 1.0),
        beta_bounds=(1.5, 2.5),
        fix_components=[0],
        reference_matrix=W_init
    )
    # W_param = SplineParameterization(
    #     shape=(n_wavelengths, k_components), 
    #     n_control_points=30,
    #     value_bounds=(0, 5.0)
    # )
    
    # H_param = GaussianParameterization(
    #     shape=(k_components, n_timepoints),
    #     axis=0,  # Row-wise for H matrix
    #     mean_bounds=(0, n_timepoints),
    #     std_bounds=(0.1, n_timepoints / 8),
    #     scale_bounds=(0, 1.0)
    # )
    # H_param = GeneralizedGaussianParameterization(
    #     shape=(k_components, n_timepoints),
    #     axis=0,
    #     mean_bounds=(0, n_timepoints),
    #     std_bounds=(1, 15.0),
    #     beta_bounds=(1.3, 4.0),
    #     scale_bounds=(0, 1.0)
    # )

    H_param = SkewNormalParameterization(
        shape=(k_components, n_timepoints),
        axis=0,  # Row-wise for H matrix
        mean_bounds=(-0.5*n_timepoints, 1.5*n_timepoints),
        std_bounds=(0.1, n_timepoints),
        skewness_bounds=(-10.0, 10.0),
        scale_bounds=(0, 50.0)
    )
    # H_param = SkewTParameterization(
    #     shape=(k_components, n_timepoints),
    #     axis=0,  # Row-wise for H matrix
    #     mean_bounds=(-0.5*n_timepoints, 1.5*n_timepoints),
    #     std_bounds=(0.1, n_timepoints),
    #     skewness_bounds=(-10.0, 10.0),
    #     df_bounds=(1.0, 90.0),
    #     scale_bounds=(0, 50.0)
    # )


    # Create composite loss function
    loss_function = CompositeLoss([
        # (FittingLoss(), 1.0, 'fitting_loss'),
        (RobustFittingLoss(sigma=0.7*An.max().item(), beta=20), 1.0, 'fitting_loss'),
        (SumPenaltyLoss(), 0.01, 'penalty_loss', True),
        (HFirstLoss(), 0.01, 'H_first_loss', True)
    ])


    model = ParameterizedNMFSolver(
        W_param=W_param,
        H_param=H_param,
        A_observed=An,
        loss_function=loss_function,
        device=device
    )
    
    W, H, loss_history, _ = model.solve(
        n_iterations=10000,
        lr=0.01,
        print_every=1000,
        nmf_init=True,
        n_nmf_iterations=10000,
        n_runs=5
    )
    W_param.set_params(**model.best_params['W_params'])
    H_param.set_params(**model.best_params['H_params'])
    
    
    plot_wh(W, H, prefix="final")
    plot_results(An, W @ H, prefix="final")

    error = F.mse_loss(W @ H, An).item()
    print(f"\nFinal MSE Loss: {error:.6f}")
    print(f"Max deviation: {np.max(np.abs(np.sum(H.detach().cpu().numpy(), axis=0) - 1.0)):.6f}")
    print(f"First time point in H: {H[:, 0].detach().cpu().numpy()}")


    # print("\nFinal Parameter Values:")
    # print("W parameters:")
    # print(W_param)
    # print("\nH parameters:")
    # print(H_param)

if __name__ == "__main__":
    main() 