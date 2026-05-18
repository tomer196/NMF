import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from scipy.signal import find_peaks

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nmf.parameterized_nmf_v2 import *

prefix = "synth_v1"
data_path = f'/Users/tomer/private/NMF/data/{prefix}/'
out_folder = f"/Users/tomer/private/NMF/out_parameterized_nmf/{prefix}"
os.makedirs(out_folder, exist_ok=True)


def main():
    # Load UV-vis data
    df_An = pd.read_csv(data_path + "An.csv", header=None)
    uv_wavelengths = df_An.iloc[1:, 0].values.astype(float)
    An = torch.from_numpy(df_An.iloc[1:, 1:].values.astype(float)).to(torch.float32)
    # An += 1e-2 * torch.randn_like(An)  # Add small noise  - robustness test
    An /= 100  # Scale down An to improve optimization stability
    
    df_W_true = pd.read_csv(data_path + "W_true.csv", header=None)
    W_true = torch.from_numpy(df_W_true.iloc[1:, 1:].values.astype(float)).to(torch.float32)
    W_true /= 100  # Scale down W_true to match the scale of An and improve optimization stability
    
    df_H_true = pd.read_csv(data_path + "H_true.csv", header=None)
    try:
        H_true = torch.from_numpy(df_H_true.iloc[1:, 1:].values.astype(float)).to(torch.float32)
    except:
        H_true = torch.from_numpy(df_H_true.iloc[1:, 2:].values.astype(float)).to(torch.float32)
    
    An_est = W_true @ H_true
    loss_uv = F.mse_loss(An_est, An)
    print(f"Initial UV-vis reconstruction MSE with true factors: {loss_uv.item():.6f}")
    # Print data statistics
    print("\nData Statistics:")
    print(f"UV-vis range: [{torch.min(An):.3f}, {torch.max(An):.3f}]")
    print(f"UV-vis shape: {An.shape}")
    print(f"W_true shape: {W_true.shape}, range: [{torch.min(W_true):.3f}, {torch.max(W_true):.3f}]")
    print(f"H_true shape: {H_true.shape}, range: [{torch.min(H_true):.3f}, {torch.max(H_true):.3f}]")    
    # Initialize and run the improved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    k_components = W_true.shape[1]
    print(f"Number of components: {k_components}")
    
    # Move data to device
    An = An.to(device)
    W_true = W_true.to(device)
    H_true = H_true.to(device)
    
    # Create parameterizations
    n_wavelengths, n_timepoints = An.shape
    
    print("\n" + "="*70)
    # W_param = MixtureOfGeneralizedGaussiansParameterization(
    #     shape=(n_wavelengths, k_components), 
    #     n_gaussians=2,
    #     axis=1,  # Column-wise for W matrix
    #     mean_bounds=(0, n_wavelengths),
    #     std_bounds=(0.1, n_wavelengths/10),
    #     scale_bounds=(0, 10.0),
    #     beta_bounds=(1.5, 2.5)
    # )
    W_init = torch.zeros((n_wavelengths, k_components), device=device)
    fix_components = [0]
    W_init[:, fix_components] = W_true[:, fix_components]  # Initialize known components
    # W_init[:, fix_components] = An[:, fix_components]  # Initialize known components
    
    W_param = PartiallyFixedMixtureOfGeneralizedGaussiansParameterization(
        shape=(n_wavelengths, k_components), 
        n_gaussians=2,
        mean_bounds=(0, n_wavelengths),
        std_bounds=(0.1, n_wavelengths/10),
        scale_bounds=(0, 10.0),
        beta_bounds=(1.5, 2.5),
        fix_components=fix_components,
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

    h_mean_bounds = [
        (-0.5 * n_timepoints, 0.5 * n_timepoints),
        (0.3 * n_timepoints, 0.7 * n_timepoints),
        (0.5 * n_timepoints, 1 * n_timepoints),
        (0.5 * n_timepoints, 1 * n_timepoints),
        (0.75 * n_timepoints, 1.5 * n_timepoints),
    ]

    H_param = SkewNormalParameterization(
        shape=(k_components, n_timepoints),
        axis=0,  # Row-wise for H matrix
        # mean_bounds=(-0.5*n_timepoints, 1.5*n_timepoints),
        mean_bounds=h_mean_bounds,
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
        (FittingLoss(), 1.0, 'fitting_loss'),
        (SumPenaltyLoss(), 0.01, 'penalty_loss', True),
        (HFirstLoss(), 0.01, 'H_first_loss', True),
        # (HComponentLimitBeforeTimeLoss(max_k=2, t=int(0.15 * n_timepoints)), 0.01, 'H_component_limit_loss', True)
    ],
    )

    model = ParameterizedNMFSolver(
        W_param=W_param,
        H_param=H_param,
        A_observed=An,
        loss_function=loss_function,
        device=device,
    )
    
    W, H, loss_history, _ = model.solve(
        n_iterations=10000,
        lr=0.1,
        print_every=1000,
        nmf_init=True,
        n_nmf_iterations=10000,
        n_runs=5
    )
    W_param.set_params(**model.best_params['W_params'])
    H_param.set_params(**model.best_params['H_params'])
    
    
    plot_results(W_param, H_param, An, W_true, H_true, uv_wavelengths, loss_history, prefix, out_folder=out_folder)
    # print("\nFinal Parameter Values:")
    # print("W parameters:")
    # print(W_param)
    # print("\nH parameters:")
    # print(H_param)
    # H_param.matrix()

if __name__ == "__main__":
    main() 