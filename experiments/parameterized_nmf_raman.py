import os
import sys  
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nmf.parameterized_nmf import *

data_path = "/Users/tomer/private/NMF/Li2S-main/data/"
out_folder = "/Users/tomer/private/NMF/out_parameterized_nmf_raman"
os.makedirs(out_folder, exist_ok=True)

# ========================================
# Abstract Parameterization Classes
# ========================================

class GaussianThresholdParameterization(GaussianParameterization):
    """
    Gaussian parameterization with STE (Straight-Through Estimator) thresholding.
    After computing the Gaussian, values > 1e-2 are set to 1.
    Uses STE to allow gradients to flow through the thresholding operation.
    
    Args:
        threshold: float - threshold value (default: 1e-2)
    """
    def __init__(self, shape, axis=1, mean_bounds=None, std_bounds=None, scale_bounds=None, threshold=1e-2):
        """
        Initialize Gaussian parameterization with STE thresholding.
        
        Args:
            shape: tuple (n_rows, n_cols) - shape of the matrix to generate
            axis: int - 0 for row-wise, 1 for column-wise (default: 1)
            mean_bounds: tuple (min, max) - bounds for mean parameters
            std_bounds: tuple (min, max) - bounds for std parameters
            scale_bounds: tuple (min, max) - bounds for scale parameters
            threshold: float - threshold value for STE (default: 1e-2)
        """
        self.threshold = threshold
        super(GaussianThresholdParameterization, self).__init__(
            shape, axis, mean_bounds, std_bounds, scale_bounds
        )
    
    def forward(self):
        """
        Generate matrix where each column/row is a Gaussian, with STE thresholding.
        
        Returns:
            torch.Tensor: Matrix of shape (n_rows, n_cols) with thresholding applied
        """
        # Get Gaussian matrix from parent class
        gaussian = super().forward()
        
        # Apply STE thresholding
        # hard threshold
        hard = torch.where(
            gaussian > self.threshold,
            torch.ones_like(gaussian),
            gaussian
        )

        # soft approximation
        k = 50
        gate = torch.sigmoid(k * (gaussian - self.threshold))
        soft = gaussian * (1 - gate) + gate

        # gradient-replaced output
        output = hard + soft - soft.detach()
        return output
    
    def __repr__(self):
        """String representation showing constrained parameter values and threshold."""
        base_repr = super().__repr__()
        return base_repr.replace(
            f"{self.__class__.__base__.__name__}",
            f"{self.__class__.__name__}"
        ) + f"\n  threshold: {self.threshold}"

class ParameterizedRamanNMFSolver(ParameterizedNMFSolver):

    def initialize_with_nmf(self, n_nmf_iterations=50):
        """
        Initialize parameters using standard NMF.
        
        Args:
            n_nmf_iterations: int - number of NMF iterations
        """
        print("\nInitializing with NMF...")
        N_rows, N_cols = self.A_observed.shape
        k = self.W_param.n_cols if hasattr(self.W_param, 'n_cols') else self.H_param.n_rows
        
        # Initialize with random values
        W_init = torch.rand(N_rows, k, device=self.device)
        H_init = torch.rand(k, N_cols, device=self.device)
        
        # NMF iterations with binary constraint on H (project H to binary after each update)
        for iteration in range(n_nmf_iterations):
            # Update H
            WtA = torch.mm(W_init.T, self.A_observed)
            WtW = torch.mm(W_init.T, W_init)
            H_init = H_init * WtA / (torch.mm(WtW, H_init) + 1e-10)
            # Project H to binary
            H_init = (H_init > 0.01).float()
            
            # Update W
            AHt = torch.mm(self.A_observed, H_init.T)
            HHt = torch.mm(H_init, H_init.T)
            W_init = W_init * AHt / (torch.mm(W_init, HHt) + 1e-10)
            
            if (iteration + 1) % 10 == 0:
                An_est = W_init @ H_init
                loss_nmf = F.mse_loss(An_est, self.A_observed)
                print(f"  NMF iteration {iteration+1}/{n_nmf_iterations}, Loss: {loss_nmf.item():.6f}")
        
        # Final loss
        An_est = W_init @ H_init
        loss_nmf = F.mse_loss(An_est, self.A_observed)
        print(f"NMF initialization complete. Final loss: {loss_nmf.item():.6f}")
        
        # Initialize parameterizations from NMF result
        print("Fitting Gaussian parameters to NMF solution...")
        self.W_param.initialize_from_matrix(W_init)
        self.H_param.initialize_from_matrix(H_init)
        
        # Check how well the parameterization fits
        W_param_matrix = self.W_param.matrix
        H_param_matrix = self.H_param.matrix
        An_param = W_param_matrix @ H_param_matrix
        loss_param = F.mse_loss(An_param, self.A_observed)
        print(f"Parameterized approximation loss: {loss_param.item():.6f}\n")
    
def main():
    # Load UV-vis data
    df_An = pd.read_csv(data_path + "An_raman.csv", header=None)
    uv_wavelengths = df_An.iloc[1:, 0].values.astype(float)
    An = torch.from_numpy(df_An.iloc[1:, 1:].values.astype(float)).to(torch.float32)
    # An += 1e-2 * torch.randn_like(An)  # Add small noise  - robustness test
    
    df_W_true = pd.read_csv(data_path + "W_true_raman.csv", header=None)
    W_true = torch.from_numpy(df_W_true.iloc[1:, 1:].values.astype(float)).to(torch.float32)
    
    df_H_true = pd.read_csv(data_path + "H_true_raman.csv", header=None)
    H_true = torch.from_numpy(df_H_true.iloc[1:, 1:].values.astype(float)).to(torch.float32)

    
    df_H_true_vis = pd.read_csv(data_path + "H_true.csv", header=None)
    H_true_vis = torch.from_numpy(df_H_true_vis.iloc[1:, 1:].values.astype(float)).to(torch.float32)


    An_est = W_true @ H_true
    loss_uv = F.mse_loss(An_est, An)
    print(f"Initial Raman reconstruction MSE with true factors: {loss_uv.item():.6f}")
    # Print data statistics
    print("\nData Statistics:")
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
    print("Running with Gaussian Parameterization")
    print("="*70)
    
    W_param = MixtureOfGeneralizedGaussiansParameterization(
        shape=(n_wavelengths, k_components), 
        n_gaussians=3,
        axis=1,  # Column-wise for W matrix
        mean_bounds=(0, n_wavelengths),
        std_bounds=(0.1, n_wavelengths / 5),
        scale_bounds=(0, 15.0),
        beta_bounds=(1.0, 4.0)
    )
    H_param = GaussianThresholdParameterization(
        shape=(k_components, n_timepoints),
        axis=0,  # Row-wise for H matrix
        mean_bounds=(0, n_timepoints),
        std_bounds=(0.1, n_timepoints),
        scale_bounds=(0, 1.0)
    )
    H_param.initialize_from_matrix(H_true) # optimize only W after known H from vis data
    
    model = ParameterizedNMFSolver(
        W_param=W_param,
        H_param=H_param,
        A_observed=An,
        lambda_penalty=0, # no sum-to-one penalty for Raman data
        device=device
    )
    
    W, H, loss_history = model.solve(
        n_iterations=10000,
        lr=0.1,
        print_every=1000,
        nmf_init=False, 
    )
    W_param.set_params(**model.best_params['W_params'])
    H_param.set_params(**model.best_params['H_params'])
    
    
    plot_results(W_param, H_param, An, W_true, H_true, uv_wavelengths, loss_history, prefix='raman')
    print("\nFinal Parameter Values:")
    print("W parameters:")
    print(W_param)
    print("\nH parameters:")
    print(H_param)

if __name__ == "__main__":
    main() 