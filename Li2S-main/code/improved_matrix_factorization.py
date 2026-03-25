import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from spline import UnimodalPositiveSpline

data_path = "/Users/tomer/private/NMF/Li2S-main/data/"
out2_folder = "/Users/tomer/private/NMF/Li2S-main/out2/"
os.makedirs(out2_folder, exist_ok=True)

class ImprovedMatrixFactorization:
    def __init__(self, k_components, num_spline_points=7, device='cpu'):
        """
        Initialize the improved matrix factorization model with spline parameterization.
        
        Args:
            k_components (int): Number of components to factorize into
            num_spline_points (int): Number of control points for spline parameterization
            device (str): Device to run computations on ('cpu' or 'cuda')
        """
        self.k = k_components
        self.device = device
        self.num_spline_points = num_spline_points
        self.W_opt = None
        self.H_spline_params = None  # Spline control points
        self.time_points = None
        self.splines = [UnimodalPositiveSpline(num_spline_points) for _ in range(k_components)]
        
    def soft_threshold(self, x, epsilon=0.01, beta=20.0):
        """
        Improved threshold function with sharper transition
        """
        return torch.sigmoid(beta * (x - epsilon))
    
    def compute_orthogonality_penalty(self, W):
        """
        Compute orthogonality penalty for spectral components
        """
        W_normalized = F.normalize(W, p=2, dim=0)
        gram_matrix = torch.mm(W_normalized.T, W_normalized)
        I = torch.eye(W.shape[1], device=self.device)
        return torch.norm(gram_matrix - I)

    def align_components(self, W, W_target):
        """
        Align components using Hungarian algorithm to match reference
        """
        from scipy.optimize import linear_sum_assignment
        W_norm = F.normalize(W, p=2, dim=0)
        W_target_norm = F.normalize(W_target, p=2, dim=0)
        cost_matrix = -torch.abs(torch.mm(W_norm.T, W_target_norm)).cpu().detach().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return col_ind
    
    def spline_to_H(self):
        """
        Convert spline parameters to H matrix and normalize to ensure sum to 1 across components
        at each time point
        """
        H = torch.zeros((self.k, len(self.time_points)), device=self.device)
        time_np = self.time_points.cpu().numpy()
        
        # First evaluate all splines
        for i in range(self.k):
            control_x = np.linspace(time_np.min(), time_np.max(), self.num_spline_points)
            control_y = self.H_spline_params[i].detach().cpu().numpy()
            self.splines[i].fit(control_x, control_y)
            H[i] = torch.tensor(self.splines[i](time_np), device=self.device)
        
        # Ensure non-negativity
        H = F.relu(H)
        
        # # First normalize each component to be between 0 and 1
        # H = H / (H.max(dim=1, keepdim=True)[0] + 1e-10)
        
        # Then normalize each time point to sum to 1
        H = H / (H.sum(dim=0, keepdim=True) + 1e-10)
        
        # Double check the normalization
        sums = H.sum(dim=0)
        if not torch.allclose(sums, torch.ones_like(sums), rtol=1e-3):
            print(f"Warning: H sums not close to 1. Min: {sums.min().item():.6f}, Max: {sums.max().item():.6f}")
            # Force normalization again
            H = H / (H.sum(dim=0, keepdim=True))
        
        return H

    def verify_sum_to_one(self, H):
        """
        Verify that the concentrations sum to 1 and print diagnostics if they don't
        """
        sums = H.sum(dim=0)
        max_dev = torch.max(torch.abs(sums - 1.0)).item()
        min_sum = torch.min(sums).item()
        max_sum = torch.max(sums).item()
        
        if max_dev > 1e-3:
            print(f"Sum-to-one violation detected:")
            print(f"Min sum: {min_sum:.6f}")
            print(f"Max sum: {max_sum:.6f}")
            print(f"Max deviation: {max_dev:.6f}")
            return False
        return True

    def initialize_parameters(self, An, W_true=None):
        """
        Initialize using NMF for UV-vis with spline parameterization
        """
        N_uv_wl, N_time = An.shape
        
        # Create time points array
        self.time_points = torch.arange(N_time, device=self.device, dtype=torch.float32)
        
        # Initialize UV-vis components using NMF
        W_init = torch.rand(N_uv_wl, self.k, device=self.device) * 0.1
        H_init = torch.rand(self.k, N_time, device=self.device) * 0.1
        
        # NMF iterations for UV-vis with sum-to-one constraint
        for _ in range(10):
            # Update H
            WtA = torch.mm(W_init.T, An)
            WtW = torch.mm(W_init.T, W_init)
            H_init = H_init * WtA / (torch.mm(WtW, H_init) + 1e-10)
            # Normalize H columns to sum to 1
            H_init = H_init / (H_init.sum(dim=0, keepdim=True) + 1e-10)
            
            # Update W
            AHt = torch.mm(An, H_init.T)
            HHt = torch.mm(H_init, H_init.T)
            W_init = W_init * AHt / (torch.mm(W_init, HHt) + 1e-10)
            
            An_est = W_init @ H_init
            loss_uv = F.mse_loss(An_est, An)
            print(f"NMF initialization error: {loss_uv.item():.6f}")
        
        # Initialize spline parameters from H_init
        self.H_spline_params = torch.zeros((self.k, self.num_spline_points), device=self.device, requires_grad=True)
        time_np = self.time_points.cpu().numpy()
        
        # Sample H_init at control points
        control_x = np.linspace(time_np.min(), time_np.max(), self.num_spline_points)
        for i in range(self.k):
            self.H_spline_params.data[i] = torch.tensor(
                np.interp(control_x, time_np, H_init[i].cpu().numpy()),
                device=self.device
            )
        
        # Normalize spline control points to maintain reasonable scale and sum-to-one
        norm_factors = torch.zeros(self.k, device=self.device)
        for i in range(self.k):
            control_x = np.linspace(time_np.min(), time_np.max(), self.num_spline_points)
            control_y = self.H_spline_params[i].detach().cpu().numpy()
            self.splines[i].fit(control_x, control_y)
            H = torch.tensor(self.splines[i](time_np), device=self.device)
            norm_factors[i] = H.sum()
        self.H_spline_params.data = self.H_spline_params.data / (norm_factors[:, None] + 1e-10)
        
        # Align components if W_true is provided
        if W_true is not None:
            perm = self.align_components(W_init, W_true)
            W_init = W_init[:, perm]
            self.H_spline_params.data = self.H_spline_params.data[perm]
        
        # Initialize parameters
        self.W_opt = W_init.requires_grad_(True)

    def compute_regularization(self, lambda_smooth=0.01, lambda_sparse=0.01, lambda_ortho=0.1):
        """
        Enhanced regularization with orthogonality constraint and spline smoothness
        """
        # Smoothness regularization for spectral components
        W_smooth = torch.sum(torch.square(self.W_opt[1:] - self.W_opt[:-1]))
        
        # Sparsity regularization for spline control points
        H_sparse = torch.sum(torch.abs(self.H_spline_params))
        
        # Orthogonality regularization
        W_ortho = self.compute_orthogonality_penalty(self.W_opt)
        
        # Add smoothness penalty for spline control points
        H_smooth = torch.sum(torch.square(self.H_spline_params[:, 1:] - self.H_spline_params[:, :-1]))
        
        return (lambda_smooth * (W_smooth + H_smooth) + 
                lambda_sparse * H_sparse +
                lambda_ortho * W_ortho)

    def optimize(self, An, W_true=None, max_iterations=10000, patience=200,
                lambda_smooth=0.01, lambda_ortho=0.1, verbose=True):
        """
        Improved optimization strategy with spline-based concentration profiles
        ensuring sum-to-one constraint
        """
        # Move data to device if needed
        An = An.to(self.device)
        
        # Initialize parameters
        self.initialize_parameters(An, W_true)
        
        # Optimizers with momentum
        optimizer = optim.SGD([
            {'params': self.W_opt, 'lr': 1e-3},
            {'params': self.H_spline_params, 'lr': 1e-3}
        ], weight_decay=0)
        
        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2)
        
        best_loss = float('inf')
        patience_counter = 0
        loss_history = []
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Get current H matrix from spline parameters (automatically normalized)
            H_current = self.spline_to_H()
            
            # Verify sum-to-one constraint
            if not self.verify_sum_to_one(H_current) and verbose:
                print(f"Sum-to-one violation at iteration {iteration}")
            
            # UV-vis reconstruction
            An_pred = self.W_opt @ H_current
            loss_uv = F.mse_loss(An_pred, An)
            
            # Add strong sum-to-one constraint penalty
            sum_penalty = 100.0 * torch.mean(torch.square(torch.sum(H_current, dim=0) - 1.0))
            
            # Regularization
            reg_loss = self.compute_regularization(
                lambda_smooth=lambda_smooth,
                lambda_ortho=lambda_ortho
            )
            
            # Total loss with sum-to-one penalty
            total_loss = loss_uv + reg_loss + sum_penalty
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([self.W_opt, self.H_spline_params], 1.0)
            
            optimizer.step()
            # scheduler.step()
            
            # Project to non-negative orthant and normalize
            with torch.no_grad():
                self.W_opt.data.clamp_(min=0)
                self.H_spline_params.data.clamp_(min=0)
                
                # Normalize spline parameters more aggressively
                # # First normalize to [0,1] range
                # self.H_spline_params.data = self.H_spline_params.data / (self.H_spline_params.data.max(dim=1, keepdim=True)[0] + 1e-10)
                # Then ensure sum-to-one
                self.H_spline_params.data = self.H_spline_params.data / (self.H_spline_params.data.sum(dim=0, keepdim=True) + 1e-10)
                
                # Verify the normalization worked
                H_check = self.spline_to_H()
                if not self.verify_sum_to_one(H_check) and verbose:
                    print("Warning: Normalization failed after parameter update")
            
            loss_history.append(total_loss.item())
            
            if total_loss < best_loss:
                best_loss = total_loss
                patience_counter = 0
                best_state = {
                    'W_opt': self.W_opt.data.clone(),
                    'H_spline_params': self.H_spline_params.data.clone()
                }
                
                # Verify best state maintains sum-to-one
                H_best = self.spline_to_H()
                if not self.verify_sum_to_one(H_best) and verbose:
                    print("Warning: Best state violates sum-to-one constraint")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at iteration {iteration}")
                self.W_opt.data = best_state['W_opt']
                self.H_spline_params.data = best_state['H_spline_params']
                break
            
            if verbose and (iteration + 1) % 100 == 0:
                # Get current H for validation
                H_current = self.spline_to_H()
                sum_check = torch.sum(H_current, dim=0)
                max_dev = torch.max(torch.abs(sum_check - 1.0)).item()
                min_sum = torch.min(sum_check).item()
                max_sum = torch.max(sum_check).item()
                
                print(f"Iteration {iteration+1}/{max_iterations}, "
                      f"Loss: {total_loss.item():.4f} "
                      f"(UV: {loss_uv.item():.4f}, "
                      f"Sum penalty: {sum_penalty.item():.6f}), "
                      f"reg: {reg_loss.item():.6f}, "
                      f"Sum range: [{min_sum:.6f}, {max_sum:.6f}]")
        
        # Final verification
        final_H = self.spline_to_H()
        if not self.verify_sum_to_one(final_H):
            print("Warning: Final solution violates sum-to-one constraint")
        
        return loss_history

    def get_factors(self):
        """
        Return the optimized factors
        """
        return (self.W_opt.detach(), self.spline_to_H().detach())

def plot_results(model, An, W_true, H_true, uv_wavelengths, loss_history):
    """
    Plot the results of the matrix factorization with proper scaling
    """
    W_opt, H_opt = model.get_factors()
    k = W_opt.shape[1]
    
    # Convert to numpy for plotting
    W_opt = W_opt.cpu().numpy()
    H_opt = H_opt.cpu().numpy()
    W_true = W_true.cpu().numpy()
    H_true = H_true.cpu().numpy()
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.grid(True)
    plt.savefig(os.path.join(out2_folder, 'loss_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Scale factors for each component
    W_scale_factors = np.zeros(k)
    H_scale_factors = np.zeros(k)
    
    for i in range(k):
        # Calculate scaling factors to match true components
        W_scale_factors[i] = np.sum(W_true[:, i] * W_opt[:, i]) / (np.sum(W_opt[:, i] * W_opt[:, i]) + 1e-10)
        H_scale_factors[i] = np.sum(H_true[i, :] * H_opt[i, :]) / (np.sum(H_opt[i, :] * H_opt[i, :]) + 1e-10)
    
    # Plot UV-vis components
    plt.figure(figsize=(15, 5*k))
    for i in range(k):
        plt.subplot(k, 1, i+1)
        plt.plot(uv_wavelengths, W_true[:, i], 'b-', label='True', linewidth=2)
        plt.plot(uv_wavelengths, W_opt[:, i] * W_scale_factors[i], 'r--', 
                label='Estimated (scaled)', linewidth=2)
        plt.title(f'UV-vis Component {i+1}')
        plt.xlabel('Wavelength (eV)')
        plt.ylabel('Absorbance')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out2_folder, 'uv_vis_components.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot concentration profiles
    plt.figure(figsize=(15, 5*k))
    for i in range(k):
        plt.subplot(k, 1, i+1)
        plt.plot(H_true[i], 'b-', label='True', linewidth=2)
        plt.plot(H_opt[i] * H_scale_factors[i], 'r--', 
                label='Estimated (scaled)', linewidth=2)
        plt.title(f'Concentration Profile {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out2_folder, 'concentration_profiles.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print scaling information
    print("\nScaling factors used for visualization:")
    for i in range(k):
        print(f"\nComponent {i+1}:")
        print(f"  UV-vis scale: {W_scale_factors[i]:.3f}")
        print(f"  Concentration scale: {H_scale_factors[i]:.3f}")
        
    # Calculate and print reconstruction errors
    An_recon = W_opt @ np.diag(H_scale_factors) @ H_opt
    uv_error = np.mean((An.cpu().numpy() - An_recon)**2)
    
    print("\nReconstruction Errors:")
    print(f"UV-vis MSE: {uv_error:.6f}")

def main():
    
    # Load UV-vis data
    df_An = pd.read_csv(data_path + "An.csv", header=None)
    uv_wavelengths = df_An.iloc[1:, 0].values.astype(float)
    An = torch.from_numpy(df_An.iloc[1:, 1:].values.astype(float)).to(torch.float32)
    
    df_W_true = pd.read_csv(data_path + "W_true.csv", header=None)
    W_true = torch.from_numpy(df_W_true.iloc[1:, 1:].values.astype(float)).to(torch.float32)
    W_true /= 100  # Scale down W_true to match the scale of An and improve optimization stability
    
    df_H_true = pd.read_csv(data_path + "H_true.csv", header=None)
    H_true = torch.from_numpy(df_H_true.iloc[1:, 1:].values.astype(float)).to(torch.float32)
    
    An_est = W_true @ H_true
    loss_uv = F.mse_loss(An_est, An)
    print(f"Initial UV-vis reconstruction MSE with true factors: {loss_uv.item():.6f}")
    # Print data statistics
    print("\nData Statistics:")
    print(f"UV-vis range: [{torch.min(An):.3f}, {torch.max(An):.3f}]")
    print(f"UV-vis shape: {An.shape}")
    
    # Initialize and run the improved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    k_components = W_true.shape[1]
    print(f"Number of components: {k_components}")
    
    # Move data to device
    An = An.to(device)
    W_true = W_true.to(device)
    H_true = H_true.to(device)
    
    # Create and run model with improved parameters
    model = ImprovedMatrixFactorization(k_components, device=device)
    loss_history = model.optimize(
        An,
        W_true=W_true,  # Use reference for alignment
        max_iterations=20000,  # Increased iterations
        patience=100,  # Increased patience
        lambda_smooth=0.01,
        lambda_ortho=0.,  # Added orthogonality
        verbose=True
    )
    
    # Plot results with improved scaling
    plot_results(model, An, W_true, H_true, uv_wavelengths, loss_history)

if __name__ == "__main__":
    main() 