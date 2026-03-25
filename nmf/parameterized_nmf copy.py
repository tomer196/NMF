import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

data_path = "/Users/tomer/private/NMF/data/synth0/"
out_folder = "/Users/tomer/private/NMF/out_parameterized_nmf/synth0"
os.makedirs(out_folder, exist_ok=True)


# ========================================
# Abstract Parameterization Classes
# ========================================

class MatrixParameterization(ABC, nn.Module):
    """
    Abstract class for matrix parameterization.
    Subclasses should implement how to generate a matrix from parameters.
    """
    def __init__(self, shape, mean_bounds=None, std_bounds=None, scale_bounds=None):
        """
        Initialize parameterization with matrix shape.
        
        Args:
            shape: tuple (n_rows, n_cols) - shape of the matrix to generate
            mean_bounds: tuple (min, max) - bounds for mean parameters
            std_bounds: tuple (min, max) - bounds for std parameters
            scale_bounds: tuple (min, max) - bounds for scale parameters
        """
        super(MatrixParameterization, self).__init__()
        self.shape = shape
        self.n_rows, self.n_cols = shape
        self.mean_bounds = mean_bounds
        self.std_bounds = std_bounds
        self.scale_bounds = scale_bounds
        self._init_params()
    
    @abstractmethod
    def _init_params(self):
        """Initialize parameters randomly. Should be implemented by subclasses."""
        pass
    
    @abstractmethod
    def forward(self):
        """
        Generate the full matrix from parameters.
        Should be differentiable.
        
        Returns:
            torch.Tensor: Matrix of shape self.shape
        """
        pass
    
    @property
    def matrix(self):
        """Property to get the current matrix."""
        return self.forward()

    def initialize_from_matrix(self, target_matrix, n_iterations=500, lr=0.01, patience=50):
        """
        Initialize parameters by fitting to a target matrix using optimization.
        
        Args:
            target_matrix: torch.Tensor - matrix to fit parameters to
            n_iterations: int - number of optimization iterations
            lr: float - learning rate for optimization
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience//2)
        target_matrix = target_matrix.to(next(self.parameters()).device)
        
        best_loss = float('inf')
        best_iteration = 0
        best_params = {name: param.clone().detach() for name, param in self.params.items()}
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            generated_matrix = self.forward()
            loss = F.mse_loss(generated_matrix, target_matrix)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_iteration = iteration
                best_params = {name: param.clone().detach() for name, param in self.params.items()}
            elif iteration - best_iteration > patience:
                print(f"Early stopping at iteration {iteration} with best loss {best_loss:.6f}")
                break 
        print(f"Finish matrix fitting, Loss: {best_loss:.6f}")
        # Load best parameters
        for name, param in self.params.items():
            param.data = best_params[name].data

    @abstractmethod
    def set_params(self, **kwargs):
        """Set parameters directly (for testing/initialization)."""
        pass

class GaussianParameterization(MatrixParameterization):
    """
    Each column or row of the matrix is a Gaussian function.
    Parameters: mean, std, scale for each column/row.
    Uses sigmoid transformation to enforce bounds.
    
    Args:
        axis: int - 0 for row-wise Gaussians, 1 for column-wise Gaussians (default: 1)
    """
    def __init__(self, shape, axis=1, mean_bounds=None, std_bounds=None, scale_bounds=None):
        """
        Initialize Gaussian parameterization.
        
        Args:
            shape: tuple (n_rows, n_cols) - shape of the matrix to generate
            axis: int - 0 for row-wise, 1 for column-wise (default: 1)
            mean_bounds: tuple (min, max) - bounds for mean parameters
            std_bounds: tuple (min, max) - bounds for std parameters
            scale_bounds: tuple (min, max) - bounds for scale parameters
        """
        self.axis = axis
        super(GaussianParameterization, self).__init__(shape, mean_bounds, std_bounds, scale_bounds)
    
    def _init_params(self):
        """Initialize unconstrained Gaussian parameters."""
        # Set default bounds based on axis
        if self.axis == 1:  # Column-wise (W matrix)
            n_gaussians = self.n_cols
            axis_size = self.n_rows
            default_scale_max = 10.0
        else:  # axis == 0, Row-wise (H matrix)
            n_gaussians = self.n_rows
            axis_size = self.n_cols
            default_scale_max = 1.0
        
        # Set default bounds if not provided
        if self.mean_bounds is None:
            self.mean_bounds = (0, axis_size)
        if self.std_bounds is None:
            self.std_bounds = (0.5, axis_size / 3)
        if self.scale_bounds is None:
            self.scale_bounds = (0, default_scale_max)
        
        # Store unconstrained parameters (will be transformed via sigmoid)
        # Initialize near 0 for sigmoid (maps to middle of range)
        self.means_unconstrained = nn.Parameter(2 * (torch.rand(n_gaussians) - 0.5))
        self.stds_unconstrained = nn.Parameter(2 * (torch.rand(n_gaussians) - 0.5))
        self.scales_unconstrained = nn.Parameter(2 * (torch.rand(n_gaussians) - 0.5))
    
    @property
    def params(self):
        """Expose unconstrained parameters."""
        return {
            'means_unconstrained': self.means_unconstrained,
            'stds_unconstrained': self.stds_unconstrained,
            'scales_unconstrained': self.scales_unconstrained
        }
    
    def set_params(self, means_unconstrained=None, stds_unconstrained=None, scales_unconstrained=None):
        """Set parameters directly (for testing/initialization)."""
        if means_unconstrained is not None:
            self.means_unconstrained.data = means_unconstrained
        if stds_unconstrained is not None:
            self.stds_unconstrained.data = stds_unconstrained
        if scales_unconstrained is not None:
            self.scales_unconstrained.data = scales_unconstrained
    
    def forward(self):
        """
        Generate matrix where each column/row is a Gaussian.
        
        Returns:
            torch.Tensor: Matrix of shape (n_rows, n_cols)
        """
        # Transform unconstrained parameters to bounded values using sigmoid
        mean_min, mean_max = self.mean_bounds
        means = mean_min + (mean_max - mean_min) * torch.sigmoid(self.means_unconstrained)
        
        std_min, std_max = self.std_bounds
        stds = std_min + (std_max - std_min) * torch.sigmoid(self.stds_unconstrained) + 1e-6
        
        scale_min, scale_max = self.scale_bounds
        scales = scale_min + (scale_max - scale_min) * torch.sigmoid(self.scales_unconstrained)
        
        if self.axis == 1:  # Column-wise
            # Create row indices
            x = torch.arange(self.n_rows, dtype=torch.float32, device=self.means_unconstrained.device)
            x = x.unsqueeze(1)  # Shape: (n_rows, 1)
            means = means.unsqueeze(0)  # Shape: (1, n_cols)
            stds = stds.unsqueeze(0)  # Shape: (1, n_cols)
            scales = scales.unsqueeze(0)  # Shape: (1, n_cols)
        else:  # axis == 0, Row-wise
            # Create column indices
            x = torch.arange(self.n_cols, dtype=torch.float32, device=self.means_unconstrained.device)
            x = x.unsqueeze(0)  # Shape: (1, n_cols)
            means = means.unsqueeze(1)  # Shape: (n_rows, 1)
            stds = stds.unsqueeze(1)  # Shape: (n_rows, 1)
            scales = scales.unsqueeze(1)  # Shape: (n_rows, 1)
        
        # Compute Gaussian
        # gaussian = scale * exp(-0.5 * ((x - mean) / std)^2)
        matrix = scales * torch.exp(-0.5 * ((x - means) / stds) ** 2)
        
        return matrix  # Shape: (n_rows, n_cols)
    
    def __repr__(self):
        """String representation showing constrained parameter values."""
        mean_min, mean_max = self.mean_bounds
        means = mean_min + (mean_max - mean_min) * torch.sigmoid(self.means_unconstrained)
        std_min, std_max = self.std_bounds
        stds = std_min + (std_max - std_min) * torch.sigmoid(self.stds_unconstrained)
        scale_min, scale_max = self.scale_bounds
        scales = scale_min + (scale_max - scale_min) * torch.sigmoid(self.scales_unconstrained)
        
        return (f"{self.__class__.__name__}(shape={self.shape}, axis={self.axis})\n"
                f"  means: {means.detach().cpu().numpy()}\n"
                f"  stds: {stds.detach().cpu().numpy()}\n"
                f"  scales: {scales.detach().cpu().numpy()}")
    
    def initialize_from_matrix(self, target_matrix):
        """Initialize parameters by finding Gaussian characteristics in each column/row."""
        target_matrix_np = target_matrix.detach().cpu().numpy()
        
        if self.axis == 1:  # Column-wise
            n_iters = self.n_cols
            axis_size = self.n_rows
        else:  # Row-wise
            n_iters = self.n_rows
            axis_size = self.n_cols
        
        for idx in range(n_iters):
            # Extract data
            if self.axis == 1:
                data = target_matrix_np[:, idx]
            else:
                data = target_matrix_np[idx, :]
            
            # Find mean as position of maximum value
            mean_init = float(np.argmax(data))
            
            # Scale is the maximum value
            scale_init = float(np.max(data))
            
            # Estimate std from weighted variance
            x = np.arange(axis_size)
            # Use data as weights to compute weighted variance
            weights = np.maximum(data, 0)  # Ensure non-negative
            if weights.sum() > 0:
                weights = weights / weights.sum()
                weighted_mean = np.sum(x * weights)
                weighted_var = np.sum(weights * (x - weighted_mean) ** 2)
                std_init = float(np.sqrt(weighted_var) + 1e-3)  # Add small value to avoid zero
            else:
                std_init = axis_size / 10.0
            
            # Apply inverse sigmoid (logit) transformation
            mean_ratio = np.clip((mean_init - self.mean_bounds[0]) / (self.mean_bounds[1] - self.mean_bounds[0]), 1e-7, 1 - 1e-7)
            self.means_unconstrained.data[idx] = torch.tensor(np.log(mean_ratio / (1 - mean_ratio)), dtype=torch.float32)
            
            std_ratio = np.clip((std_init - self.std_bounds[0]) / (self.std_bounds[1] - self.std_bounds[0]), 1e-7, 1 - 1e-7)
            self.stds_unconstrained.data[idx] = torch.tensor(np.log(std_ratio / (1 - std_ratio)), dtype=torch.float32)
            
            scale_ratio = np.clip((scale_init - self.scale_bounds[0]) / (self.scale_bounds[1] - self.scale_bounds[0]), 1e-7, 1 - 1e-7)
            self.scales_unconstrained.data[idx] = torch.tensor(np.log(scale_ratio / (1 - scale_ratio)), dtype=torch.float32)
        
        super().initialize_from_matrix(target_matrix)  # Call the base class method to run optimization after initialization

class MixtureOfGaussiansParameterization(MatrixParameterization):
    """
    Each column or row of the matrix is a mixture of Gaussians.
    Parameters: means, stds, scales for each Gaussian in each column/row.
    Uses sigmoid transformation to enforce bounds.
    
    Args:
        axis: int - 0 for row-wise Gaussians, 1 for column-wise Gaussians (default: 1)
    """
    def __init__(self, shape, n_gaussians=2, axis=1, mean_bounds=None, std_bounds=None, scale_bounds=None):
        """
        Initialize mixture of Gaussians parameterization.
        
        Args:
            shape: tuple (n_rows, n_cols) - shape of the matrix to generate
            n_gaussians: int - number of Gaussians in the mixture for each column/row
            axis: int - 0 for row-wise, 1 for column-wise (default: 1)
            mean_bounds: tuple (min, max) - bounds for mean parameters
            std_bounds: tuple (min, max) - bounds for std parameters
            scale_bounds: tuple (min, max) - bounds for scale parameters
        """
        self.n_gaussians = n_gaussians
        self.axis = axis
        super(MixtureOfGaussiansParameterization, self).__init__(shape, mean_bounds, std_bounds, scale_bounds)
    
    def _init_params(self):
        """Initialize unconstrained mixture of Gaussians parameters."""
        #Set default bounds based on axis
        if self.axis == 1:  # Column-wise (W matrix)
            n_components = self.n_cols
            axis_size = self.n_rows
            default_scale_max = 10.0
        else:  # axis == 0, Row-wise (H matrix)
            n_components = self.n_rows
            axis_size = self.n_cols
            default_scale_max = 1.0
        
        # Set default bounds if not provided
        if self.mean_bounds is None:
            self.mean_bounds = (0, axis_size)
        if self.std_bounds is None:
            self.std_bounds = (0.5, axis_size / 3)
        if self.scale_bounds is None:
            self.scale_bounds = (0, default_scale_max)
        
        # Store unconstrained parameters
        self.means_unconstrained = nn.Parameter(2 * (torch.rand(n_components, self.n_gaussians) - 0.5))
        self.stds_unconstrained = nn.Parameter(2 * (torch.rand(n_components, self.n_gaussians) - 0.5))
        self.scales_unconstrained = nn.Parameter(2 * (torch.rand(n_components, self.n_gaussians) - 0.5))
    
    @property
    def params(self):
        """Expose unconstrained parameters."""
        return {
            'means_unconstrained': self.means_unconstrained,
            'stds_unconstrained': self.stds_unconstrained,
            'scales_unconstrained': self.scales_unconstrained
        }
    
    def set_params(self, means_unconstrained=None, stds_unconstrained=None, scales_unconstrained=None):
        """Set parameters directly (for testing/initialization)."""
        if means_unconstrained is not None:
            self.means_unconstrained.data = means_unconstrained
        if stds_unconstrained is not None:
            self.stds_unconstrained.data = stds_unconstrained
        if scales_unconstrained is not None:
            self.scales_unconstrained.data = scales_unconstrained
    
    def forward(self):
        """
        Generate matrix where each column/row is a mixture of Gaussians.
        
        Returns:
            torch.Tensor: Matrix of shape (n_rows, n_cols)
        """
        # Transform unconstrained parameters to bounded values using sigmoid
        mean_min, mean_max = self.mean_bounds
        means = mean_min + (mean_max - mean_min) * torch.sigmoid(self.means_unconstrained)
        
        std_min, std_max = self.std_bounds
        stds = std_min + (std_max - std_min) * torch.sigmoid(self.stds_unconstrained) + 1e-6
        
        scale_min, scale_max = self.scale_bounds
        scales = scale_min + (scale_max - scale_min) * torch.sigmoid(self.scales_unconstrained)
        
        if self.axis == 1:  # Column-wise
            # Create row indices
            x = torch.arange(self.n_rows, dtype=torch.float32, device=self.means_unconstrained.device)
            x = x.unsqueeze(1).unsqueeze(1)  # Shape: (n_rows, 1, 1)
            means = means.unsqueeze(0)  # Shape: (1, n_cols, n_gaussians)
            stds = stds.unsqueeze(0)  # Shape: (1, n_cols, n_gaussians)
            scales = scales.unsqueeze(0)  # Shape: (1, n_cols, n_gaussians)
            
            # Compute Gaussian for each component
            gaussians = scales * torch.exp(-0.5 * ((x - means) / stds) ** 2)
            # Shape: (n_rows, n_cols, n_gaussians)
            
            # Sum over Gaussians to get mixture
            matrix = torch.sum(gaussians, dim=2)  # Shape: (n_rows, n_cols)
        else:  # axis == 0, Row-wise
            # Create column indices
            x = torch.arange(self.n_cols, dtype=torch.float32, device=self.means_unconstrained.device)
            x = x.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_cols)
            means = means.unsqueeze(2)  # Shape: (n_rows, n_gaussians, 1)
            stds = stds.unsqueeze(2)  # Shape: (n_rows, n_gaussians, 1)
            scales = scales.unsqueeze(2)  # Shape: (n_rows, n_gaussians, 1)
            
            # Compute Gaussian for each component
            gaussians = scales * torch.exp(-0.5 * ((x - means) / stds) ** 2)
            # Shape: (n_rows, n_gaussians, n_cols)
            
            # Sum over Gaussians to get mixture
            matrix = torch.sum(gaussians, dim=1)  # Shape: (n_rows, n_cols)
        
        return matrix
    
    def __repr__(self):
        """String representation showing constrained parameter values."""
        mean_min, mean_max = self.mean_bounds
        means = mean_min + (mean_max - mean_min) * torch.sigmoid(self.means_unconstrained)
        std_min, std_max = self.std_bounds
        stds = std_min + (std_max - std_min) * torch.sigmoid(self.stds_unconstrained)
        scale_min, scale_max = self.scale_bounds
        scales = scale_min + (scale_max - scale_min) * torch.sigmoid(self.scales_unconstrained)
        
        return (f"{self.__class__.__name__}(shape={self.shape}, axis={self.axis}, n_gaussians={self.n_gaussians})\n"
                f"  means: {means.detach().cpu().numpy()}\n"
                f"  stds: {stds.detach().cpu().numpy()}\n"
                f"  scales: {scales.detach().cpu().numpy()}")

    def initialize_from_matrix(self, target_matrix):
        """Find n_gaussians best local maxima in each column/row and initialize parameters."""
        from scipy.signal import find_peaks
        target_matrix_np = target_matrix.detach().cpu().numpy()
        
        if self.axis == 1:  # Column-wise
            n_iters = self.n_cols
            axis_size = self.n_rows
        else:  # Row-wise
            n_iters = self.n_rows
            axis_size = self.n_cols
        
        for idx in range(n_iters):
            # Extract data
            if self.axis == 1:
                data = target_matrix_np[:, idx]
            else:
                data = target_matrix_np[idx, :]
            
            # Find local maxima
            peaks, _ = find_peaks(data)
            if len(peaks) < self.n_gaussians:
                # add top values as peaks if not enough local maxima (don't include the peaks found)
                additionale_data = np.copy(data)
                additionale_data[peaks] = -np.inf  # Exclude already found peaks
                additional_peaks = np.argsort(additionale_data)[-(self.n_gaussians - len(peaks)):]
                peaks = np.concatenate([peaks, additional_peaks])
            else:
                peaks = peaks[np.argsort(data[peaks])][-self.n_gaussians:]

            # Initialize means to peak positions, stds to some fraction of distance between peaks, scales to value at peaks
            means_init = peaks
            if len(peaks) > 1:
                std_value = np.diff(np.sort(peaks)).min() / 2.0
            else:
                std_value = axis_size / 10.0  # Arbitrary default if only one peak
            stds_init = np.full(self.n_gaussians, std_value)  # Create array of same std for all Gaussians
            scales_init = data[peaks]
            
            # Update parameters for this column/row
            # Apply inverse sigmoid (logit) transformation: unconstrained = logit(ratio) where ratio = (value - min) / (max - min)
            means_ratio = np.clip((means_init - self.mean_bounds[0]) / (self.mean_bounds[1] - self.mean_bounds[0]), 1e-7, 1 - 1e-7)
            self.means_unconstrained.data[idx] = torch.from_numpy(np.log(means_ratio / (1 - means_ratio))).float()
            
            stds_ratio = np.clip((stds_init - self.std_bounds[0]) / (self.std_bounds[1] - self.std_bounds[0]), 1e-7, 1 - 1e-7)
            self.stds_unconstrained.data[idx] = torch.from_numpy(np.log(stds_ratio / (1 - stds_ratio))).float()
            
            scales_ratio = np.clip((scales_init - self.scale_bounds[0]) / (self.scale_bounds[1] - self.scale_bounds[0]), 1e-7, 1 - 1e-7)
            self.scales_unconstrained.data[idx] = torch.from_numpy(np.log(scales_ratio / (1 - scales_ratio))).float()
        
        super().initialize_from_matrix(target_matrix)  # Call the base class method to run optimization after initialization

class SplineParameterization(MatrixParameterization):
    """
    Each column or row of the matrix is a cubic spline.
    Parameters: y-values at control points for each column/row.
    Uses sigmoid transformation to enforce value bounds.
    
    Args:
        n_control_points: int - number of control points for each spline
        axis: int - 0 for row-wise splines, 1 for column-wise splines (default: 1)
    """
    def __init__(self, shape, n_control_points=5, axis=1, value_bounds=[0, 10]):
        """
        Initialize spline parameterization.
        
        Args:
            shape: tuple (n_rows, n_cols) - shape of the matrix to generate
            n_control_points: int - number of control points for spline (default: 5)
            axis: int - 0 for row-wise, 1 for column-wise (default: 1)
            value_bounds: tuple (min, max) - bounds for spline y-values
            mean_bounds, std_bounds, scale_bounds: ignored (for API compatibility)
        """
        self.n_control_points = n_control_points
        self.axis = axis
        self.value_bounds = value_bounds
        super(SplineParameterization, self).__init__(shape)
    
    def _init_params(self):
        """Initialize unconstrained spline control point parameters."""
        # Determine dimensions based on axis
        if self.axis == 1:  # Column-wise (W matrix)
            n_splines = self.n_cols
            axis_size = self.n_rows
            default_value_max = 10.0
        else:  # axis == 0, Row-wise (H matrix)
            n_splines = self.n_rows
            axis_size = self.n_cols
            default_value_max = 1.0
        
        # Set default value bounds if not provided
        if self.value_bounds is None:
            self.value_bounds = (0, default_value_max)
        
        # Store unconstrained parameters for control point y-values
        # Shape: (n_splines, n_control_points)
        self.control_values_unconstrained = nn.Parameter(
            2 * (torch.rand(n_splines, self.n_control_points) - 0.5)
        )
        
        # Control point x-positions are fixed and evenly spaced
        self.control_x = np.linspace(0, axis_size - 1, self.n_control_points)
        self.axis_size = axis_size
    
    @property
    def params(self):
        """Expose unconstrained parameters."""
        return {
            'control_values_unconstrained': self.control_values_unconstrained
        }
    
    def set_params(self, control_values_unconstrained):
        self.control_values_unconstrained.data = control_values_unconstrained   
    
    def forward(self):
        """
        Generate matrix where each column/row is a cubic spline using torchcubicspline (differentiable).
        Uses batching to process all splines simultaneously for better performance.
        
        Returns:
            torch.Tensor: Matrix of shape (n_rows, n_cols)
        """
        # Transform unconstrained parameters to bounded values using sigmoid
        value_min, value_max = self.value_bounds
        control_values = value_min + (value_max - value_min) * torch.sigmoid(
            self.control_values_unconstrained
        )
        
        if self.axis == 1:  # Column-wise
            n_splines = self.n_cols
            output_size = self.n_rows
        else:  # Row-wise
            n_splines = self.n_rows
            output_size = self.n_cols
        
        device = control_values.device
        
        # Control point x-positions (fixed, evenly spaced) - 1D tensor
        x_ctrl = torch.from_numpy(self.control_x).float().to(device)
        
        # Evaluation points (where we want to evaluate the spline) - 1D tensor
        x_eval = torch.arange(output_size, dtype=torch.float32, device=device)
        
        # Batch process ALL splines at once
        # control_values shape: (n_splines, n_control_points)
        # Add channel dimension for torchcubicspline: (n_splines, n_control_points, 1)
        y_ctrl_batch = control_values.unsqueeze(-1)  # Shape: (n_splines, n_control_points, 1)
        
        # Compute spline coefficients for all splines in batch
        # t is 1D, x has batch dimension: (n_splines, n_control_points, 1)
        coeffs = natural_cubic_spline_coeffs(x_ctrl, y_ctrl_batch)
        
        # Create spline object
        spline = NaturalCubicSpline(coeffs)
        
        # Evaluate all splines at evaluation points
        # Output shape: (n_splines, output_size, 1)
        spline_values = spline.evaluate(x_eval)
        
        # Remove channel dimension: (n_splines, output_size)
        spline_values = spline_values.squeeze(-1)
        
        # Clip to bounds
        # spline_values = torch.clamp(spline_values, value_min, value_max)
        
        # Arrange into output matrix based on axis
        if self.axis == 1:  # Column-wise: transpose to (output_size, n_splines)
            matrix = spline_values.T  # Shape: (n_rows, n_cols)
        else:  # Row-wise: already correct shape (n_rows, n_cols)
            matrix = spline_values
        
        return matrix
    
    def __repr__(self):
        """String representation showing constrained parameter values."""
        value_min, value_max = self.value_bounds
        control_values = value_min + (value_max - value_min) * torch.sigmoid(self.control_values_unconstrained)
        
        return (f"{self.__class__.__name__}(shape={self.shape}, axis={self.axis}, n_control_points={self.n_control_points})\n"
                f"  control_values: {control_values.detach().cpu().numpy()}")
    
    def initialize_from_matrix(self, target_matrix):
        # just put in each control point the value of the target matrix at the corresponding position (evenly spaced)
        target_matrix_np = target_matrix.detach().cpu().numpy()
        if self.axis == 1:  # Column-wise
            for idx in range(self.n_cols):
                control_values_init = np.interp(self.control_x, np.arange(self.n_rows), target_matrix_np[:, idx])
                value_min, value_max = self.value_bounds
                control_values_init = np.clip(control_values_init, value_min + 1e-3, value_max - 1e-3)
                control_values_ratio = (control_values_init - value_min) / (value_max - value_min)
                self.control_values_unconstrained.data[idx] = torch.from_numpy(np.log(control_values_ratio / (1 - control_values_ratio))).float()
        else:  # Row-wise
            for idx in range(self.n_rows):
                control_values_init = np.interp(self.control_x, np.arange(self.n_cols), target_matrix_np[idx, :])
                value_min, value_max = self.value_bounds
                control_values_init = np.clip(control_values_init, value_min + 1e-3, value_max - 1e-3)
                control_values_ratio = (control_values_init - value_min) / (value_max - value_min)
                self.control_values_unconstrained.data[idx] = torch.from_numpy(np.log(control_values_ratio / (1 - control_values_ratio))).float()
        super().initialize_from_matrix(target_matrix)

class GeneralizedGaussianParameterization(MatrixParameterization):
    """
    Each column or row is a Generalized Gaussian distribution.
    Generalized Gaussian PDF: f(x) ~ exp(-0.5 * (|x-mean|/std)^beta)
    
    Parameters: mean (location), std (scale), beta (shape), amplitude.
    beta = 2 gives normal distribution (identical to GaussianParameterization).
    beta = 1 gives Laplace-like distribution.
    """
    def __init__(self, shape, axis=1, mean_bounds=None, std_bounds=None, 
                 beta_bounds=None, scale_bounds=None):
        """
        Initialize Generalized Gaussian parameterization.
        
        Args:
            shape: tuple (n_rows, n_cols) - shape of the matrix to generate
            axis: int - 0 for row-wise, 1 for column-wise (default: 1)
            mean_bounds: tuple (min, max) - bounds for mean parameter
            std_bounds: tuple (min, max) - bounds for std (scale) parameter
            beta_bounds: tuple (min, max) - bounds for shape parameter
            scale_bounds: tuple (min, max) - bounds for amplitude
        """
        self.axis = axis
        self.beta_bounds = beta_bounds
        super(GeneralizedGaussianParameterization, self).__init__(shape, mean_bounds, std_bounds, scale_bounds)
    
    def _init_params(self):
        """Initialize unconstrained Generalized Gaussian parameters."""
        if self.axis == 1:  # Column-wise
            n_distributions = self.n_cols
            axis_size = self.n_rows
            default_scale_max = 10.0
        else:  # Row-wise
            n_distributions = self.n_rows
            axis_size = self.n_cols
            default_scale_max = 1.0
        
        # Set default bounds
        if self.mean_bounds is None:
            self.mean_bounds = (0, axis_size)
        if self.std_bounds is None:
            self.std_bounds = (0.5, axis_size / 3)
        if self.beta_bounds is None:
            self.beta_bounds = (0.5, 4.0)
        if self.scale_bounds is None:
            self.scale_bounds = (0, default_scale_max)
        
        self.means_unconstrained = nn.Parameter(2 * (torch.rand(n_distributions) - 0.5))
        self.stds_unconstrained = nn.Parameter(2 * (torch.rand(n_distributions) - 0.5))
        self.beta_unconstrained = nn.Parameter(2 * (torch.rand(n_distributions) - 0.5))
        self.scales_unconstrained = nn.Parameter(2 * (torch.rand(n_distributions) - 0.5))
    
    @property
    def params(self):
        """Expose unconstrained parameters."""
        return {
            'means_unconstrained': self.means_unconstrained,
            'stds_unconstrained': self.stds_unconstrained,
            'beta_unconstrained': self.beta_unconstrained,
            'scales_unconstrained': self.scales_unconstrained
        }
    
    def set_params(self, means_unconstrained=None, stds_unconstrained=None, 
                   beta_unconstrained=None, scales_unconstrained=None):
        """Set parameters directly."""
        if means_unconstrained is not None:
            self.means_unconstrained.data = means_unconstrained
        if stds_unconstrained is not None:
            self.stds_unconstrained.data = stds_unconstrained
        if beta_unconstrained is not None:
            self.beta_unconstrained.data = beta_unconstrained
        if scales_unconstrained is not None:
            self.scales_unconstrained.data = scales_unconstrained
    
    def forward(self):
        """Generate matrix where each column/row is a Generalized Gaussian distribution."""
        # Transform parameters
        mean_min, mean_max = self.mean_bounds
        means = mean_min + (mean_max - mean_min) * torch.sigmoid(self.means_unconstrained)
        
        std_min, std_max = self.std_bounds
        stds = std_min + (std_max - std_min) * torch.sigmoid(self.stds_unconstrained) + 1e-6
        
        beta_min, beta_max = self.beta_bounds
        beta = beta_min + (beta_max - beta_min) * torch.sigmoid(self.beta_unconstrained)
        
        scale_min, scale_max = self.scale_bounds
        scales = scale_min + (scale_max - scale_min) * torch.sigmoid(self.scales_unconstrained)
        
        if self.axis == 1:  # Column-wise
            x = torch.arange(self.n_rows, dtype=torch.float32, device=means.device)
            x = x.unsqueeze(1)
            means = means.unsqueeze(0)
            stds = stds.unsqueeze(0)
            beta = beta.unsqueeze(0)
            scales = scales.unsqueeze(0)
        else:  # Row-wise
            x = torch.arange(self.n_cols, dtype=torch.float32, device=means.device)
            x = x.unsqueeze(0)
            means = means.unsqueeze(1)
            stds = stds.unsqueeze(1)
            beta = beta.unsqueeze(1)
            scales = scales.unsqueeze(1)
        
        # Generalized Gaussian: exp(-0.5 * (|x-mean|/std)^beta)
        # When beta=2, this is identical to regular Gaussian
        abs_diff = torch.abs(x - means)
        matrix = scales * torch.exp(-0.5 * ((abs_diff / stds) ** beta))
        
        return matrix
    
    def __repr__(self):
        """String representation showing constrained parameter values."""
        mean_min, mean_max = self.mean_bounds
        means = mean_min + (mean_max - mean_min) * torch.sigmoid(self.means_unconstrained)
        std_min, std_max = self.std_bounds
        stds = std_min + (std_max - std_min) * torch.sigmoid(self.stds_unconstrained)
        beta_min, beta_max = self.beta_bounds
        beta = beta_min + (beta_max - beta_min) * torch.sigmoid(self.beta_unconstrained)
        scale_min, scale_max = self.scale_bounds
        scales = scale_min + (scale_max - scale_min) * torch.sigmoid(self.scales_unconstrained)
        
        return (f"{self.__class__.__name__}(shape={self.shape}, axis={self.axis})\n"
                f"  means: {means.detach().cpu().numpy()}\n"
                f"  stds: {stds.detach().cpu().numpy()}\n"
                f"  beta (shape): {beta.detach().cpu().numpy()}\n"
                f"  scales: {scales.detach().cpu().numpy()}")

    def initialize_from_matrix(self, target_matrix):
        """Initialize parameters by finding Gaussian characteristics in each column/row."""
        target_matrix_np = target_matrix.detach().cpu().numpy()
        
        if self.axis == 1:  # Column-wise
            n_iters = self.n_cols
            axis_size = self.n_rows
        else:  # Row-wise
            n_iters = self.n_rows
            axis_size = self.n_cols
        
        for idx in range(n_iters):
            # Extract data
            if self.axis == 1:
                data = target_matrix_np[:, idx]
            else:
                data = target_matrix_np[idx, :]
            
            # Find mean as position of maximum value
            mean_init = float(np.argmax(data))
            
            # Scale is the maximum value
            scale_init = float(np.max(data))
            
            # Estimate std from weighted variance
            x = np.arange(axis_size)
            # Use data as weights to compute weighted variance
            weights = np.maximum(data, 0)  # Ensure non-negative
            if weights.sum() > 0:
                weights = weights / weights.sum()
                weighted_mean = np.sum(x * weights)
                weighted_var = np.sum(weights * (x - weighted_mean) ** 2)
                std_init = float(np.sqrt(weighted_var) + 1e-3)  # Add small value to avoid zero
            else:
                std_init = axis_size / 10.0
            
            # Apply inverse sigmoid (logit) transformation
            mean_ratio = np.clip((mean_init - self.mean_bounds[0]) / (self.mean_bounds[1] - self.mean_bounds[0]), 1e-7, 1 - 1e-7)
            self.means_unconstrained.data[idx] = torch.tensor(np.log(mean_ratio / (1 - mean_ratio)), dtype=torch.float32)
            
            std_ratio = np.clip((std_init - self.std_bounds[0]) / (self.std_bounds[1] - self.std_bounds[0]), 1e-7, 1 - 1e-7)
            self.stds_unconstrained.data[idx] = torch.tensor(np.log(std_ratio / (1 - std_ratio)), dtype=torch.float32)
            
            scale_ratio = np.clip((scale_init - self.scale_bounds[0]) / (self.scale_bounds[1] - self.scale_bounds[0]), 1e-7, 1 - 1e-7)
            self.scales_unconstrained.data[idx] = torch.tensor(np.log(scale_ratio / (1 - scale_ratio)), dtype=torch.float32)
        
        super().initialize_from_matrix(target_matrix)  # Call the base class method to run optimization after initialization

class MixtureOfGeneralizedGaussiansParameterization(MatrixParameterization):
    """
    Each column or row of the matrix is a mixture of Generalized Gaussians.
    Generalized Gaussian PDF: f(x) ~ exp(-0.5 * (|x-mean|/std)^beta)
    
    Parameters: means, stds, betas (shape), scales for each Gaussian in each column/row.
    Uses sigmoid transformation to enforce bounds.
    
    Args:
        n_gaussians: int - number of Generalized Gaussians in the mixture for each column/row
        axis: int - 0 for row-wise Gaussians, 1 for column-wise Gaussians (default: 1)
    """
    def __init__(self, shape, n_gaussians=2, axis=1, mean_bounds=None, std_bounds=None, 
                 beta_bounds=None, scale_bounds=None):
        """
        Initialize mixture of Generalized Gaussians parameterization.
        
        Args:
            shape: tuple (n_rows, n_cols) - shape of the matrix to generate
            n_gaussians: int - number of Generalized Gaussians in the mixture for each column/row
            axis: int - 0 for row-wise, 1 for column-wise (default: 1)
            mean_bounds: tuple (min, max) - bounds for mean parameters
            std_bounds: tuple (min, max) - bounds for std (scale) parameters
            beta_bounds: tuple (min, max) - bounds for shape parameters
            scale_bounds: tuple (min, max) - bounds for amplitude parameters
        """
        self.n_gaussians = n_gaussians
        self.axis = axis
        self.beta_bounds = beta_bounds
        super(MixtureOfGeneralizedGaussiansParameterization, self).__init__(
            shape, mean_bounds, std_bounds, scale_bounds
        )
    
    def _init_params(self):
        """Initialize unconstrained mixture of Generalized Gaussians parameters."""
        # Set default bounds based on axis
        if self.axis == 1:  # Column-wise (W matrix)
            n_components = self.n_cols
            axis_size = self.n_rows
            default_scale_max = 10.0
        else:  # axis == 0, Row-wise (H matrix)
            n_components = self.n_rows
            axis_size = self.n_cols
            default_scale_max = 1.0
        
        # Set default bounds if not provided
        if self.mean_bounds is None:
            self.mean_bounds = (0, axis_size)
        if self.std_bounds is None:
            self.std_bounds = (0.5, axis_size / 3)
        if self.beta_bounds is None:
            self.beta_bounds = (0.5, 4.0)
        if self.scale_bounds is None:
            self.scale_bounds = (0, default_scale_max)
        
        # Store unconstrained parameters
        # Shape: (n_components, n_gaussians) for each parameter
        self.means_unconstrained = nn.Parameter(2 * (torch.rand(n_components, self.n_gaussians) - 0.5))
        self.stds_unconstrained = nn.Parameter(2 * (torch.rand(n_components, self.n_gaussians) - 0.5))
        self.beta_unconstrained = nn.Parameter(2 * (torch.rand(n_components, self.n_gaussians) - 0.5))
        self.scales_unconstrained = nn.Parameter(2 * (torch.rand(n_components, self.n_gaussians) - 0.5))
    
    @property
    def params(self):
        """Expose unconstrained parameters."""
        return {
            'means_unconstrained': self.means_unconstrained,
            'stds_unconstrained': self.stds_unconstrained,
            'beta_unconstrained': self.beta_unconstrained,
            'scales_unconstrained': self.scales_unconstrained
        }
    
    def set_params(self, means_unconstrained=None, stds_unconstrained=None, 
                   beta_unconstrained=None, scales_unconstrained=None):
        """Set parameters directly (for testing/initialization)."""
        if means_unconstrained is not None:
            self.means_unconstrained.data = means_unconstrained
        if stds_unconstrained is not None:
            self.stds_unconstrained.data = stds_unconstrained
        if beta_unconstrained is not None:
            self.beta_unconstrained.data = beta_unconstrained
        if scales_unconstrained is not None:
            self.scales_unconstrained.data = scales_unconstrained
    
    def forward(self):
        """
        Generate matrix where each column/row is a mixture of Generalized Gaussians.
        
        Returns:
            torch.Tensor: Matrix of shape (n_rows, n_cols)
        """
        # Transform unconstrained parameters to bounded values using sigmoid
        mean_min, mean_max = self.mean_bounds
        means = mean_min + (mean_max - mean_min) * torch.sigmoid(self.means_unconstrained)
        
        std_min, std_max = self.std_bounds
        stds = std_min + (std_max - std_min) * torch.sigmoid(self.stds_unconstrained) + 1e-6
        
        beta_min, beta_max = self.beta_bounds
        beta = beta_min + (beta_max - beta_min) * torch.sigmoid(self.beta_unconstrained)
        
        scale_min, scale_max = self.scale_bounds
        scales = scale_min + (scale_max - scale_min) * torch.sigmoid(self.scales_unconstrained)
        
        if self.axis == 1:  # Column-wise
            # Create row indices
            x = torch.arange(self.n_rows, dtype=torch.float32, device=self.means_unconstrained.device)
            x = x.unsqueeze(1).unsqueeze(1)  # Shape: (n_rows, 1, 1)
            means = means.unsqueeze(0)  # Shape: (1, n_cols, n_gaussians)
            stds = stds.unsqueeze(0)  # Shape: (1, n_cols, n_gaussians)
            beta = beta.unsqueeze(0)  # Shape: (1, n_cols, n_gaussians)
            scales = scales.unsqueeze(0)  # Shape: (1, n_cols, n_gaussians)
            
            # Compute Generalized Gaussian for each component
            abs_diff = torch.abs(x - means)
            gaussians = scales * torch.exp(-0.5 * ((abs_diff / stds) ** beta))
            # Shape: (n_rows, n_cols, n_gaussians)
            
            # Sum over Gaussians to get mixture
            matrix = torch.sum(gaussians, dim=2)  # Shape: (n_rows, n_cols)
        else:  # axis == 0, Row-wise
            # Create column indices
            x = torch.arange(self.n_cols, dtype=torch.float32, device=self.means_unconstrained.device)
            x = x.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_cols)
            means = means.unsqueeze(2)  # Shape: (n_rows, n_gaussians, 1)
            stds = stds.unsqueeze(2)  # Shape: (n_rows, n_gaussians, 1)
            beta = beta.unsqueeze(2)  # Shape: (n_rows, n_gaussians, 1)
            scales = scales.unsqueeze(2)  # Shape: (n_rows, n_gaussians, 1)
            
            # Compute Generalized Gaussian for each component
            abs_diff = torch.abs(x - means)
            gaussians = scales * torch.exp(-0.5 * ((abs_diff / stds) ** beta))
            # Shape: (n_rows, n_gaussians, n_cols)
            
            # Sum over Gaussians to get mixture
            matrix = torch.sum(gaussians, dim=1)  # Shape: (n_rows, n_cols)
        
        return matrix
    
    def __repr__(self):
        """String representation showing constrained parameter values."""
        mean_min, mean_max = self.mean_bounds
        means = mean_min + (mean_max - mean_min) * torch.sigmoid(self.means_unconstrained)
        std_min, std_max = self.std_bounds
        stds = std_min + (std_max - std_min) * torch.sigmoid(self.stds_unconstrained)
        beta_min, beta_max = self.beta_bounds
        beta = beta_min + (beta_max - beta_min) * torch.sigmoid(self.beta_unconstrained)
        scale_min, scale_max = self.scale_bounds
        scales = scale_min + (scale_max - scale_min) * torch.sigmoid(self.scales_unconstrained)
        
        return (f"{self.__class__.__name__}(shape={self.shape}, axis={self.axis}, n_gaussians={self.n_gaussians})\n"
                f"  means: {means.detach().cpu().numpy()}\n"
                f"  stds: {stds.detach().cpu().numpy()}\n"
                f"  beta (shape): {beta.detach().cpu().numpy()}\n"
                f"  scales: {scales.detach().cpu().numpy()}")

    def initialize_from_matrix(self, target_matrix):
        """Find n_gaussians best local maxima in each column/row and initialize parameters."""
        from scipy.signal import find_peaks
        target_matrix_np = target_matrix.detach().cpu().numpy()
        
        if self.axis == 1:  # Column-wise
            n_iters = self.n_cols
            axis_size = self.n_rows
        else:  # Row-wise
            n_iters = self.n_rows
            axis_size = self.n_cols
        
        for idx in range(n_iters):
            # Extract data
            if self.axis == 1:
                data = target_matrix_np[:, idx]
            else:
                data = target_matrix_np[idx, :]
            
            # Find local maxima
            peaks, _ = find_peaks(data)
            if len(peaks) < self.n_gaussians:
                # add top values as peaks if not enough local maxima (don't include the peaks found)
                additionale_data = np.copy(data)
                additionale_data[peaks] = -np.inf  # Exclude already found peaks
                additional_peaks = np.argsort(additionale_data)[-(self.n_gaussians - len(peaks)):]
                peaks = np.concatenate([peaks, additional_peaks])
            else:
                peaks = peaks[np.argsort(data[peaks])][-self.n_gaussians:]
            
            # Initialize means to peak positions
            means_init = peaks
            
            # Initialize stds to some fraction of distance between peaks
            if len(peaks) > 1:
                std_value = np.diff(np.sort(peaks)).min() / 2.0
            else:
                std_value = axis_size / 10.0  # Arbitrary default if only one peak
            stds_init = np.full(self.n_gaussians, std_value)
            
            # Initialize scales to value at peaks
            scales_init = data[peaks]
            
            # Initialize beta to middle of range (e.g., 2.0 for normal Gaussian-like behavior)
            beta_init = np.full(self.n_gaussians, 2.0)
            
            # Update parameters for this column/row
            # Apply inverse sigmoid (logit) transformation
            means_ratio = np.clip((means_init - self.mean_bounds[0]) / (self.mean_bounds[1] - self.mean_bounds[0]), 1e-7, 1 - 1e-7)
            self.means_unconstrained.data[idx] = torch.from_numpy(np.log(means_ratio / (1 - means_ratio))).float()
            
            stds_ratio = np.clip((stds_init - self.std_bounds[0]) / (self.std_bounds[1] - self.std_bounds[0]), 1e-7, 1 - 1e-7)
            self.stds_unconstrained.data[idx] = torch.from_numpy(np.log(stds_ratio / (1 - stds_ratio))).float()
            
            beta_ratio = np.clip((beta_init - self.beta_bounds[0]) / (self.beta_bounds[1] - self.beta_bounds[0]), 1e-7, 1 - 1e-7)
            self.beta_unconstrained.data[idx] = torch.from_numpy(np.log(beta_ratio / (1 - beta_ratio))).float()
            
            scales_ratio = np.clip((scales_init - self.scale_bounds[0]) / (self.scale_bounds[1] - self.scale_bounds[0]), 1e-7, 1 - 1e-7)
            self.scales_unconstrained.data[idx] = torch.from_numpy(np.log(scales_ratio / (1 - scales_ratio))).float()
        
        super().initialize_from_matrix(target_matrix)  # Call the base class method to run optimization after initialization


# ========================================
# Parameterized NMF Solver
# ========================================

class ParameterizedNMFSolver:
    """
    NMF Solver using parameterized matrices for W and H.
    """
    def __init__(self, W_param: MatrixParameterization, H_param: MatrixParameterization, 
                 A_observed: torch.Tensor, lambda_penalty: float = 0.1, device: str = 'cpu'):
        """
        Initialize the NMF solver.
        
        Args:
            W_param: MatrixParameterization - parameterization for W matrix
            H_param: MatrixParameterization - parameterization for H matrix
            A_observed: torch.Tensor - observed matrix to factorize
            lambda_penalty: float - penalty weight for sum constraint
            device: str or torch.device - device to run on
        """
        self.W_param = W_param.to(device)
        self.H_param = H_param.to(device)
        self.A_observed = A_observed.to(device)
        self.lambda_penalty = lambda_penalty
        self.device = device
        
        # Collect all parameters from both parameterizations
        self.params = list(self.W_param.parameters()) + list(self.H_param.parameters())
        
        # Tracking
        self.best_loss = float('inf')
        self.best_W = None
        self.best_H = None
        self.best_params = None
        self.loss_history = []
    
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
        
        # NMF iterations with sum-to-one constraint on H
        for iteration in range(n_nmf_iterations):
            # Update H
            WtA = torch.mm(W_init.T, self.A_observed)
            WtW = torch.mm(W_init.T, W_init)
            H_init = H_init * WtA / (torch.mm(WtW, H_init) + 1e-10)
            # Normalize H columns to sum to 1
            H_init = H_init / (H_init.sum(dim=0, keepdim=True) + 1e-10)
            
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
    
    def compute_loss(self):
        """
        Compute the total loss: fitting term + penalty term.
        
        Returns:
            tuple: (total_loss, fitting_loss, penalty_loss)
        """
        # Get current matrices
        W_current = self.W_param.matrix
        H_current = self.H_param.matrix
        
        # Fitting term: ||A - W @ H||^2
        A_recon = torch.mm(W_current, H_current)
        fitting_loss = F.mse_loss(A_recon, self.A_observed)
        
        # Penalty term: encourage columns of H to sum to 1
        # sum_penalty = lambda * mean((sum(H, dim=0) - 1)^2)
        penalty_loss = self.lambda_penalty * torch.mean(
            torch.square(torch.sum(H_current, dim=0) - 1.0)
        )

        total_loss = fitting_loss + penalty_loss
        
        return total_loss, fitting_loss, penalty_loss
    
    def solve(self, n_iterations=1000, lr=0.01, print_every=10, use_scheduler=True, 
              nmf_init=False, n_nmf_iterations=100):
        """
        Solve the NMF problem using gradient-based optimization.
        
        Args:
            n_iterations: int - number of optimization iterations
            lr: float - initial learning rate
            print_every: int - print loss every N iterations
            use_scheduler: bool - whether to use learning rate scheduler (Adam only)
            nmf_init: bool - whether to initialize with standard NMF
            n_nmf_iterations: int - number of NMF iterations if nmf_init=True
        
        Returns:
            tuple: (W_final, H_final, loss_history)
        """
        # Initialize with NMF if requested
        if nmf_init:
            self.initialize_with_nmf(n_nmf_iterations=n_nmf_iterations)
        
        # Create optimizer
        optimizer = optim.Adam(self.params, lr=lr, weight_decay=1e-5)
        
        # Create learning rate scheduler (Adam only)
        if use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=500, min_lr=1e-6
            )
        
        print(f"Starting optimization for {n_iterations} iterations...")
        print(f"Lambda penalty: {self.lambda_penalty}")
        if use_scheduler:
            print(f"LR Scheduler: Enabled (ReduceLROnPlateau)")
        print("-" * 70)
        
        current_lr = lr
        iteration = 0
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # Compute loss
            total_loss, fitting_loss, penalty_loss = self.compute_loss()
            
            # Backward pass
            total_loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Track loss
            loss_val = total_loss.item()
            self.loss_history.append(loss_val)
            
            # Update learning rate scheduler
            if use_scheduler:
                scheduler.step(loss_val)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    print(f"\nLR reduced: {current_lr:.2e} -> {new_lr:.2e} at iteration {iteration+1}")
                    current_lr = new_lr
            
            # Update best parameters if this is the best loss so far
            if loss_val < self.best_loss:
                self.best_loss = loss_val
                self.best_W = self.W_param.matrix.detach().clone()
                self.best_H = self.H_param.matrix.detach().clone()
                self.best_params = {
                    'W_params': {k: v.detach().clone() for k, v in self.W_param.params.items()},
                    'H_params': {k: v.detach().clone() for k, v in self.H_param.params.items()},
                    'iteration': iteration
                }
            
            # Print progress
            if (iteration + 1) % print_every == 0 or iteration == 0:
                lr_str = f"LR: {current_lr:.2e} | " if use_scheduler else ""
                print(f"Iter {iteration+1:5d} | {lr_str}Total Loss: {loss_val:.6f} | "
                        f"Fitting: {fitting_loss.item():.6f} | "
                        f"Penalty: {penalty_loss.item():.6f} | "
                        f"Best: {self.best_loss:.6f}")
    
        print("-" * 70)
        print(f"Optimization complete!")
        print(f"Final loss: {self.loss_history[-1]:.6f}")
        print(f"Best loss: {self.best_loss:.6f} (at iteration {self.best_params['iteration']+1})")
        print(f"Final learning rate: {current_lr:.2e}")
        
        return self.best_W, self.best_H, self.loss_history
    
    def get_factors(self):
        """Get the best W and H matrices."""
        return self.best_W, self.best_H


# ========================================
# Utility Functions
# ========================================
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
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f'{prefix}_concentration_profiles.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_results(W_opt, H_opt, An, W_true, H_true, uv_wavelengths, loss_history=None, prefix=""):
    """
    Plot the results of the matrix factorization with proper scaling
    """
    if isinstance(W_opt, torch.Tensor):
        W_opt = W_opt.cpu().detach()
        H_opt = H_opt.cpu().detach()
    elif isinstance(W_opt, GaussianParameterization):
        # Align components to match the true factors for better comparison
        alignment = H_opt.params['means_unconstrained'].argsort()
        W_opt = W_opt.matrix.detach().cpu()
        H_opt = H_opt.matrix.detach().cpu()
        W_opt = W_opt[:, alignment]
        H_opt = H_opt[alignment, :]
    else:
        W_opt = W_opt.matrix.detach().cpu()
        H_opt = H_opt.matrix.detach().cpu()
        # Sort by the position of maximum (argmax) along each row of H_opt
        # This aligns components by their peak position, not peak value
        alignment = H_opt.argmax(dim=1).argsort()
        W_opt = W_opt[:, alignment]
        H_opt = H_opt[alignment, :]
    k = W_opt.shape[1]
    
    An_recon = W_opt @ H_opt
    loss_uv = F.mse_loss(An_recon, An)
    
    # Convert to numpy for plotting
    W_opt = W_opt.cpu().numpy()
    H_opt = H_opt.cpu().numpy()
    W_true = W_true.cpu().numpy()
    H_true = H_true.cpu().numpy()
    
    # Plot loss history
    if loss_history is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss History')
        plt.grid(True)
        plt.savefig(os.path.join(out_folder, f'{prefix}_loss_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    # Plot UV-vis components
    plt.figure(figsize=(15, 5*k))
    for i in range(k):
        plt.subplot(k, 1, i+1)
        plt.plot(uv_wavelengths, W_true[:, i], 'b-', label='True', linewidth=2)
        plt.plot(uv_wavelengths, W_opt[:, i], 'r--', 
                label='Estimated', linewidth=2)
        plt.title(f'UV-vis Component {i+1}')
        plt.xlabel('Wavelength (eV)')
        plt.ylabel('Absorbance')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f'{prefix}_spectrum_components.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot concentration profiles
    plt.figure(figsize=(15, 5*k))
    for i in range(k):
        plt.subplot(k, 1, i+1)
        plt.plot(H_true[i], 'b-', label='True', linewidth=2)
        plt.plot(H_opt[i], 'r--', 
                label='Estimated', linewidth=2)
        plt.title(f'Concentration Profile {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f'{prefix}_concentration_profiles.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    #plot An, An_recon, and the difference
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(An.cpu().numpy(), aspect='auto', origin='lower')
    plt.title('Observed An')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(An_recon.cpu().numpy(), aspect='auto', origin='lower')
    plt.title('Reconstructed An')       
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow((An - An_recon).cpu().numpy(), aspect='auto', origin='lower', cmap='bwr')
    plt.title('Difference (An - An_recon)')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f'{prefix}_An_reconstruction.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nReconstruction Errors:")
    print(f"MSE: {loss_uv:.6f}")
    print(f"Max deviation: {np.max(np.abs(np.sum(H_opt, axis=0) - 1.0)):.6f}")

def main():
    # Load UV-vis data
    df_An = pd.read_csv(data_path + "An.csv", header=None)
    uv_wavelengths = df_An.iloc[1:, 0].values.astype(float)
    An = torch.from_numpy(df_An.iloc[1:, 1:].values.astype(float)).to(torch.float32)
    # An += 1e-2 * torch.randn_like(An)  # Add small noise  - robustness test
    
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
    
    W_param = MixtureOfGaussiansParameterization(
        shape=(n_wavelengths, k_components), 
        n_gaussians=2,
        axis=1,  # Column-wise for W matrix
        mean_bounds=(0, n_wavelengths),
        std_bounds=(0.1, n_wavelengths / 5),
        scale_bounds=(0, 3.0)
    )
    # W_param = SplineParameterization(
    #     shape=(n_wavelengths, k_components), 
    #     n_control_points=30,
    #     value_bounds=(0, 4.0)
    # )
    
    # H_param = GaussianParameterization(
    #     shape=(k_components, n_timepoints),
    #     axis=0,  # Row-wise for H matrix
    #     mean_bounds=(0, n_timepoints),
    #     std_bounds=(0.1, n_timepoints / 8),
    #     scale_bounds=(0, 1.0)
    # )
    H_param = GeneralizedGaussianParameterization(
        shape=(k_components, n_timepoints),
        axis=0,
        mean_bounds=(0, n_timepoints),
        std_bounds=(1, 15.0),
        beta_bounds=(1.3, 4.0),
        scale_bounds=(0, 1.0)
    )
    
    model = ParameterizedNMFSolver(
        W_param=W_param,
        H_param=H_param,
        A_observed=An,
        lambda_penalty=1,
        device=device
    )
    
    W, H, loss_history = model.solve(
        n_iterations=10000,
        lr=0.01,
        print_every=1000,
        nmf_init=True, 
    )
    W_param.set_params(**model.best_params['W_params'])
    H_param.set_params(**model.best_params['H_params'])
    
    
    plot_results(W_param, H_param, An, W_true, H_true, uv_wavelengths, loss_history)
    print("\nFinal Parameter Values:")
    # print("W parameters:")
    # print(W_param)
    print("\nH parameters:")
    print(H_param)

if __name__ == "__main__":
    main() 