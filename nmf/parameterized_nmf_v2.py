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
from scipy.signal import find_peaks

# ========================================
# Abstract Parameterization Classes
# ========================================

class MatrixParameterization(ABC, nn.Module):
    """
    Abstract class for matrix parameterization.
    Subclasses should implement how to generate a matrix from parameters.
    """
    def __init__(self, shape):
        """
        Initialize parameterization with matrix shape.
        
        Args:
            shape: tuple (n_rows, n_cols) - shape of the matrix to generate
        """
        super(MatrixParameterization, self).__init__()
        self.shape = shape
        self.n_rows, self.n_cols = shape
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
    
    def matrix(self):
        """Get the current matrix."""
        return self.forward()

    def initialize_from_matrix(self, target_matrix, n_iterations=1000, lr=0.1, patience=100):
        """
        Initialize parameters by fitting to a target matrix using optimization.
        
        Args:
            target_matrix: torch.Tensor - matrix to fit parameters to
            n_iterations: int - number of optimization iterations
            lr: float - learning rate for optimization
        """
        optimizer = optim.AdamW(self.parameters(), lr=lr)
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
            # print(f"Iteration {iteration}, Loss: {loss.item():.6f}")
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
        self.mean_bounds = mean_bounds
        self.std_bounds = std_bounds
        self.scale_bounds = scale_bounds
        super(GaussianParameterization, self).__init__(shape)
    
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
        self.mean_bounds = mean_bounds
        self.std_bounds = std_bounds
        self.scale_bounds = scale_bounds
        super(MixtureOfGaussiansParameterization, self).__init__(shape)
    
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

class UnimodalSplineEnvelopeParameterization(MatrixParameterization):
    """
    Unimodal parameterization using positive spline multiplied by a Gaussian envelope.
    
    Each column or row is: spline(x) × exp(-(x - μ)² / σ²)
    
    This enforces unimodality by multiplying a flexible spline with a unimodal Gaussian envelope.
    
    Args:
        n_control_points: int - number of control points for the base spline
        axis: int - 0 for row-wise, 1 for column-wise (default: 1)
    """
    def __init__(self, shape, n_control_points=5, axis=1, value_bounds=None, 
                 mean_bounds=None, std_bounds=None):
        """
        Initialize unimodal spline-envelope parameterization.
        
        Args:
            shape: tuple (n_rows, n_cols) - shape of the matrix to generate
            n_control_points: int - number of control points for base spline (default: 5)
            axis: int - 0 for row-wise, 1 for column-wise (default: 1)
            value_bounds: tuple (min, max) - bounds for spline y-values (before envelope)
            mean_bounds: tuple (min, max) - bounds for Gaussian envelope mean
            std_bounds: tuple (min, max) - bounds for Gaussian envelope std
        """
        self.n_control_points = n_control_points
        self.axis = axis
        self.value_bounds = value_bounds
        self.mean_bounds_custom = mean_bounds
        self.std_bounds_custom = std_bounds
        super(UnimodalSplineEnvelopeParameterization, self).__init__(shape)
    
    def _init_params(self):
        """Initialize parameters for spline and Gaussian envelope."""
        # Determine dimensions based on axis
        if self.axis == 1:  # Column-wise (W matrix)
            n_components = self.n_cols
            axis_size = self.n_rows
            default_value_max = 10.0
        else:  # axis == 0, Row-wise (H matrix)
            n_components = self.n_rows
            axis_size = self.n_cols
            default_value_max = 1.0
        
        # Set default bounds
        if self.value_bounds is None:
            self.value_bounds = (0, default_value_max)
        if self.mean_bounds_custom is None:
            self.mean_bounds_custom = (0, axis_size)
        if self.std_bounds_custom is None:
            self.std_bounds_custom = (axis_size / 10, axis_size / 2)
        
        # Spline control point y-values (unconstrained)
        # Shape: (n_components, n_control_points)
        self.control_values_unconstrained = nn.Parameter(
            2 * (torch.rand(n_components, self.n_control_points) - 0.5)
        )
        
        # Gaussian envelope parameters (unconstrained)
        # Shape: (n_components,) for each
        self.envelope_mean_unconstrained = nn.Parameter(
            2 * (torch.rand(n_components) - 0.5)
        )
        self.envelope_std_unconstrained = nn.Parameter(
            2 * (torch.rand(n_components) - 0.5)
        )
        
        # Control point x-positions are fixed and evenly spaced
        self.control_x = np.linspace(0, axis_size - 1, self.n_control_points)
        self.axis_size = axis_size
    
    @property
    def params(self):
        """Expose unconstrained parameters."""
        return {
            'control_values_unconstrained': self.control_values_unconstrained,
            'envelope_mean_unconstrained': self.envelope_mean_unconstrained,
            'envelope_std_unconstrained': self.envelope_std_unconstrained
        }
    
    def set_params(self, control_values_unconstrained=None, envelope_mean_unconstrained=None, 
                   envelope_std_unconstrained=None):
        """Set parameters directly."""
        if control_values_unconstrained is not None:
            self.control_values_unconstrained.data = control_values_unconstrained
        if envelope_mean_unconstrained is not None:
            self.envelope_mean_unconstrained.data = envelope_mean_unconstrained
        if envelope_std_unconstrained is not None:
            self.envelope_std_unconstrained.data = envelope_std_unconstrained
    
    def forward(self):
        """
        Generate matrix where each column/row is spline(x) × Gaussian_envelope(x).
        
        Returns:
            torch.Tensor: Matrix of shape (n_rows, n_cols)
        """
        # Transform spline control values using sigmoid
        value_min, value_max = self.value_bounds
        control_values = value_min + (value_max - value_min) * torch.sigmoid(
            self.control_values_unconstrained
        )
        
        # Transform envelope parameters using sigmoid
        mean_min, mean_max = self.mean_bounds_custom
        envelope_means = mean_min + (mean_max - mean_min) * torch.sigmoid(
            self.envelope_mean_unconstrained
        )
        
        std_min, std_max = self.std_bounds_custom
        envelope_stds = std_min + (std_max - std_min) * torch.sigmoid(
            self.envelope_std_unconstrained
        ) + 1e-6
        
        if self.axis == 1:  # Column-wise
            n_components = self.n_cols
            output_size = self.n_rows
        else:  # Row-wise
            n_components = self.n_rows
            output_size = self.n_cols
        
        device = control_values.device
        
        # Compute base spline
        x_ctrl = torch.from_numpy(self.control_x).float().to(device)
        x_eval = torch.arange(output_size, dtype=torch.float32, device=device)
        
        y_ctrl_batch = control_values.unsqueeze(-1)  # Shape: (n_components, n_control_points, 1)
        coeffs = natural_cubic_spline_coeffs(x_ctrl, y_ctrl_batch)
        spline = NaturalCubicSpline(coeffs)
        spline_values = spline.evaluate(x_eval).squeeze(-1)  # Shape: (n_components, output_size)
        
        # Compute Gaussian envelope
        # x_eval shape: (output_size,)
        # envelope_means shape: (n_components,)
        # Need to broadcast to (n_components, output_size)
        x_grid = x_eval.unsqueeze(0)  # Shape: (1, output_size)
        means_grid = envelope_means.unsqueeze(1)  # Shape: (n_components, 1)
        stds_grid = envelope_stds.unsqueeze(1)  # Shape: (n_components, 1)
        
        envelope = torch.exp(-0.5 * ((x_grid - means_grid) / stds_grid) ** 2)
        # Shape: (n_components, output_size)
        
        # Multiply spline by envelope
        result = spline_values * envelope  # Shape: (n_components, output_size)
        
        # Arrange into output matrix based on axis
        if self.axis == 1:  # Column-wise: transpose to (output_size, n_components)
            matrix = result.T  # Shape: (n_rows, n_cols)
        else:  # Row-wise: already correct shape
            matrix = result  # Shape: (n_rows, n_cols)
        
        return matrix
    
    def __repr__(self):
        """String representation showing constrained parameter values."""
        value_min, value_max = self.value_bounds
        control_values = value_min + (value_max - value_min) * torch.sigmoid(
            self.control_values_unconstrained
        )
        
        mean_min, mean_max = self.mean_bounds_custom
        envelope_means = mean_min + (mean_max - mean_min) * torch.sigmoid(
            self.envelope_mean_unconstrained
        )
        
        std_min, std_max = self.std_bounds_custom
        envelope_stds = std_min + (std_max - std_min) * torch.sigmoid(
            self.envelope_std_unconstrained
        )
        
        return (f"{self.__class__.__name__}(shape={self.shape}, axis={self.axis}, "
                f"n_control_points={self.n_control_points})\n"
                f"  envelope_means: {envelope_means.detach().cpu().numpy()}\n"
                f"  envelope_stds: {envelope_stds.detach().cpu().numpy()}")
    
    def initialize_from_matrix(self, target_matrix):
        """Initialize parameters by finding peak and fitting spline."""
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
            
            # Find envelope parameters from peak
            peak_idx = np.argmax(data)
            envelope_mean_init = float(peak_idx)
            
            # Estimate std from weighted variance
            x = np.arange(axis_size)
            weights = np.maximum(data, 0)
            if weights.sum() > 0:
                weights = weights / weights.sum()
                weighted_mean = np.sum(x * weights)
                weighted_var = np.sum(weights * (x - weighted_mean) ** 2)
                envelope_std_init = float(np.sqrt(weighted_var) + 1e-3)
            else:
                envelope_std_init = axis_size / 5.0
            
            # Initialize control values from data (will be refined by envelope)
            control_values_init = np.interp(self.control_x, np.arange(axis_size), data)
            value_min, value_max = self.value_bounds
            control_values_init = np.clip(control_values_init, value_min + 1e-3, value_max - 1e-3)
            control_values_ratio = (control_values_init - value_min) / (value_max - value_min)
            control_values_ratio = np.clip(control_values_ratio, 1e-7, 1 - 1e-7)
            self.control_values_unconstrained.data[idx] = torch.from_numpy(
                np.log(control_values_ratio / (1 - control_values_ratio))
            ).float()
            
            # Apply inverse sigmoid for envelope parameters
            mean_min, mean_max = self.mean_bounds_custom
            mean_ratio = np.clip((envelope_mean_init - mean_min) / (mean_max - mean_min), 1e-7, 1 - 1e-7)
            self.envelope_mean_unconstrained.data[idx] = torch.tensor(
                np.log(mean_ratio / (1 - mean_ratio)), dtype=torch.float32
            )
            
            std_min, std_max = self.std_bounds_custom
            std_ratio = np.clip((envelope_std_init - std_min) / (std_max - std_min), 1e-7, 1 - 1e-7)
            self.envelope_std_unconstrained.data[idx] = torch.tensor(
                np.log(std_ratio / (1 - std_ratio)), dtype=torch.float32
            )
        
        super().initialize_from_matrix(target_matrix)

class UnimodalMonotonicSplineParameterization(MatrixParameterization):
    """
    Unimodal parameterization using gate-controlled slope blending.
    
    Each column or row is generated by:
    1. Three unconstrained parameter sets: up_slopes, down_slopes, gate
    2. Transform to positive: u = softplus(up), d = softplus(down)
    3. Build monotone decreasing gate: gate = 1 - cumsum(softplus(gate)) / sum(softplus(gate))
    4. Blend slopes: slope = gate * u - (1 - gate) * d
    5. Integrate slopes to get control points
    6. Fit smooth cubic spline through control points
    
    The gate transitions monotonically from 1 → 0, ensuring slopes change sign exactly once.
    The model learns where this transition happens, creating a single peak.
    
    Args:
        n_control_points: int - number of control points for the spline
        axis: int - 0 for row-wise, 1 for column-wise (default: 1)
    """
    def __init__(self, shape, n_control_points=8, axis=1, value_bounds=None):
        """
        Initialize simple unimodal spline parameterization.
        
        Args:
            shape: tuple (n_rows, n_cols) - shape of the matrix to generate
            n_control_points: int - number of control points for spline (default: 8)
            axis: int - 0 for row-wise, 1 for column-wise (default: 1)
            value_bounds: tuple (min, max) - bounds for output values
        """
        self.n_control_points = n_control_points
        self.axis = axis
        self.value_bounds = value_bounds
        super(UnimodalMonotonicSplineParameterization, self).__init__(shape)
    
    def _init_params(self):
        """Initialize parameters: up slopes, down slopes, and gate."""
        # Determine dimensions based on axis
        if self.axis == 1:  # Column-wise (W matrix)
            n_components = self.n_cols
            axis_size = self.n_rows
            default_value_max = 10.0
        else:  # axis == 0, Row-wise (H matrix)
            n_components = self.n_rows
            axis_size = self.n_cols
            default_value_max = 1.0
        
        # Set default bounds
        if self.value_bounds is None:
            self.value_bounds = (0, default_value_max)
        
        self.value_max = self.value_bounds[1]
        
        # We need n_control_points - 1 slopes (one per interval between control points)
        n_slopes = self.n_control_points - 1
        
        # Three parameter sets: up slopes, down slopes, gate
        self.up_unconstrained = nn.Parameter(
            torch.randn(n_components, n_slopes) * 0.5
        )
        
        self.down_unconstrained = nn.Parameter(
            torch.randn(n_components, n_slopes) * 0.5
        )
        
        self.gate_unconstrained = nn.Parameter(
            torch.randn(n_components, n_slopes) * 0.5
        )
        
        # Initial value parameter (starting height of each curve)
        self.initial_value_unconstrained = nn.Parameter(
            torch.randn(n_components) * 0.5
        )
        
        # Control point x-positions are fixed and evenly spaced
        self.control_x = np.linspace(0, axis_size - 1, self.n_control_points)
        self.axis_size = axis_size
    
    @property
    def params(self):
        """Expose unconstrained parameters."""
        return {
            'up_unconstrained': self.up_unconstrained,
            'down_unconstrained': self.down_unconstrained,
            'gate_unconstrained': self.gate_unconstrained,
            'initial_value_unconstrained': self.initial_value_unconstrained
        }
    
    def set_params(self, up_unconstrained=None, down_unconstrained=None, 
                   gate_unconstrained=None, initial_value_unconstrained=None):
        """Set parameters directly."""
        if up_unconstrained is not None:
            self.up_unconstrained.data = up_unconstrained
        if down_unconstrained is not None:
            self.down_unconstrained.data = down_unconstrained
        if gate_unconstrained is not None:
            self.gate_unconstrained.data = gate_unconstrained
        if initial_value_unconstrained is not None:
            self.initial_value_unconstrained.data = initial_value_unconstrained
    
    def forward(self):
        """
        Generate matrix where each column/row is a unimodal spline.
        
        Steps:
        1. Transform to positive slopes: u = softplus(up), d = softplus(down)
        2. Build monotone decreasing gate: gate = 1 - cumsum(softplus(gate)) / sum
        3. Blend slopes: slope = gate * u - (1 - gate) * d
        4. Integrate slopes to get control point values
        5. Fit smooth spline through control points
        
        Returns:
            torch.Tensor: Matrix of shape (n_rows, n_cols)
        """
        if self.axis == 1:  # Column-wise
            n_components = self.n_cols
            output_size = self.n_rows
        else:  # Row-wise
            n_components = self.n_rows
            output_size = self.n_cols
        
        device = self.up_unconstrained.device
        
        # Step 1: Transform to positive slopes using softplus
        u = F.softplus(self.up_unconstrained)  # (n_components, n_slopes)
        d = F.softplus(self.down_unconstrained)  # (n_components, n_slopes)
        
        # Step 2: Build monotone decreasing gate: 1 → 0
        gate_steps = F.softplus(self.gate_unconstrained)  # (n_components, n_slopes)
        gate_steps_sum = gate_steps.sum(dim=1, keepdim=True)  # (n_components, 1)
        gate = 1.0 - torch.cumsum(gate_steps, dim=1) / gate_steps_sum # (n_components, n_slopes)

        # Step 3: Blend slopes: positive before peak, negative after peak
        slopes = gate * u - (1 - gate) * d  # (n_components, n_slopes)
        
        # Step 4: Integrate slopes to get control point values
        # Initial value (starting height)
        value_min, value_max = self.value_bounds
        y0 = value_min + (value_max - value_min) * torch.sigmoid(self.initial_value_unconstrained)
        y0 = y0.unsqueeze(1)  # (n_components, 1)
        
        # Compute dx between control points
        x_ctrl = torch.from_numpy(self.control_x).float().to(device)
        dx = x_ctrl[1:] - x_ctrl[:-1]  # (n_slopes,)
        
        # Integrate: y[i+1] = y[i] + slope[i] * dx[i]
        y_values = [y0]  # Start with initial value
        for i in range(slopes.shape[1]):
            y_next = y_values[-1] + slopes[:, i:i+1] * dx[i]
            y_values.append(y_next)
        
        control_values = torch.cat(y_values, dim=1)  # (n_components, n_control_points)
        
        # Step 5: Fit smooth spline through control points
        x_eval = torch.arange(output_size, dtype=torch.float32, device=device)
        
        # Add channel dimension for torchcubicspline
        control_values_batch = control_values.unsqueeze(-1)  # (n_components, n_control_points, 1)
        
        # Compute spline coefficients and evaluate
        coeffs = natural_cubic_spline_coeffs(x_ctrl, control_values_batch)
        spline = NaturalCubicSpline(coeffs)
        spline_values = spline.evaluate(x_eval).squeeze(-1)  # (n_components, output_size)
        
        # Clamp to valid range
        result = torch.clamp(spline_values, value_min, value_max)
        
        # Arrange into output matrix based on axis
        if self.axis == 1:  # Column-wise: transpose
            matrix = result.T  # Shape: (n_rows, n_cols)
        else:  # Row-wise: already correct shape
            matrix = result  # Shape: (n_rows, n_cols)
        
        return matrix
    
    def __repr__(self):
        """String representation."""
        return (f"{self.__class__.__name__}(shape={self.shape}, axis={self.axis}, "
                f"n_control_points={self.n_control_points})")
    
    def initialize_from_matrix(self, target_matrix):
        """Initialize parameters by analyzing the target matrix."""
        target_matrix_np = target_matrix.detach().cpu().numpy()
        
        if self.axis == 1:  # Column-wise
            n_iters = self.n_cols
            axis_size = self.n_rows
        else:  # Row-wise
            n_iters = self.n_rows
            axis_size = self.n_cols
        
        n_slopes = self.n_control_points - 1
        
        for idx in range(n_iters):
            # Extract data
            if self.axis == 1:
                data = target_matrix_np[:, idx]
            else:
                data = target_matrix_np[idx, :]
            
            # Find peak location
            peak_idx = np.argmax(data)
            
            # Sample control points and compute target slopes
            control_indices = np.linspace(0, axis_size - 1, self.n_control_points, dtype=int)
            target_values = data[control_indices]
            
            # Initial value
            y0_init = target_values[0]
            value_min, value_max = self.value_bounds
            y0_ratio = np.clip((y0_init - value_min) / (value_max - value_min), 1e-7, 1 - 1e-7)
            inv_sigmoid = lambda r: np.log(r / (1 - r))
            self.initial_value_unconstrained.data[idx] = torch.tensor(inv_sigmoid(y0_ratio), dtype=torch.float32)
            
            # Compute target slopes from consecutive values
            dx = np.diff(self.control_x)
            target_slopes = np.diff(target_values) / dx
            
            # Find where peak occurs in control points
            peak_ctrl_idx = np.argmin(np.abs(control_indices - peak_idx))
            
            # Initialize gate: 1 before peak, 0 after peak
            # Use smooth transition around peak
            gate_target = np.zeros(n_slopes)
            for i in range(n_slopes):
                if i < peak_ctrl_idx:
                    gate_target[i] = 1.0
                elif i == peak_ctrl_idx:
                    gate_target[i] = 0.5  # Transition at peak
                else:
                    gate_target[i] = 0.0
            
            # Initialize up and down slopes
            # For positive target slopes: mostly from up (gate≈1)
            # For negative target slopes: mostly from down (gate≈0)
            up_init = np.abs(target_slopes) + 0.1
            down_init = np.abs(target_slopes) + 0.1
            
            # Inverse softplus: x = log(exp(y) - 1), but numerically stable
            inv_softplus = lambda y: np.log(np.exp(np.clip(y, 0, 10)) - 1 + 1e-7)
            
            self.up_unconstrained.data[idx] = torch.from_numpy(inv_softplus(up_init)).float()
            self.down_unconstrained.data[idx] = torch.from_numpy(inv_softplus(down_init)).float()
            
            # Initialize gate_steps to produce desired gate
            # gate = 1 - cumsum(gate_steps) / sum(gate_steps)
            # We want gate to transition from 1 to 0 around peak
            gate_steps_init = np.ones(n_slopes) / n_slopes
            # Make transition sharper around peak
            for i in range(n_slopes):
                if i < peak_ctrl_idx:
                    gate_steps_init[i] = 0.1
                elif i == peak_ctrl_idx:
                    gate_steps_init[i] = 5.0  # Sharp transition
                else:
                    gate_steps_init[i] = 0.1
            
            self.gate_unconstrained.data[idx] = torch.from_numpy(inv_softplus(gate_steps_init)).float()
        
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
        self.mean_bounds = mean_bounds
        self.std_bounds = std_bounds
        self.scale_bounds = scale_bounds
        super(GeneralizedGaussianParameterization, self).__init__(shape)
    
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
        self.mean_bounds = mean_bounds
        self.std_bounds = std_bounds
        self.scale_bounds = scale_bounds
        super(MixtureOfGeneralizedGaussiansParameterization, self).__init__(shape)
    
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

class PartiallyFixedMixtureOfGeneralizedGaussiansParameterization(MixtureOfGeneralizedGaussiansParameterization):
    """
    Mixture of Generalized Gaussians with arbitrary columns fixed from a reference matrix.
    Only supports axis=1 (column-wise).
    
    Args:
        fix_components: List[int] - indices of columns to fix
        reference_matrix: torch.Tensor - reference matrix to extract fixed column parameters from
    """
    def __init__(self, shape, n_gaussians=2, mean_bounds=None, std_bounds=None, 
                 beta_bounds=None, scale_bounds=None, fix_components=None, reference_matrix=None):
        """
        Initialize with some columns fixed.
        
        Args:
            shape: tuple (n_rows, n_cols) - shape of the matrix to generate
            n_gaussians: int - number of Generalized Gaussians in the mixture
            mean_bounds, std_bounds, beta_bounds, scale_bounds: parameter bounds
            fix_components: List[int] - indices of columns to fix
            reference_matrix: torch.Tensor - matrix to extract fixed parameters from
        """
        if reference_matrix is None:
            raise ValueError("reference_matrix must be provided")
        if fix_components is None:
            fix_components = []
        self.fix_components = list(sorted(set(fix_components)))
        self.reference_matrix = reference_matrix
        # Initialize parent (axis is always 1 for this class)
        super().__init__(shape, n_gaussians=n_gaussians, axis=1, 
                        mean_bounds=mean_bounds, std_bounds=std_bounds,
                        beta_bounds=beta_bounds, scale_bounds=scale_bounds)
    
    def _init_params(self):
        """Initialize parameters with some columns fixed."""
        # Create a temporary parent class instance to fit the reference matrix
        temp_param = MixtureOfGeneralizedGaussiansParameterization(
            shape=self.shape,
            n_gaussians=self.n_gaussians,
            axis=1,
            mean_bounds=self.mean_bounds,
            std_bounds=self.std_bounds,
            beta_bounds=self.beta_bounds,
            scale_bounds=self.scale_bounds
        )
        temp_param.initialize_from_matrix(self.reference_matrix)
        n_components = self.n_cols
        means_data = temp_param.means_unconstrained.data.clone()
        stds_data = temp_param.stds_unconstrained.data.clone()
        beta_data = temp_param.beta_unconstrained.data.clone()
        scales_data = temp_param.scales_unconstrained.data.clone()
        # Store bounds (they were set by temp_param)
        self.mean_bounds = temp_param.mean_bounds
        self.std_bounds = temp_param.std_bounds
        self.beta_bounds = temp_param.beta_bounds
        self.scale_bounds = temp_param.scale_bounds
        # Determine which columns to fix
        fixed_indices = self.fix_components
        learnable_indices = [i for i in range(n_components) if i not in fixed_indices]
        self.fixed_indices = fixed_indices
        self.learnable_indices = learnable_indices
        # Register fixed columns as buffers (non-trainable)
        if len(fixed_indices) > 0:
            self.register_buffer('means_fixed', means_data[fixed_indices])
            self.register_buffer('stds_fixed', stds_data[fixed_indices])
            self.register_buffer('beta_fixed', beta_data[fixed_indices])
            self.register_buffer('scales_fixed', scales_data[fixed_indices])
        else:
            self.register_buffer('means_fixed', torch.empty((0, self.n_gaussians)))
            self.register_buffer('stds_fixed', torch.empty((0, self.n_gaussians)))
            self.register_buffer('beta_fixed', torch.empty((0, self.n_gaussians)))
            self.register_buffer('scales_fixed', torch.empty((0, self.n_gaussians)))
        # Create parameters for learnable columns (requires_grad=True)
        if len(learnable_indices) > 0:
            self.means_unconstrained = nn.Parameter(2 * (torch.rand(len(learnable_indices), self.n_gaussians) - 0.5))
            self.stds_unconstrained = nn.Parameter(2 * (torch.rand(len(learnable_indices), self.n_gaussians) - 0.5))
            self.beta_unconstrained = nn.Parameter(2 * (torch.rand(len(learnable_indices), self.n_gaussians) - 0.5))
            self.scales_unconstrained = nn.Parameter(2 * (torch.rand(len(learnable_indices), self.n_gaussians) - 0.5))
        else:
            self.means_unconstrained = None
            self.stds_unconstrained = None
            self.beta_unconstrained = None
            self.scales_unconstrained = None
    
    def forward(self):
        """Generate matrix by combining fixed and learnable columns, then calling parent's computation."""
        n_components = self.n_cols
        device = self.reference_matrix.device if hasattr(self.reference_matrix, 'device') else torch.device('cpu')
        full_means = torch.zeros(n_components, self.n_gaussians, device=device)
        full_stds = torch.zeros(n_components, self.n_gaussians, device=device)
        full_beta = torch.zeros(n_components, self.n_gaussians, device=device)
        full_scales = torch.zeros(n_components, self.n_gaussians, device=device)
        # Fill fixed columns from buffers
        for i, idx in enumerate(self.fixed_indices):
            full_means[idx] = self.means_fixed[i]
            full_stds[idx] = self.stds_fixed[i]
            full_beta[idx] = self.beta_fixed[i]
            full_scales[idx] = self.scales_fixed[i]
        # Fill learnable columns from parameters
        if self.means_unconstrained is not None:
            for i, idx in enumerate(self.learnable_indices):
                full_means[idx] = self.means_unconstrained[i]
                full_stds[idx] = self.stds_unconstrained[i]
                full_beta[idx] = self.beta_unconstrained[i]
                full_scales[idx] = self.scales_unconstrained[i]
        # Temporarily assign full tensors and call parent's forward logic
        mean_min, mean_max = self.mean_bounds
        means = mean_min + (mean_max - mean_min) * torch.sigmoid(full_means)
        std_min, std_max = self.std_bounds
        stds = std_min + (std_max - std_min) * torch.sigmoid(full_stds) + 1e-6
        beta_min, beta_max = self.beta_bounds
        beta = beta_min + (beta_max - beta_min) * torch.sigmoid(full_beta)
        scale_min, scale_max = self.scale_bounds
        scales = scale_min + (scale_max - scale_min) * torch.sigmoid(full_scales)
        # Column-wise computation (axis=1)
        x = torch.arange(self.n_rows, dtype=torch.float32, device=device)
        x = x.unsqueeze(1).unsqueeze(1)  # Shape: (n_rows, 1, 1)
        means = means.unsqueeze(0)  # Shape: (1, n_cols, n_gaussians)
        stds = stds.unsqueeze(0)
        beta = beta.unsqueeze(0)
        scales = scales.unsqueeze(0)
        abs_diff = torch.abs(x - means)
        gaussians = scales * torch.exp(-0.5 * ((abs_diff / stds) ** beta))
        matrix = torch.sum(gaussians, dim=2)
        return matrix
    
    def initialize_from_matrix(self, target_matrix):
        """Initialize only the learnable columns from target matrix."""
        if self.means_unconstrained is None:
            # All columns are fixed, nothing to initialize
            return
        from scipy.signal import find_peaks
        target_matrix_np = target_matrix.detach().cpu().numpy()
        axis_size = self.n_rows
        for i, col_idx in enumerate(self.learnable_indices):
            data = target_matrix_np[:, col_idx]
            peaks, _ = find_peaks(data)
            if len(peaks) < self.n_gaussians:
                additionale_data = np.copy(data)
                additionale_data[peaks] = -np.inf
                additional_peaks = np.argsort(additionale_data)[-(self.n_gaussians - len(peaks)):]
                peaks = np.concatenate([peaks, additional_peaks])
            else:
                peaks = peaks[np.argsort(data[peaks])][-self.n_gaussians:]
            means_init = peaks
            std_value = axis_size / 10.0
            stds_init = np.full(self.n_gaussians, std_value)
            scales_init = data[peaks]
            beta_init = np.full(self.n_gaussians, 2.0)
            means_ratio = np.clip((means_init - self.mean_bounds[0]) / (self.mean_bounds[1] - self.mean_bounds[0]), 1e-7, 1 - 1e-7)
            self.means_unconstrained.data[i] = torch.from_numpy(np.log(means_ratio / (1 - means_ratio))).float()
            stds_ratio = np.clip((stds_init - self.std_bounds[0]) / (self.std_bounds[1] - self.std_bounds[0]), 1e-7, 1 - 1e-7)
            self.stds_unconstrained.data[i] = torch.from_numpy(np.log(stds_ratio / (1 - stds_ratio))).float()
            beta_ratio = np.clip((beta_init - self.beta_bounds[0]) / (self.beta_bounds[1] - self.beta_bounds[0]), 1e-7, 1 - 1e-7)
            self.beta_unconstrained.data[i] = torch.from_numpy(np.log(beta_ratio / (1 - beta_ratio))).float()
            scales_ratio = np.clip((scales_init - self.scale_bounds[0]) / (self.scale_bounds[1] - self.scale_bounds[0]), 1e-7, 1 - 1e-7)
            self.scales_unconstrained.data[i] = torch.from_numpy(np.log(scales_ratio / (1 - scales_ratio))).float()
        MatrixParameterization.initialize_from_matrix(self, target_matrix)

class SkewNormalParameterization(MatrixParameterization):
    """
    Each column or row is a Skew-Normal distribution.
    Skew-Normal allows asymmetric peaks, useful for modeling skewed data.
    
    Parameters: location (mean), scale (std), skewness (alpha), amplitude.
    Uses sigmoid transformation to enforce bounds.
    
    Args:
        axis: int - 0 for row-wise, 1 for column-wise (default: 1)
    """
    def __init__(self, shape, axis=1, mean_bounds=None, std_bounds=None, 
                 skewness_bounds=None, scale_bounds=None):
        """
        Initialize Skew-Normal parameterization.
        
        Args:
            shape: tuple (n_rows, n_cols) - shape of the matrix to generate
            axis: int - 0 for row-wise, 1 for column-wise (default: 1)
            location_bounds: tuple (min, max) - bounds for location parameter
            scale_bounds: tuple (min, max) - bounds for scale parameter
            skewness_bounds: tuple (min, max) - bounds for skewness parameter (alpha)
            amplitude_bounds: tuple (min, max) - bounds for amplitude
        """
        self.axis = axis
        self.skewness_bounds = skewness_bounds
        # Reuse mean_bounds for location, std_bounds for scale, scale_bounds for amplitude
        self.mean_bounds = mean_bounds
        self.std_bounds = std_bounds
        self.scale_bounds = scale_bounds
        super(SkewNormalParameterization, self).__init__(shape)

    def _mean_bounds_are_per_component(self):
        """Return True when mean_bounds is provided as one (min, max) pair per component."""
        if self.mean_bounds is None:
            return False
        if isinstance(self.mean_bounds, torch.Tensor):
            return self.mean_bounds.ndim == 2 and self.mean_bounds.shape[-1] == 2
        if isinstance(self.mean_bounds, np.ndarray):
            return self.mean_bounds.ndim == 2 and self.mean_bounds.shape[-1] == 2
        return (
            isinstance(self.mean_bounds, (list, tuple))
            and len(self.mean_bounds) > 0
            and isinstance(self.mean_bounds[0], (list, tuple, np.ndarray, torch.Tensor))
        )

    def _get_mean_bounds(self, device=None):
        """Return mean bounds as either scalars or per-component tensors."""
        if not self._mean_bounds_are_per_component():
            return self.mean_bounds

        bounds = torch.as_tensor(self.mean_bounds, dtype=torch.float32, device=device)
        expected_components = self.n_cols if self.axis == 1 else self.n_rows
        if bounds.shape != (expected_components, 2):
            raise ValueError(
                f"mean_bounds must have shape ({expected_components}, 2) when provided per component; got {tuple(bounds.shape)}"
            )
        return bounds[:, 0], bounds[:, 1]
    
    def _init_params(self):
        """Initialize unconstrained Skew-Normal parameters."""
        if self.axis == 1:  # Column-wise
            n_distributions = self.n_cols
            axis_size = self.n_rows
            default_amplitude_max = 10.0
        else:  # Row-wise
            n_distributions = self.n_rows
            axis_size = self.n_cols
            default_amplitude_max = 1.0
        
        # Set default bounds
        if self.mean_bounds is None:  # location_bounds
            self.mean_bounds = (0, axis_size)
        if self.std_bounds is None:  # scale_bounds
            self.std_bounds = (0.5, axis_size / 3)
        if self.skewness_bounds is None:
            self.skewness_bounds = (-5.0, 5.0)
        if self.scale_bounds is None:  # amplitude_bounds
            self.scale_bounds = (0, default_amplitude_max)
        
        self.location_unconstrained = nn.Parameter(2 * (torch.rand(n_distributions) - 0.5))
        self.scale_unconstrained = nn.Parameter(2 * (torch.rand(n_distributions) - 0.5))
        self.skewness_unconstrained = nn.Parameter(2 * (torch.rand(n_distributions) - 0.5))
        self.amplitude_unconstrained = nn.Parameter(2 * (torch.rand(n_distributions) - 0.5))
    
    @property
    def params(self):
        """Expose unconstrained parameters."""
        return {
            'location_unconstrained': self.location_unconstrained,
            'scale_unconstrained': self.scale_unconstrained,
            'skewness_unconstrained': self.skewness_unconstrained,
            'amplitude_unconstrained': self.amplitude_unconstrained
        }
    
    def set_params(self, location_unconstrained=None, scale_unconstrained=None, 
                   skewness_unconstrained=None, amplitude_unconstrained=None):
        """Set parameters directly."""
        if location_unconstrained is not None:
            self.location_unconstrained.data = location_unconstrained
        if scale_unconstrained is not None:
            self.scale_unconstrained.data = scale_unconstrained
        if skewness_unconstrained is not None:
            self.skewness_unconstrained.data = skewness_unconstrained
        if amplitude_unconstrained is not None:
            self.amplitude_unconstrained.data = amplitude_unconstrained
    
    def forward(self):
        """Generate matrix where each column/row is a Skew-Normal distribution."""
        # Transform parameters
        location_bounds = self._get_mean_bounds(device=self.location_unconstrained.device)
        if self._mean_bounds_are_per_component():
            location_min, location_max = location_bounds
        else:
            location_min, location_max = location_bounds
        location = location_min + (location_max - location_min) * torch.sigmoid(self.location_unconstrained)
        
        scale_min, scale_max = self.std_bounds
        scale = scale_min + (scale_max - scale_min) * torch.sigmoid(self.scale_unconstrained) + 1e-6
        
        skewness_min, skewness_max = self.skewness_bounds
        skewness = skewness_min + (skewness_max - skewness_min) * torch.sigmoid(self.skewness_unconstrained)
        
        amplitude_min, amplitude_max = self.scale_bounds
        amplitude = amplitude_min + (amplitude_max - amplitude_min) * torch.sigmoid(self.amplitude_unconstrained)
        
        if self.axis == 1:  # Column-wise
            x = torch.arange(self.n_rows, dtype=torch.float32, device=location.device)
            x = x.unsqueeze(1)
            location = location.unsqueeze(0)
            scale = scale.unsqueeze(0)
            skewness = skewness.unsqueeze(0)
            amplitude = amplitude.unsqueeze(0)
        else:  # Row-wise
            x = torch.arange(self.n_cols, dtype=torch.float32, device=location.device)
            x = x.unsqueeze(0)
            location = location.unsqueeze(1)
            scale = scale.unsqueeze(1)
            skewness = skewness.unsqueeze(1)
            amplitude = amplitude.unsqueeze(1)
        
        # Skew-Normal PDF: 2 * phi((x-loc)/scale) * Phi(alpha * (x-loc)/scale)
        # where phi is the standard normal PDF and Phi is the standard normal CDF
        z = (x - location) / scale
        
        # Standard normal PDF: phi(z) = exp(-0.5 * z^2) / sqrt(2*pi)
        phi_z = torch.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
        
        # Standard normal CDF: Phi(alpha*z) using erf
        # Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
        Phi_alpha_z = 0.5 * (1 + torch.erf(skewness * z / np.sqrt(2)))
        
        # Skew-Normal PDF (unnormalized by amplitude)
        matrix = amplitude * 2 * phi_z * Phi_alpha_z / scale

        return matrix
    
    def __repr__(self):
        """String representation showing constrained parameter values."""
        location_bounds = self._get_mean_bounds(device=self.location_unconstrained.device)
        if self._mean_bounds_are_per_component():
            location_min, location_max = location_bounds
        else:
            location_min, location_max = location_bounds
        location = location_min + (location_max - location_min) * torch.sigmoid(self.location_unconstrained)
        scale_min, scale_max = self.std_bounds
        scale = scale_min + (scale_max - scale_min) * torch.sigmoid(self.scale_unconstrained)
        skewness_min, skewness_max = self.skewness_bounds
        skewness = skewness_min + (skewness_max - skewness_min) * torch.sigmoid(self.skewness_unconstrained)
        amplitude_min, amplitude_max = self.scale_bounds
        amplitude = amplitude_min + (amplitude_max - amplitude_min) * torch.sigmoid(self.amplitude_unconstrained)
        
        return (f"{self.__class__.__name__}(shape={self.shape}, axis={self.axis})\n"
                f"  location: {location.detach().cpu().numpy()}\n"
                f"  std: {scale.detach().cpu().numpy()}\n"
                f"  skewness: {skewness.detach().cpu().numpy()}\n"
                f"  scale: {amplitude.detach().cpu().numpy()}")

    def initialize_from_matrix(self, target_matrix):
        """Initialize parameters by finding characteristics in each column/row."""
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
            
            # Find location as position of maximum value
            location_init = float(np.argmax(data))
            
            # Amplitude is the maximum value
            amplitude_init = float(np.max(data))
            
            # Estimate scale from weighted variance
            x = np.arange(axis_size)
            weights = np.maximum(data, 0)
            if weights.sum() > 0:
                weights = weights / weights.sum()
                weighted_mean = np.sum(x * weights)
                weighted_var = np.sum(weights * (x - weighted_mean) ** 2)
                scale_init = float(np.sqrt(weighted_var) + 1e-3)
            else:
                scale_init = axis_size / 10.0
            
            # Estimate skewness from the asymmetry around the peak
            # Simple heuristic: compare left and right halves
            peak_idx = int(location_init)
            if peak_idx > 0 and peak_idx < len(data) - 1:
                left_mass = np.sum(data[:peak_idx])
                right_mass = np.sum(data[peak_idx+1:])
                if left_mass + right_mass > 0:
                    skewness_init = 2.0 * (right_mass - left_mass) / (left_mass + right_mass)
                else:
                    skewness_init = 0.0
            else:
                skewness_init = 0.0
            
            # Apply inverse sigmoid transformation
            if self._mean_bounds_are_per_component():
                location_min, location_max = self.mean_bounds[idx]
            else:
                location_min, location_max = self.mean_bounds
            location_ratio = np.clip((location_init - location_min) / (location_max - location_min), 1e-7, 1 - 1e-7)
            self.location_unconstrained.data[idx] = torch.tensor(np.log(location_ratio / (1 - location_ratio)), dtype=torch.float32)
            
            scale_ratio = np.clip((scale_init - self.std_bounds[0]) / (self.std_bounds[1] - self.std_bounds[0]), 1e-7, 1 - 1e-7)
            self.scale_unconstrained.data[idx] = torch.tensor(np.log(scale_ratio / (1 - scale_ratio)), dtype=torch.float32)
            
            skewness_ratio = np.clip((skewness_init - self.skewness_bounds[0]) / (self.skewness_bounds[1] - self.skewness_bounds[0]), 1e-7, 1 - 1e-7)
            self.skewness_unconstrained.data[idx] = torch.tensor(np.log(skewness_ratio / (1 - skewness_ratio)), dtype=torch.float32)
            
            amplitude_ratio = np.clip((amplitude_init - self.scale_bounds[0]) / (self.scale_bounds[1] - self.scale_bounds[0]), 1e-7, 1 - 1e-7)
            self.amplitude_unconstrained.data[idx] = torch.tensor(np.log(amplitude_ratio / (1 - amplitude_ratio)), dtype=torch.float32)
        
        super().initialize_from_matrix(target_matrix)

class SkewTParameterization(MatrixParameterization):
    """
    Each column or row is a Skew-t distribution.
    Skew-t is more robust to outliers than Skew-Normal, with heavier tails.
    
    Parameters: location, scale, skewness (alpha), degrees of freedom (df), amplitude.
    Uses sigmoid transformation to enforce bounds.
    
    Args:
        axis: int - 0 for row-wise, 1 for column-wise (default: 1)
    """
    def __init__(self, shape, axis=1, mean_bounds=None, std_bounds=None, 
                 skewness_bounds=None, df_bounds=None, scale_bounds=None):
        """
        Initialize Skew-t parameterization.
        
        Args:
            shape: tuple (n_rows, n_cols) - shape of the matrix to generate
            axis: int - 0 for row-wise, 1 for column-wise (default: 1)
            location_bounds: tuple (min, max) - bounds for location parameter
            scale_bounds: tuple (min, max) - bounds for scale parameter
            skewness_bounds: tuple (min, max) - bounds for skewness parameter
            df_bounds: tuple (min, max) - bounds for degrees of freedom
            amplitude_bounds: tuple (min, max) - bounds for amplitude
        """
        self.axis = axis
        self.skewness_bounds = skewness_bounds
        self.df_bounds = df_bounds
        self.mean_bounds = mean_bounds
        self.std_bounds = std_bounds
        self.scale_bounds = scale_bounds
        super(SkewTParameterization, self).__init__(shape)
    
    def _init_params(self):
        """Initialize unconstrained Skew-t parameters."""
        if self.axis == 1:  # Column-wise
            n_distributions = self.n_cols
            axis_size = self.n_rows
            default_amplitude_max = 10.0
        else:  # Row-wise
            n_distributions = self.n_rows
            axis_size = self.n_cols
            default_amplitude_max = 1.0
        
        # Set default bounds
        if self.mean_bounds is None:
            self.mean_bounds = (0, axis_size)
        if self.std_bounds is None:
            self.std_bounds = (0.5, axis_size / 3)
        if self.skewness_bounds is None:
            self.skewness_bounds = (-5.0, 5.0)
        if self.df_bounds is None:
            self.df_bounds = (2.0, 30.0)
        if self.scale_bounds is None:
            self.scale_bounds = (0, default_amplitude_max)
        
        self.location_unconstrained = nn.Parameter(2 * (torch.rand(n_distributions) - 0.5))
        self.scale_unconstrained = nn.Parameter(2 * (torch.rand(n_distributions) - 0.5))
        self.skewness_unconstrained = nn.Parameter(2 * (torch.rand(n_distributions) - 0.5))
        self.df_unconstrained = nn.Parameter(2 * (torch.rand(n_distributions) - 0.5))
        self.amplitude_unconstrained = nn.Parameter(2 * (torch.rand(n_distributions) - 0.5))
    
    @property
    def params(self):
        """Expose unconstrained parameters."""
        return {
            'location_unconstrained': self.location_unconstrained,
            'scale_unconstrained': self.scale_unconstrained,
            'skewness_unconstrained': self.skewness_unconstrained,
            'df_unconstrained': self.df_unconstrained,
            'amplitude_unconstrained': self.amplitude_unconstrained
        }
    
    def set_params(self, location_unconstrained=None, scale_unconstrained=None, 
                   skewness_unconstrained=None, df_unconstrained=None, amplitude_unconstrained=None):
        """Set parameters directly."""
        if location_unconstrained is not None:
            self.location_unconstrained.data = location_unconstrained
        if scale_unconstrained is not None:
            self.scale_unconstrained.data = scale_unconstrained
        if skewness_unconstrained is not None:
            self.skewness_unconstrained.data = skewness_unconstrained
        if df_unconstrained is not None:
            self.df_unconstrained.data = df_unconstrained
        if amplitude_unconstrained is not None:
            self.amplitude_unconstrained.data = amplitude_unconstrained
    
    def forward(self):
        """Generate matrix where each column/row is a Skew-t distribution."""
        # Transform parameters
        location_min, location_max = self.mean_bounds
        location = location_min + (location_max - location_min) * torch.sigmoid(self.location_unconstrained)
        
        scale_min, scale_max = self.std_bounds
        scale = scale_min + (scale_max - scale_min) * torch.sigmoid(self.scale_unconstrained) + 1e-6
        
        skewness_min, skewness_max = self.skewness_bounds
        skewness = skewness_min + (skewness_max - skewness_min) * torch.sigmoid(self.skewness_unconstrained)
        
        df_min, df_max = self.df_bounds
        df = df_min + (df_max - df_min) * torch.sigmoid(self.df_unconstrained)
        
        amplitude_min, amplitude_max = self.scale_bounds
        amplitude = amplitude_min + (amplitude_max - amplitude_min) * torch.sigmoid(self.amplitude_unconstrained)
        
        if self.axis == 1:  # Column-wise
            x = torch.arange(self.n_rows, dtype=torch.float32, device=location.device)
            x = x.unsqueeze(1)
            location = location.unsqueeze(0)
            scale = scale.unsqueeze(0)
            skewness = skewness.unsqueeze(0)
            df = df.unsqueeze(0)
            amplitude = amplitude.unsqueeze(0)
        else:  # Row-wise
            x = torch.arange(self.n_cols, dtype=torch.float32, device=location.device)
            x = x.unsqueeze(0)
            location = location.unsqueeze(1)
            scale = scale.unsqueeze(1)
            skewness = skewness.unsqueeze(1)
            df = df.unsqueeze(1)
            amplitude = amplitude.unsqueeze(1)
        
        # Standardized variable
        z = (x - location) / scale
        
        # Student's t PDF: proportional to (1 + z^2/df)^(-(df+1)/2)
        t_pdf = torch.pow(1 + z**2 / df, -(df + 1) / 2)
        
        # Student's t CDF approximation using erf (for skewness term)
        # Use the relationship between t and normal for large df
        # For simplicity, approximate t CDF with normal CDF scaled
        t_cdf_approx = 0.5 * (1 + torch.erf(skewness * z / np.sqrt(2)))
        
        # Skew-t PDF (simplified version)
        matrix = amplitude * 2 * t_pdf * t_cdf_approx / scale
        
        return matrix
    
    def __repr__(self):
        """String representation showing constrained parameter values."""
        location_min, location_max = self.mean_bounds
        location = location_min + (location_max - location_min) * torch.sigmoid(self.location_unconstrained)
        scale_min, scale_max = self.std_bounds
        scale = scale_min + (scale_max - scale_min) * torch.sigmoid(self.scale_unconstrained)
        skewness_min, skewness_max = self.skewness_bounds
        skewness = skewness_min + (skewness_max - skewness_min) * torch.sigmoid(self.skewness_unconstrained)
        df_min, df_max = self.df_bounds
        df = df_min + (df_max - df_min) * torch.sigmoid(self.df_unconstrained)
        amplitude_min, amplitude_max = self.scale_bounds
        amplitude = amplitude_min + (amplitude_max - amplitude_min) * torch.sigmoid(self.amplitude_unconstrained)
        
        return (f"{self.__class__.__name__}(shape={self.shape}, axis={self.axis})\n"
                f"  location: {location.detach().cpu().numpy()}\n"
                f"  std: {scale.detach().cpu().numpy()}\n"
                f"  skewness: {skewness.detach().cpu().numpy()}\n"
                f"  df: {df.detach().cpu().numpy()}\n"
                f"  scale: {amplitude.detach().cpu().numpy()}")

    def initialize_from_matrix(self, target_matrix):
        """Initialize parameters by finding characteristics in each column/row."""
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
            
            # Find location as position of maximum value
            location_init = float(np.argmax(data))
            amplitude_init = float(np.max(data))
            
            # Estimate scale
            x = np.arange(axis_size)
            weights = np.maximum(data, 0)
            if weights.sum() > 0:
                weights = weights / weights.sum()
                weighted_mean = np.sum(x * weights)
                weighted_var = np.sum(weights * (x - weighted_mean) ** 2)
                scale_init = float(np.sqrt(weighted_var) + 1e-3)
            else:
                scale_init = axis_size / 10.0
            
            # Estimate skewness from asymmetry
            peak_idx = int(location_init)
            if peak_idx > 0 and peak_idx < len(data) - 1:
                left_mass = np.sum(data[:peak_idx])
                right_mass = np.sum(data[peak_idx+1:])
                if left_mass + right_mass > 0:
                    skewness_init = 2.0 * (right_mass - left_mass) / (left_mass + right_mass)
                else:
                    skewness_init = 0.0
            else:
                skewness_init = 0.0
            
            # Initialize df to middle of range
            df_init = (self.df_bounds[0] + self.df_bounds[1]) / 2
            
            # Apply inverse sigmoid transformation
            location_ratio = np.clip((location_init - self.mean_bounds[0]) / (self.mean_bounds[1] - self.mean_bounds[0]), 1e-7, 1 - 1e-7)
            self.location_unconstrained.data[idx] = torch.tensor(np.log(location_ratio / (1 - location_ratio)), dtype=torch.float32)
            
            scale_ratio = np.clip((scale_init - self.std_bounds[0]) / (self.std_bounds[1] - self.std_bounds[0]), 1e-7, 1 - 1e-7)
            self.scale_unconstrained.data[idx] = torch.tensor(np.log(scale_ratio / (1 - scale_ratio)), dtype=torch.float32)
            
            skewness_ratio = np.clip((skewness_init - self.skewness_bounds[0]) / (self.skewness_bounds[1] - self.skewness_bounds[0]), 1e-7, 1 - 1e-7)
            self.skewness_unconstrained.data[idx] = torch.tensor(np.log(skewness_ratio / (1 - skewness_ratio)), dtype=torch.float32)
            
            df_ratio = np.clip((df_init - self.df_bounds[0]) / (self.df_bounds[1] - self.df_bounds[0]), 1e-7, 1 - 1e-7)
            self.df_unconstrained.data[idx] = torch.tensor(np.log(df_ratio / (1 - df_ratio)), dtype=torch.float32)
            
            amplitude_ratio = np.clip((amplitude_init - self.scale_bounds[0]) / (self.scale_bounds[1] - self.scale_bounds[0]), 1e-7, 1 - 1e-7)
            self.amplitude_unconstrained.data[idx] = torch.tensor(np.log(amplitude_ratio / (1 - amplitude_ratio)), dtype=torch.float32)
        
        super().initialize_from_matrix(target_matrix)

# ========================================
# Loss Function Classes
# ========================================

class LossFunction(ABC):
    """
    Abstract base class for loss functions in NMF.
    All loss functions should inherit from this class and implement the forward method.
    """
    
    @abstractmethod
    def forward(self, W: torch.Tensor, H: torch.Tensor, A_observed: torch.Tensor, 
                iteration: int, total_iterations: int) -> torch.Tensor:
        """
        Compute the loss.
        
        Args:
            W: torch.Tensor - W matrix (n_wavelengths x k_components)
            H: torch.Tensor - H matrix (k_components x n_timepoints)
            A_observed: torch.Tensor - Observed matrix to factorize
            iteration: int - Current iteration number
            total_iterations: int - Total number of iterations
            
        Returns:
            torch.Tensor - Scalar loss value
        """
        pass
    
    def __call__(self, W: torch.Tensor, H: torch.Tensor, A_observed: torch.Tensor, 
                 iteration: int = 0, total_iterations: int = 1) -> torch.Tensor:
        """Make the loss function callable."""
        return self.forward(W, H, A_observed, iteration, total_iterations)


class FittingLoss(LossFunction):
    """
    Fitting loss: ||A - W @ H||^2
    Measures how well the factorization reconstructs the observed matrix.
    """
    
    def forward(self, W: torch.Tensor, H: torch.Tensor, A_observed: torch.Tensor, 
                iteration: int, total_iterations: int) -> torch.Tensor:
        """Compute MSE reconstruction loss."""
        A_recon = torch.mm(W, H)
        return F.mse_loss(A_recon, A_observed)


class RobustFittingLoss(LossFunction):
    """
    Noise-robust fitting loss: softplus(mse - sigma^2)

    For residuals below the noise floor sigma^2 the gradient is near-zero,
    so the optimiser is not penalised for noise-level deviations.
    Above the noise floor it recovers standard MSE behaviour.

    Args:
        sigma: float - expected noise standard deviation
    """

    def __init__(self, sigma: float, beta: float = 20.0):
        self.sigma = float(sigma)
        self.beta = float(beta)

    def forward(self, W: torch.Tensor, H: torch.Tensor, A_observed: torch.Tensor,
                iteration: int, total_iterations: int) -> torch.Tensor:
        A_recon = torch.mm(W, H)
        mse = F.mse_loss(A_recon, A_observed)
        return F.softplus(mse - self.sigma ** 2, beta=self.beta)


class SumPenaltyLoss(LossFunction):
    """
    Sum penalty loss: encourages columns of H to sum to 1.
    Penalty = mean((sum(H, dim=0) - 1)^2)
    
    This ensures that at each time point, the concentrations sum to 1 (conservation).
    """
    
    def forward(self, W: torch.Tensor, H: torch.Tensor, A_observed: torch.Tensor, 
                iteration: int, total_iterations: int) -> torch.Tensor:
        """Compute penalty for H columns not summing to 1."""
        return torch.mean(torch.square(torch.sum(H, dim=0) - 1.0))


class HFirstLoss(LossFunction):
    """
    H first column penalty loss: encourages first time point in H to equal [1, 0, 0, ...]
    Penalty = mean((H[:, 0] - [1, 0, 0, ...])^2)
    
    This provides an initial condition where only the first component is present.
    """
    
    def forward(self, W: torch.Tensor, H: torch.Tensor, A_observed: torch.Tensor, 
                iteration: int, total_iterations: int) -> torch.Tensor:
        """Compute penalty for first column of H not being [1, 0, 0, ...]."""
        k_components = H.shape[0]
        target = torch.eye(k_components, device=H.device)[:, 0]
        return torch.mean(torch.square(H[:, 0] - target))


class HComponentLimitBeforeTimeLoss(LossFunction):
    """
    Penalize H components above a cutoff index before a chosen time.

    If max_k = 1 and t = 10, then components 1..end are penalized whenever
    they deviate from zero in H[:, :10].
    """

    def __init__(self, max_k: int, t: int):
        self.max_k = int(max_k)
        self.t = int(t)

    def forward(self, W: torch.Tensor, H: torch.Tensor, A_observed: torch.Tensor,
                iteration: int, total_iterations: int) -> torch.Tensor:
        """Penalize components above max_k before time t."""
        if H.dim() != 2:
            raise ValueError("HComponentLimitBeforeTimeLoss expects H to be a 2D tensor")

        time_limit = max(0, min(self.t, H.shape[1]))
        component_limit = max(0, min(self.max_k, H.shape[0]))

        if time_limit == 0 or component_limit >= H.shape[0]:
            return H.new_tensor(0.0)

        return torch.mean(torch.square(H[component_limit:, :time_limit]))


class CompositeLoss(LossFunction):
    """
    Composite loss that combines multiple loss functions with weights.
    
    Args:
        losses: list of tuples (loss_function, weight, name, dynamic_penalty)
            - loss_function: LossFunction instance
            - weight: float - weight for this loss component
            - name: str - name for this loss component (for tracking)
            - dynamic_penalty: bool - if True, weight increases from weight/100 to weight
                                     over the first 2/3 of iterations (default: False).
                                     First loss weight is constant, subsequent penalties ramp up to allow initial fitting before enforcing constraints.
    """
    
    def __init__(self, losses: list):
        """
        Initialize composite loss.
        
        Args:
            losses: list of tuples (loss_function, weight, name) or 
                    (loss_function, weight, name, dynamic_penalty)
        """
        self.losses = losses
        self.loss_dict = {}
    
    def forward(self, W: torch.Tensor, H: torch.Tensor, A_observed: torch.Tensor, 
                iteration: int, total_iterations: int) -> torch.Tensor:
        """
        Compute weighted sum of all losses.
        Penalty weights increase from 1/100 to full weight over first 1/2 of iterations.
        
        Returns:
            torch.Tensor - Total weighted loss
        """
        total_loss = 0.0
        self.loss_dict = {}
        
        for loss_spec in self.losses:
            # Handle both 3-tuple and 4-tuple formats
            if len(loss_spec) == 4:
                loss_fn, weight, name, dynamic_penalty = loss_spec
            else:
                loss_fn, weight, name = loss_spec
                dynamic_penalty = False
            
            # Compute penalty weight multiplier (increases from 0.01 to 1.0)
            if dynamic_penalty:
                # Increase linearly from 1/100 to 1 over first 1/2 of iterations
                warmup_iters = int(0.5 * total_iterations)
                if iteration < warmup_iters:
                    # Linear schedule from 0.01 to 1.0
                    multiplier = 0.01 + 0.99 * (iteration / warmup_iters)
                else:
                    multiplier = 1.0
                effective_weight = weight * multiplier
            else:
                effective_weight = weight
                multiplier = 1.0
            
            loss_value = loss_fn(W, H, A_observed, iteration, total_iterations)
            weighted_loss = effective_weight * loss_value
            total_loss = total_loss + weighted_loss
            self.loss_dict[name] = loss_value
        
        return total_loss
    
    def get_loss_dict(self) -> dict:
        """Get dictionary of individual loss components (unweighted)."""
        return self.loss_dict


# ========================================
# Parameterized NMF Solver
# ========================================

class ParameterizedNMFSolver:
    """
    NMF Solver using parameterized matrices for W and H.
    """
    def __init__(self, W_param: MatrixParameterization, H_param: MatrixParameterization, 
                 A_observed: torch.Tensor, loss_function, device: str = 'cpu',
                 ):
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
        self.loss_function = loss_function
        self.device = device
        
        # Collect all parameters from both parameterizations
        self.params = list(self.W_param.parameters()) + list(self.H_param.parameters())
        
        # Tracking
        self.best_loss = float('inf')
        self.best_W = None
        self.best_H = None
        self.best_params = None
        self.loss_history = []
    
    def initialize_with_nmf(self, n_nmf_iterations=10000, print_every=1000):
        """
        Initialize parameters using standard NMF.
        
        Args:
            n_nmf_iterations: int - number of NMF iterations
        """
        print("\nInitializing with NMF...")
        N_rows, N_cols = self.A_observed.shape
        k = self.W_param.n_cols if hasattr(self.W_param, 'n_cols') else self.H_param.n_rows
        
        # W_init = torch.rand(N_rows, k, device=self.device)
        # H_init = torch.rand(k, N_cols, device=self.device)
        # # NMF iterations with sum-to-one constraint on H
        # for iteration in range(n_nmf_iterations):
        #     # Update H
        #     WtA = torch.mm(W_init.T, self.A_observed)
        #     WtW = torch.mm(W_init.T, W_init)
        #     H_init = H_init * WtA / (torch.mm(WtW, H_init) + 1e-10)
        #     # Normalize H columns to sum to 1
        #     H_init = H_init / (H_init.sum(dim=0, keepdim=True) + 1e-10)
            
        #     # Update W
        #     AHt = torch.mm(self.A_observed, H_init.T)
        #     HHt = torch.mm(H_init, H_init.T)
        #     W_init = W_init * AHt / (torch.mm(W_init, HHt) + 1e-10)
            
        #     if (iteration + 1) % 10 == 0:
        #         An_est = W_init @ H_init
        #         loss_nmf = F.mse_loss(An_est, self.A_observed)
        #         print(f"  NMF iteration {iteration+1}/{n_nmf_iterations}, Loss: {loss_nmf.item():.6f}")
        
        # Initialize with random values (requires_grad for optimization)
        W_raw = nn.Parameter(torch.randn(N_rows, k, device=self.device))
        H_raw = nn.Parameter(torch.randn(k, N_cols, device=self.device))
        init_lr = 1.0
        optimizer = optim.AdamW([W_raw, H_raw], lr=0.1, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-4
        )
        
        # Track best initialization
        best_loss = float('inf')
        best_W_init = None
        best_H_init = None
        best_iteration = 0
        
        # AdamW optimization loop with sum-to-one constraint on H
        current_lr = init_lr
        for iteration in range(n_nmf_iterations):
            optimizer.zero_grad()
            W_init = F.softplus(W_raw)
            H_init = F.softplus(H_raw)
            
            # Compute reconstruction loss
            total_loss = self.loss_function(W_init, H_init, self.A_observed, iteration, n_nmf_iterations)
            
            # Get individual losses for reporting (if using CompositeLoss)
            if isinstance(self.loss_function, CompositeLoss):
                losses = self.loss_function.get_loss_dict()
            else:
                losses = {}
            
            # Backward pass and optimization step
            total_loss.backward()
            optimizer.step()
            
            # Update learning rate scheduler
            scheduler.step(total_loss.item())
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                print(f"  LR reduced: {current_lr:.2e} -> {new_lr:.2e} at iteration {iteration+1}")
                current_lr = new_lr
            
            # Project to non-negative values
            with torch.no_grad():
                # Track best loss
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_W_init = W_init.detach().clone()
                    best_H_init = H_init.detach().clone()
                    best_iteration = iteration
            
            if (iteration + 1) % print_every == 0:
                loss_str = f"  NMF iteration {iteration+1}/{n_nmf_iterations}, Total Loss: {total_loss.item():.6f}"
                if losses:
                    for name, value in losses.items():
                        loss_str += f", {name}: {value.item():.6f}"
                loss_str += f", LR: {current_lr:.2e}"
                print(loss_str)
        
        # Use best iteration
        print(f"\nNMF initialization complete. Best loss: {best_loss:.6f} at iteration {best_iteration+1}/{n_nmf_iterations}")

        # --- Sort components by peak position in H_init ---
        # Find peak (argmax) for each component (row) in H_init
        peak_indices = torch.argmax(best_H_init, dim=1)
        sort_order = torch.argsort(peak_indices)
        # Sort W_init and H_init columns/rows accordingly
        best_W_init = best_W_init[:, sort_order]
        best_H_init = best_H_init[sort_order, :]

        # Initialize parameterizations from best NMF result
        print("Fitting parameterization to best NMF solution...")
        self.W_param.initialize_from_matrix(best_W_init)
        self.H_param.initialize_from_matrix(best_H_init)
        
        # Check how well the parameterization fits
        W_param_matrix = self.W_param.matrix()
        H_param_matrix = self.H_param.matrix()
        An_param = W_param_matrix @ H_param_matrix
        total_loss = self.loss_function(W_param_matrix, H_param_matrix, self.A_observed, 0, 1)
        
        # Get individual losses for reporting (if using CompositeLoss)
        if isinstance(self.loss_function, CompositeLoss):
            losses = self.loss_function.get_loss_dict()
            loss_str = f"Total: {total_loss.item():.6f}"
            for name, value in losses.items():
                loss_str += f", {name}: {value.item():.6f}"
            print(f"Parameterized approximation - {loss_str}\n")
        else:
            print(f"Parameterized approximation - Total: {total_loss.item():.6f}\n")
    
    def solve_single(self, n_iterations=1000, lr=0.01, print_every=10, use_scheduler=True, 
                     nmf_init=True, n_nmf_iterations=500):
        """
        Solve the NMF problem using gradient-based optimization with a single initialization.
        
        Args:
            n_iterations: int - number of optimization iterations
            lr: float - initial learning rate
            print_every: int - print loss every N iterations
            use_scheduler: bool - whether to use learning rate scheduler (Adam only)
            nmf_init: bool - whether to initialize with NMF (True) or random (False)
            n_nmf_iterations: int - number of NMF iterations for initialization (only used if nmf_init=True)
        
        Returns:
            tuple: (W_final, H_final, loss_history)
        """
        # Initialize parameters
        if nmf_init:
            self.initialize_with_nmf(n_nmf_iterations, print_every)
        
        # Create optimizer
        optimizer = optim.AdamW(self.params, lr=lr, weight_decay=1e-5)
        
        # Create learning rate scheduler
        if use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.3, patience=1000, min_lr=1e-4
            )
        
        print(f"Starting optimization for {n_iterations} iterations...")
        if use_scheduler:
            print(f"LR Scheduler: Enabled (ReduceLROnPlateau)")
        print("-" * 70)
        
        current_lr = lr
        
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # Compute loss
            total_loss = self.loss_function(self.W_param.matrix(), self.H_param.matrix(), 
                                           self.A_observed, iteration, n_iterations)
            
            # Get individual losses for reporting (if using CompositeLoss)
            if isinstance(self.loss_function, CompositeLoss):
                losses = self.loss_function.get_loss_dict()
            else:
                losses = {}
            
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
                self.best_W = self.W_param.matrix().detach().clone()
                self.best_H = self.H_param.matrix().detach().clone()
                self.best_params = {
                    'W_params': {k: v.detach().clone() for k, v in self.W_param.params.items()},
                    'H_params': {k: v.detach().clone() for k, v in self.H_param.params.items()},
                    'iteration': iteration
                }
            
            # Print progress
            if (iteration + 1) % print_every == 0 or iteration == 0:
                if use_scheduler:
                    lr_str = f"LR: {current_lr:.2e} | "
                else:
                    lr_str = ""
                    
                print_str = f"Iter {iteration+1:5d} | {lr_str}Total Loss: {loss_val:.6f} | "
                
                for k, v in losses.items():
                    print_str += f"{k}: {v.item():.6f} | "
                        
                print_str += f"Best: {self.best_loss:.6f}"
                print(print_str)
    
        print("-" * 70)
        print(f"Optimization complete!")
        print(f"Final loss: {self.loss_history[-1]:.6f}")
        print(f"Best loss: {self.best_loss:.6f} (at iteration {self.best_params['iteration']+1})")
        if use_scheduler:
            print(f"Final learning rate: {current_lr:.2e}")
        
        return self.best_W, self.best_H, self.loss_history
    
    def solve(self, n_runs=5, n_iterations=1000, lr=0.01, 
              print_every=10, use_scheduler=True, 
              nmf_init=True, n_nmf_iterations=500):
        """
        Run optimization multiple times with different initializations and return the best result.
        
        Args:
            n_runs: int - number of independent runs with different initializations
            n_iterations: int - number of optimization iterations per run
            lr: float - initial learning rate
            print_every: int - print loss every N iterations (0 to disable during runs)
            use_scheduler: bool - whether to use learning rate scheduler
            nmf_init: bool - whether to initialize with NMF (True) or random (False)
            n_nmf_iterations: int - number of NMF iterations for initialization (only used if nmf_init=True)
        
        Returns:
            W: torch.Tensor - best W matrix
            H: torch.Tensor - best H matrix
            loss_history: list - loss history from the best run
            all_results: list of dict - results from all runs for analysis
        """
        init_type = "NMF" if nmf_init else "random"
        print("=" * 70)
        print(f"Running optimization with {n_runs} different {init_type} initializations")
        print(f"Iterations per run: {n_iterations}")
        print("=" * 70)
        
        best_loss = float('inf')
        best_W = None
        best_H = None
        best_loss_history = None
        best_run_idx = -1
        all_results = []
        
        for run_idx in range(n_runs):
            print(f"\n{'='*70}")
            print(f"RUN {run_idx + 1}/{n_runs}")
            print(f"{'='*70}")
            
            # Run solve_single with specified initialization
            W, H, loss_history = self.solve_single(
                n_iterations=n_iterations,
                lr=lr,
                print_every=print_every if print_every > 0 else n_iterations + 1,  # Disable printing if 0
                use_scheduler=use_scheduler,
                nmf_init=nmf_init,
                n_nmf_iterations=n_nmf_iterations
            )
            
            final_loss = loss_history[-1]
            
            # Store results
            all_results.append({
                'run_idx': run_idx,
                'final_loss': final_loss,
                'W': W.clone(),
                'H': H.clone(),
                'loss_history': loss_history.copy()
            })
            
            # Update best if this run is better
            if final_loss < best_loss:
                best_loss = final_loss
                best_W = W.clone()
                best_H = H.clone()
                best_loss_history = loss_history.copy()
                best_run_idx = run_idx
                print(f"\n>>> New best result! Loss: {final_loss:.6f}")
            else:
                print(f"\nRun {run_idx + 1} final loss: {final_loss:.6f} (best: {best_loss:.6f})")
        
        # Final summary
        print("\n" + "=" * 70)
        print("MULTIPLE INITIALIZATION SUMMARY")
        print("=" * 70)
        print(f"Best run: {best_run_idx + 1}/{n_runs}")
        print(f"Best final loss: {best_loss:.6f}")
        print("\nAll runs:")
        for i, result in enumerate(all_results):
            marker = " <<<" if i == best_run_idx else ""
            print(f"  Run {i+1}: {result['final_loss']:.6f}{marker}")
        print("=" * 70)
        
        return best_W, best_H, best_loss_history, all_results
    
    def get_factors(self):
        """Get the best W and H matrices."""
        return self.best_W, self.best_H


# ========================================
# Utility Functions
# ========================================
def plot_wh(w, h, prefix="", out_folder="out_parameterized_nmf"):
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

def plot_results(W_opt, H_opt, An, W_true, H_true, uv_wavelengths, loss_history=None, prefix="", out_folder="out_parameterized_nmf"):
    """
    Plot the results of the matrix factorization with proper scaling
    """
    if isinstance(W_opt, torch.Tensor):
        W_opt = W_opt.cpu().detach()
        H_opt = H_opt.cpu().detach()
    elif isinstance(W_opt, GaussianParameterization):
        # Align components to match the true factors for better comparison
        alignment = H_opt.params['means_unconstrained'].argsort()
        W_opt = W_opt.matrix().detach().cpu()
        H_opt = H_opt.matrix().detach().cpu()
        W_opt = W_opt[:, alignment]
        H_opt = H_opt[alignment, :]
    else:
        W_opt = W_opt.matrix().detach().cpu()
        H_opt = H_opt.matrix().detach().cpu()
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
