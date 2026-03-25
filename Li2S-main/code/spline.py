import numpy as np
from scipy.interpolate import CubicSpline
from typing import Tuple, Optional

class UnimodalPositiveSpline:
    def __init__(self, num_control_points: int = 5):
        """
        Initialize a unimodal positive spline with specified number of control points.
        
        Args:
            num_control_points: Number of control points to use for the spline
        """
        self.num_control_points = num_control_points
        self.spline = None
        
    def _ensure_unimodal_positive(self, y: np.ndarray) -> np.ndarray:
        """
        Adjust y values to ensure they form a unimodal positive sequence.
        """
        # Ensure all values are positive
        y = np.maximum(y, 0.0)
        
        # Find the peak index
        peak_idx = np.argmax(y)
        
        # Ensure monotonic increase before peak
        for i in range(1, peak_idx):
            y[i] = max(y[i], y[i-1])
            
        # Ensure monotonic decrease after peak
        for i in range(peak_idx + 1, len(y)):
            y[i] = min(y[i], y[i-1])
            
        return y
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> 'UnimodalPositiveSpline':
        """
        Fit a unimodal positive spline to the given data points.
        
        Args:
            x: x-coordinates of the data points
            y: y-coordinates of the data points
        
        Returns:
            self: The fitted spline object
        """
        # Create control points
        x_control = np.linspace(x.min(), x.max(), self.num_control_points)
        
        # Interpolate y values at control points
        y_control = np.interp(x_control, x, y)
        
        # Ensure unimodality and positivity
        y_control = self._ensure_unimodal_positive(y_control)
        
        # Fit cubic spline
        self.spline = CubicSpline(x_control, y_control)
        
        return self
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the spline at given x points.
        
        Args:
            x: Points at which to evaluate the spline
            
        Returns:
            y: Spline values at x
        """
        if self.spline is None:
            raise ValueError("Spline must be fitted before evaluation")
        
        return np.maximum(self.spline(x), 0.0)  # Ensure output is positive

# Example usage
if __name__ == "__main__":
    # Generate sample data
    x = np.linspace(0, 10, 100)
    y = 2 * np.exp(-(x - 5)**2 / 4) + 0.1 * np.random.randn(len(x))
    
    # Create and fit the spline
    spline = UnimodalPositiveSpline(num_control_points=7)
    spline.fit(x, y)
    
    # Evaluate the spline
    y_spline = spline(x)
    
    # Plot the results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5, label='Data points')
    plt.plot(x, y_spline, 'r-', label='Unimodal positive spline')
    plt.legend()
    plt.grid(True)
    plt.title('Unimodal Positive Spline Fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
