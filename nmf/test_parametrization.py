
import os, sys
import pandas as pd
import torch
import torch.nn.functional as F


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nmf.parameterized_nmf import *
from nmf.parameterized_nmf_raman import GaussianThresholdParameterization

data_path = '/Users/tomer/private/NMF/data/synth1/'
out_folder = "/Users/tomer/private/NMF/out_parameterized_nmf/synth1"
os.makedirs(out_folder, exist_ok=True)

def test_vis():
    # Load UV-vis data
    df_An = pd.read_csv(data_path + "An.csv", header=None)
    uv_wavelengths = df_An.iloc[1:, 0].values.astype(float)
    An = torch.from_numpy(df_An.iloc[1:, 1:].values.astype(float)).to(torch.float32)
    
    df_W_true = pd.read_csv(data_path + "W_true.csv", header=None)
    W_true = torch.from_numpy(df_W_true.iloc[1:, 1:].values.astype(float)).to(torch.float32)
    W_true /= 100  # Scale down W_true to match the scale of An and improve optimization stability
    
    df_H_true = pd.read_csv(data_path + "H_true.csv", header=None)
    H_true = torch.from_numpy(df_H_true.iloc[1:, 2:].values.astype(float)).to(torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    An = An.to(device)
    W_true = W_true.to(device)
    H_true = H_true.to(device)
    # plot_wh(W_true, H_true, An, W_true, H_true, uv_wavelengths, prefix='gt')
    
    # Create parameterizations
    k_components = W_true.shape[1]
    n_wavelengths, n_timepoints = An.shape

    # W_param = MixtureOfGaussiansParameterization(
    #     shape=(n_wavelengths, k_components), 
    #     n_gaussians=2,
    #     mean_bounds=(0, n_wavelengths),
    #     std_bounds=(0.1, n_wavelengths / 5),
    #     scale_bounds=(0, 4.0)
    # )
    
    H_param = GaussianParameterization(
        shape=(k_components, n_timepoints),
        axis=0,  # Row-wise for H matrix
        mean_bounds=(0, n_timepoints),
        std_bounds=(0.1, n_timepoints / 8),
        scale_bounds=(0, 1.0)
    )
    H_param = GeneralizedGaussianParameterization(
        shape=(k_components, n_timepoints),
        axis=0,
        mean_bounds=(0, n_timepoints),
        std_bounds=(0.1, n_timepoints / 8),
        beta_bounds=(1., 4.0),
        scale_bounds=(0, 1.0)
    )
    W_param = SplineParameterization(
        shape=(n_wavelengths, k_components), 
        n_control_points=30,
        value_bounds=(0, 4.0)
    )
    
    # H_param = SplineParameterization(
    #     shape=(k_components, n_timepoints),
    #     n_control_points=15,
    #     axis=0,  # Row-wise for H matrix
    #     value_bounds=(0, 1.0)
    # )
    plot_results(W_param, H_param, An, W_true, H_true, uv_wavelengths, prefix='initial')


    W_param.initialize_from_matrix(W_true)
    H_param.initialize_from_matrix(H_true)
    plot_results(W_param, H_param, An, W_true, H_true, uv_wavelengths, prefix='fit')

def test():
    # Load 
    df_An = pd.read_csv(data_path + "An.csv", header=None)
    wavelengths = df_An.iloc[1:, 0].values.astype(float)
    An = torch.from_numpy(df_An.iloc[1:, 1:].values.astype(float)).to(torch.float32)
    An /= 100.0  # Scale down An to improve optimization stability
    
    df_W_true = pd.read_csv(data_path + "W_true.csv", header=None)
    W_true = torch.from_numpy(df_W_true.iloc[1:, 1:].values.astype(float)).to(torch.float32)
    W_true /= 100.0  # Scale down W_true to improve optimization stability
    
    df_H_true = pd.read_csv(data_path + "H_true.csv", header=None)
    H_true = torch.from_numpy(df_H_true.iloc[1:, 2:].values.astype(float)).to(torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    An = An.to(device)
    W_true = W_true.to(device)
    H_true = H_true.to(device)
    
    # Create parameterizations
    k_components = W_true.shape[1]
    n_wavelengths, n_timepoints = An.shape
    print(f"n_wavelengths: {n_wavelengths}, n_timepoints: {n_timepoints}, k_components: {k_components}")

    # W_param = MixtureOfGaussiansParameterization(
    #     shape=(n_wavelengths, k_components), 
    #     n_gaussians=3,
    #     axis=1,  # Column-wise for W matrix
    #     mean_bounds=(0, n_wavelengths),
    #     std_bounds=(0.1, n_wavelengths / 5),
    #     scale_bounds=(0, 10.0)
    # )
    W_param = MixtureOfGeneralizedGaussiansParameterization(
        shape=(n_wavelengths, k_components), 
        n_gaussians=3,
        axis=1,  # Column-wise for W matrix
        mean_bounds=(0, n_wavelengths),
        std_bounds=(10, n_wavelengths/10),
        scale_bounds=(0, 10.0),
        beta_bounds=(1.5, 3)
    )
    # W_param = SplineParameterization(
    #     shape=(n_wavelengths, k_components), 
    #     n_control_points=30,
    #     value_bounds=(0, 5.0)
    # )
    
    # H_param = GeneralizedGaussianParameterization(
    #     shape=(k_components, n_timepoints),
    #     axis=0,  # Row-wise for H matrix
    #     mean_bounds=(0, n_timepoints),
    #     std_bounds=(0.1, n_timepoints),
    #     beta_bounds=(1., 4.0),
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
    # plot_results(W_param, H_param, An, W_true, H_true, wavelengths, prefix='initial')


    # W_param.initialize_from_matrix(W_true)
    H_param.initialize_from_matrix(H_true)
    # print(W_param)
    # print(H_param)
    plot_results(W_param, H_param, An, W_true, H_true, wavelengths, prefix='fit')


    
    
if __name__ == "__main__":    
    test()