# Optimization Problem

## Objective

$$
\min_{W_{UV}, W_{RS}, H} \ell_{UV}(A_{UV}, W_{UV}H) + \ell_{RS}(A_{RS}, W_{RS}H))
$$

## Constraints

1. **Positivity:**
   $$H, W_{UV}, W_{RS} \geq 0$$

2. **Closedness: (consetrations some to 100% at each timestep)**
   $$H\mathbf{1} = \mathbf{1}$$

3.  **Unimodality:**
   $$H \text{ is unimodal}$$

4. **Maybe we should add smoothness constraint on W or also use spline with higher order**


## Likelihood Functions

The likelihood functions $\ell_{UV}$ and $\ell_{RS}$ encode how we choose to impose data fidelity for UV and RS respectively. 

Since UV is quantitative, we can write:

$$
\ell_{UV}(E_{UV}, W_{UV}H_{liq}) = \|E_{UV} - W_{UV}H_{liq}\|^2_2
$$

Since RS is qualitative in terms of the peak intensities but quantitative in terms of peak positions, we can demand that:

$$
\ell_{RS}(E_{RS}, W_{RS}H) = \|\text{peaks}(E_{RS}) - \text{peaks}(W_{RS}(H_{liq} + H_{sol}))\|
$$

## Notation

- $W_{UV}$ : UV-vis spectral components
- $W_{RS}$ : Raman spectral components  
- $H$ : concentration profiles
- $A_{UV}$ : UV-vis experimental data
- $A_{RS}$ : Raman experimental data
- $\ell_{UV}$ : Loss function for UV-vis data
- $\ell_{RS}$ : Loss function for Raman data
- $\mathbf{1}$ : Vector of ones
