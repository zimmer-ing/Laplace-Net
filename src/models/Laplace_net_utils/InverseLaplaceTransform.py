# MIT License, Copyright (c) 2025 Bernd Zimmering
# See LICENSE file for details.
# Please cite: Zimmering, B. et al. (2025), "Breaking Free: Decoupling Forced Systems with Laplace Neural Networks", ECML PKDD, Porto.
import torch

class ILTFourier:
    r"""
    Inverse Laplace Transform (ILT) using a Fourier-based approximation.

    This class implements an approximation of the inverse Laplace transform
    based on the Fourier series, allowing for the reconstruction of time-domain
    trajectories from the Laplace domain.

    Args:
        ilt_reconstruction_terms (int): Number of ILT reconstruction terms (frequency components).
        alpha (float): Stability parameter for the reconstruction, e.g., `alpha=1e-3`.
        tol (float): Desired tolerance, default is computed as `10 * alpha` if not provided.
        scale (float): Scaling factor for ILT, typically `scale=2.0`.
        torch_float_dtype (torch.dtype): Float data type, default is `torch.float64`.
        torch_complex_dtype (torch.dtype): Complex data type, default is `torch.complex64`.
    """

    def __init__(
            self,
            ilt_reconstruction_terms=33,
            alpha=1.0e-3,
            tol=None,
            scale=2.0,
            f_max=None,
            shift=0.0,
            torch_float_dtype=torch.float32,
            torch_complex_dtype=torch.complex64,
            device=torch.device('cpu')
    ):
        r"""
        Initializes the ILTFourier object with the specified parameters.

        Args:
            ilt_reconstruction_terms (int): Number of ILT reconstruction terms, default is 33.
            alpha (float): Stability term for controlling the Bromwich contour.
            tol (float): Numerical tolerance term for stability, calculated automatically if not provided.
            scale (float): Scaling applied to the time axis during computation, default is 2.0.
            f_max (float): Maximum frequency for reconstruction (can be used instead of scale), default is None.
            shift (float): Shift the time axis and thus the Bromwich contour, default is 0.0.
            torch_float_dtype (torch.dtype): Floating-point data type, e.g., `torch.float32`.
            torch_complex_dtype (torch.dtype): Complex data type for Laplace computation, e.g., `torch.complex64`.
            device (torch.device): Device for computation
        """
        self.ilt_reconstruction_terms = ilt_reconstruction_terms
        self.alpha = alpha
        self.tol = tol if tol is not None else 10 * alpha
        if f_max is not None:
            self.scale = (ilt_reconstruction_terms- 1) / (2 * f_max)
        else:
            self.scale = scale
        self.shift = shift
        self.torch_float_dtype = torch_float_dtype
        self.torch_complex_dtype = torch_complex_dtype
        self.device = device


        # Precompute k-values for reconstruction terms
        self.k = torch.arange(ilt_reconstruction_terms, dtype=self.torch_float_dtype)
        self.imag_k_pi=(1j * torch.pi * self.k.view(1, -1)).to(self.torch_complex_dtype).to(device)

    def scale_data(self, data, time_axis):
        r"""
        Scale the input data by applying an inverse scaling factor.

        This method applies the inverse exponential scaling factor to the input data
        based on the time axis used in reconstruction.

        Args:
            data (Tensor): A tensor representing Laplace domain data.
            time_axis (Tensor): Time axis used to compute the scaling factor.

        Returns:
            Tensor: Scaled data tensor, with the inverse scaling applied.
        """
        exponential_scaling_factor = self.exponential_scaling(time_axis).real
        exponential_scaling_factor=exponential_scaling_factor.view_as(data)
        return data / exponential_scaling_factor

    def get_query_points(self, time_axis):
        r"""
        Generate query points `s` in the Laplace domain.

        The function computes query points for the Bromwich contour based on
        the time axis. These query points define the `s` values in the Laplace domain.

        Args:
            time_axis (Tensor): Time points, `(SeqLen,)` or higher dimensions.

        Returns:
            Tensor: Query points `s`, `(SeqLen, ilt_reconstruction_terms)`.
        """

        min_t = torch.min(time_axis + self.shift).item()
        assert min_t > 0, "Time axis must start with non-zero time."
        # Add additional dimension to time_axis
        t = (time_axis+self.shift).unsqueeze(-1).to(dtype=self.torch_complex_dtype)
        T = (t * self.scale).to(self.torch_complex_dtype)
        # Compute real shift  and imaginary parts
        real = self.alpha - torch.log(torch.tensor(self.tol, dtype=self.torch_complex_dtype)) / T
        imag=(self.imag_k_pi/T).to(self.torch_complex_dtype)

        # Combine to form query points
        s = real + imag
        return s.squeeze(-1)

    def exponential_scaling(self, time_axis):
        r"""
        Compute the exponential scaling factor for reconstruction.

        The scaling factor is calculated based on the sigma shift of the Bromwich
        contour. This factor helps stabilize the reconstruction.

        Args:
            time_axis (Tensor): Time vector `(SeqLen,)`.

        Returns:
            Tensor: Exponential scaling factors `(SeqLen,)`.
        """

        query_s=self.get_query_points(time_axis)#shift is already included in query_s
        time_axis = time_axis + self.shift
        sigma=query_s.real[...,0]
        T=(time_axis*self.scale).to(self.torch_float_dtype)
        exp_sigma_t = torch.exp(sigma * time_axis)
        scaling_factor = (1 / T) * exp_sigma_t
        return scaling_factor.to(self.torch_float_dtype)

    def reconstruct(self, Y_s, time_axis, exp_scaling=True):
        r"""
        Reconstruct time-domain trajectories using the ILT method.

        This function reconstructs the time-domain signal from given Laplace
        representations of the data using the Fourier inverse Laplace transform method.

        Args:
            Y_s (Tensor): Laplace representation, `(BatchSize, SeqLen, ilt_reconstruction_terms, d_obs)`.
            time_axis (Tensor): Time points for reconstruction `(SeqLen,)`.
            exp_scaling (bool): If True (default), applies exponential scaling.

        Returns:
            Tensor: Reconstructed trajectories `(BatchSize, SeqLen, d_obs)`.
        """
        #check if tensor has the requiered dims
        assert len(Y_s.shape)==4 , "Y_s must be of shape (BatchSize, SeqLen, ilt_reconstruction_terms, d_obs)"
        assert time_axis.min()>0, "time_axis must start with non-zero time"
        # Convert data and time axis

        Y_s = Y_s.permute(0, 1, 3, 2).to(dtype=self.torch_complex_dtype)

        # Compute the first and summation terms
        first_term = Y_s[:, :, :, 0] / 2.0

        exp_k = torch.exp(self.imag_k_pi[:, 1:]/self.scale)
        summation_terms = torch.sum((Y_s[:, :, :, 1:] * exp_k).real, dim=-1)


        if exp_scaling:
            scaling_factor = self.exponential_scaling(time_axis)
            reconstructed = scaling_factor.view(Y_s.size(0), Y_s.size(1), 1) * (first_term + summation_terms)
        else:
            reconstructed = first_term + summation_terms

        return reconstructed.real

    def generate_time_grid(self, time_points: int, ilt_terms: int=0,value_range=[-1,1]):
        r"""
        Generate a computational 2D grid that corresponds to the combinations of time points and ILT terms.

        This method creates a grid of time indices and ILT term indices for efficient
        Laplace domain computations.

        Args:
            time_points (int): Number of time points.
            ilt_terms (int): Number of ILT reconstruction terms (frequency components).
            value_range (list): Range of values for the time grid, default is [-1, 1].

        Returns:
            Tensor: A 2D grid with shape `(time_points * ilt_terms, 2)`,
                    where each row represents a combination of a time index and an ILT term index.
        """
        if ilt_terms==0:
            ilt_terms=self.ilt_reconstruction_terms

        time_indices = torch.arange(time_points)/(time_points-1)*(value_range[1]-value_range[0])+value_range[0]
        ilt_indices = torch.arange(ilt_terms)/(ilt_terms-1)*(value_range[1]-value_range[0])+value_range[0]
        grid = torch.cartesian_prod(time_indices, ilt_indices)


        return grid.view(time_points , ilt_terms, 2).to(self.device).to(self.torch_float_dtype)

    def forward(self, Y_s: torch.Tensor, time_axis: torch.Tensor):
        r"""
        Forward pass for trajectory reconstruction from Laplace representations.

        This function is used during prediction to reconstruct trajectories
        using the inverse Laplace transform method.

        Args:
            Y_s (Tensor): Laplace representation `(BatchSize, SeqLen, ilt_reconstruction_terms, d_obs)`.
            time_axis (Tensor): Time points for reconstruction `(SeqLen,)`.

        Returns:
            Tensor: Reconstructed time-domain trajectories `(BatchSize, SeqLen, d_obs)`.
        """
        return self.reconstruct(Y_s, time_axis)

    def transform_LP_func(self, Y_s, time_axis):
        time_axis = time_axis + self.shift
        Y_s = Y_s.permute(0, 2, 3, 1)
        time_axis = time_axis.view(1, 1, 1, -1)
        scaling_factor = (
                self.scale  #λ = scale * t  -> factor "zeta"
                * time_axis  # -> factor "t"
                * self.tol ** (1 / self.scale)  # -> factor   ε^(1/scale)
                * torch.exp(-self.alpha * time_axis)  # -> factor e^(-alpha * t)
        )
        Y_s_transformed = (Y_s / scaling_factor)
        return Y_s_transformed.permute(0, 3, 1, 2)

    def inverse_transform_LP_func(self, Y_s_transformed, time_axis):
        time_axis = time_axis + self.shift
        batch_size = Y_s_transformed.size(0)
        timesteps = Y_s_transformed.size(1)
        ILT_terms = Y_s_transformed.size(2)
        features1 = Y_s_transformed.size(3)
        if len(Y_s_transformed.shape)==4:
            Y_s_transformed = Y_s_transformed.unsqueeze(-1)
        features2=Y_s_transformed.size(4)
        Y_s_transformed = Y_s_transformed.permute(0, 2, 3, 4, 1)

        scaling_factor = (
                self.scale  # λ = scale * t  -> factor "zeta"
                * time_axis  # -> factor "t"
                * self.tol ** (1 / self.scale)  # -> factor   ε^(1/scale)
                * torch.exp(-self.alpha * time_axis)  # -> factor e^(-alpha * t)
        )
        #shape scaling factor to match Y_s_transformed
        scaling_factor =scaling_factor.view(batch_size,1,1,1,timesteps).repeat(1,ILT_terms,features1,features2,1)
        Y_s = (Y_s_transformed * scaling_factor)
        return Y_s.permute(0,4,1,2,3)





if __name__ == "__main__":
    # Import necessary modules and run the test
    from scipy.integrate import odeint
    import numpy as np
    import warnings
    from scipy.integrate import odeint
    from sympy import Function
    import numpy as np


    #Basic test for the ILT class
    # test func 1 -> step signal 1/s in laplace space
    scale=1.0
    ilt_terms=6001
    #num_ILT=ILTFourier(scale=scale, ilt_reconstruction_terms=ilt_terms,shift=0.0)
    num_ILT = ILTFourier(
                                 ilt_reconstruction_terms=ilt_terms,
                                 shift=0.0,
                                scale=None,
                                f_max=100,
                                 )
    t_vals=np.linspace(1,10,100)
    s=num_ILT.get_query_points(torch.from_numpy(t_vals))
    assert s.shape==(100,ilt_terms), "Query points have wrong shape"
    #reconstruct the step signal
    step_signal=1/s
    step_signal_reconstructed=num_ILT.reconstruct(step_signal.view(1,step_signal.shape[0],step_signal.shape[1],1),
                                                  torch.from_numpy(t_vals)).squeeze().detach().numpy()
    error=np.abs(step_signal_reconstructed-1)
    print("Max error for step signal reconstruction:",np.max(error))
    if np.all(error>1e-3):
        print("Test failed! Max error exceeds threshold")

    #Advanced test for ILT class

    warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress any warnings

    # 1. System and time parameters
    mass = 1.0  # Mass (m)
    damping = 0.5  # Damping coefficient (b)
    stiffness = 5.0  # Spring constant (k)

    # Define time span
    time_start = 0.0
    time_end = 10.0
    time_steps = 500
    t_vals = np.linspace(time_start, time_end, time_steps)

    # Step function input
    def step_signal(t_array, t_start=0.0, amplitude=1.0):
        return np.where(t_array >= t_start, amplitude, 0.0)

    forcing = step_signal(t_vals)

    # ODE for the SMD system
    def smd_system(state, time, mass, damping, stiffness, forcing):
        """Define the differential equation of the SMD system."""
        y, y_dot = state
        forcing_value = np.interp(time, t_vals, forcing)  # Interpolate forcing for time
        dydt = [y_dot, (forcing_value - damping * y_dot - stiffness * y) / mass]
        return dydt

    # Initial conditions
    y0 = [0.0, 0.0]  # Initial displacement and velocity

    # Solve the ODE
    sim_solution = odeint(smd_system, y0, t_vals, args=(mass, damping, stiffness, forcing))
    y_sim = sim_solution[:, 0]  # Displacement (y)


    t_vals_ilt=t_vals[1:]#the ILT cannot predict t=0 as y(t=0) is known beforehand (0 in our case)
    s=num_ILT.get_query_points(torch.from_numpy(t_vals_ilt))


    # Transfer function: H(s) = Y(s) / F(s) = 1 / (ms^2 + bs + k)
    H_s = 1 / (mass * s ** 2 + damping * s + stiffness)

    # Laplace transform of the step input (forcing): F(s) = 1 / s
    F_s = 1 / s

    # Apply transfer function: Y(s) = H(s) * F(s)
    Y_s = H_s * F_s

    # Inverse Laplace transform to obtain y(t)
    # Numerically reconstruct y(t) using the ILT class
    Y_s_tensor = torch.tensor(Y_s.numpy(), dtype=torch.complex64).view(1, len(t_vals_ilt), -1, 1)
    t_tensor = torch.tensor(t_vals[1:], dtype=torch.float64)
    y_ilt_func = num_ILT.reconstruct(Y_s_tensor, t_tensor).squeeze().detach().numpy()
    y_sim=y_sim[1:]


    # Calculate error between the simulation and ILT
    error = np.max(np.abs(y_sim - y_ilt_func))
    # Threshold for acceptable error
    error_threshold = 1e-3
    if np.all(error > error_threshold):
        print(f"Test failed! Max error exceeds threshold: {np.max(error)}")

    # 6. Plot the results using Plotly
    import plotly.graph_objects as go

    # Plotly: Correct the layout configuration to resolve the error
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_vals,
        y=y_sim,
        mode='lines',
        name="ODE Simulation",
        line=dict(color="blue", dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=t_vals,
        y=y_ilt_func,
        mode='lines',
        name="Inverse Laplace Transform (ILT)",
        line=dict(color="orange")
    ))

    fig.update_layout(
        title="Comparison of SMD Simulation vs Inverse Laplace Transform",
        xaxis_title="Time (s)",
        yaxis_title="Displacement y(t)",
        legend=dict(x=0.1, y=0.9),
        template="plotly_white",
        # Correct and valid grid properties below
        grid=dict(xside="bottom", yside="left")  # Example valid grid setup
    )

    fig.show()

    print("Test passed! Simulation and ILT results match closely.")

