# MIT License, Copyright (c) 2025 Bernd Zimmering
# See LICENSE file for details.
# Please cite: Zimmering, B. et al. (2025), "Breaking Free: Decoupling Forced Systems with Laplace Neural Networks", ECML PKDD, Porto.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .InverseLaplaceTransform import ILTFourier
HIGH_PRECISION_FLOAT_DTYPE=torch.float64
LOW_PRECISION_FLOAT_DTYPE=torch.float32
HIGH_PRECISION_COMPLEX_DTYPE=torch.complex128
LOW_PRECISION_COMPLEX_DTYPE=torch.complex64

def spherical_riemann_to_complex(theta, phi):
    r""" (Taken from Holt et al. 2022: https://github.com/samholt/NeuralLaplace/blob/fd2d5f0f3475c2b0bb0078e2872cdd4e174615ed/torchlaplace/transformations.py
    Spherical Riemann stereographic projection coordinates to complex number coordinates. I.e. inverse Spherical Riemann stereographic projection map.

    The inverse transform, :math:`v: \mathcal{D} \rightarrow \mathbb{C}`, is given as

    .. math::
        \begin{aligned}
            s = v(\theta, \phi) = \tan \left( \frac{\phi}{2} + \frac{\pi}{4} \right) e^{i \theta}
        \end{aligned}

    Args:
        theta (Tensor): Spherical Riemann stereographic projection coordinates, shape :math:`\theta` component of shape :math:`(\text{dimension})`.
        phi (Tensor): Spherical Riemann stereographic projection coordinates, shape :math:`\phi` component of shape :math:`(\text{dimension})`.

    Returns:
        Tuple Tensor of real and imaginary components of the complex numbers coordinate, of :math:`(\Re(s), \Im(s))`. Where :math:`s \in \mathbb{C}^d`, with :math:`d` is :math:`\text{dimension}`.

    """
    # dim(phi) == dim(theta), takes in spherical co-ordinates returns comlex real & imaginary parts
    if phi.shape != theta.shape:
        raise ValueError("Invalid phi theta shapes")
    r = torch.tan(phi / 2 + torch.pi / 4)
    s_real = r * torch.cos(theta)
    s_imag = r * torch.sin(theta)
    return s_real, s_imag


class InitialStateNN(nn.Module):
    """
    A neural network to approximate the initial states P(s), allowing for configurable hidden layers,
    activation functions, and handling of Laplace query points with optional spherical projection.
    """

    def __init__(self,
                 dim_m,
                 dim_history,
                 order_dim=2,
                 RNN_hidden_dim=64,
                 RNN_num_layers=1):
        """
        Args:
            dim_terms_ilt (int): Number of ILT terms, representing query points in the Laplace domain.
            dim_m (int): Dimensionality of the vector m, which indicates the number of input features for the excitation.
            dim_history (int): Input dimensionality of the excitation time series.
            hidden_dim (int): Number of neurons in each hidden layer of the network.
            num_hidden_layers (int): Total number of hidden layers in the network.
            activation_fn (callable): The activation function to apply within the network layers.
            use_sphere_projection (bool): Indicates whether to apply spherical projection for input/output.
            process_all_terms_as_features (bool): If True, processes all ILT terms as a single set of features for input.
            RNN_output_hidden_dim (int): Dimensionality of the RNN hidden state that is output for each sequence.
        """
        super(InitialStateNN, self).__init__()

        self.dim_m = dim_m
        self.order_dim = order_dim

        # Initialize the RNN encoder for handling the excitation sequence backwards in time.
        self.RNN_encoder = nn.GRU(
            input_size=dim_history,
            hidden_size=RNN_hidden_dim,
            num_layers=RNN_num_layers,
            batch_first=True,
        )
        self.P_net = nn.Linear(RNN_hidden_dim, order_dim*dim_m)


    @staticmethod
    def polynomial_function(coefficients, s_queries):
        """
        Computes a polynomial for a list of s-values using the given coefficients.

        Args:
            coefficients (torch.Tensor): The coefficients of the polynomial in ascending order,
                                         with shape (B, T, F, order) where F is the feature dimension.
            s_queries (torch.Tensor): A tensor of s-values for which the polynomial is evaluated.
                                      Expected shape (B, T, 1, ilt_terms).

        Returns:
            torch.Tensor: The evaluated polynomial values with shape (B, T, F, ilt_terms).
        """
        # Ensure s_queries has the correct shape.
        # For example, if s_queries has shape (B, T, ilt_terms), add a singleton feature dim.
        if s_queries.dim() == 3:
            s_queries = s_queries.unsqueeze(2)  # Now shape is (B, T, 1, ilt_terms)

        order = coefficients.shape[-1]  # the number of polynomial coefficients
        # Initialize results with the appropriate shape: (B, T, F, ilt_terms)
        B, T, F, _ = coefficients.shape
        results = torch.zeros(B, T, F, s_queries.shape[-1],
                              device=coefficients.device, dtype=s_queries.dtype)

        # Loop over each order and let broadcasting handle the feature dimension.
        for i in range(order):
            # coefficients[..., i] has shape (B, T, F).
            # Unsqueeze to shape (B, T, F, 1) so that it broadcasts with s_queries (B, T, 1, ilt_terms)
            c = coefficients[..., i].unsqueeze(-1)
            # s_queries ** i has shape (B, T, 1, ilt_terms) and will broadcast along the feature dimension.
            results += c * (s_queries ** i)

        return results.permute(0, 1, 3, 2)


    def forward(self,  s_queries: Tensor,history_sequence: Tensor,):
        """
        Forward pass for the Input Excitation NN.

        Args:
            time_grid (Tensor): Time grid for the sequences. Shape: [batch, time_dim, 1].
            history_sequence (Tensor): Excitation time series. Shape: [batch, seq_len_excitation, dim_excitation].

        Returns:
            Tensor: The complex-valued input excitation X(s).
        """

        batch_dim, time_dim, features ,ilt_terms_dim = s_queries.shape

        out, _ = self.RNN_encoder(history_sequence)
        P = self.P_net(out[:, -1, :]).view(batch_dim,self.dim_m,self.order_dim).unsqueeze(1).expand(-1, time_dim, -1, -1)

        return self.polynomial_function(P, s_queries),out[:, -1, :]

class TransferFunctionNN(nn.Module):
    """
    Neural Network architecture designed to approximate transfer functions H(s) in the complex domain.
    Supports flexible configurations of hidden layers and activation functions, optimized for complex-valued data.
    """

    def __init__(self,
                 dim_terms_ilt=1,
                 dim_m=2,
                 dim_n=2,
                 hidden_dim=64,
                 state_dim=64,
                 num_hidden_layers=3,
                 activation_fn=F.relu,
                 process_all_terms_as_features=False):
        """
        Args:
            dim_terms_ilt (int): Number of Inverse Laplace Transform (ILT) terms or evaluation points in the domain.
            dim_m (int): Dimensionality of system inputs (latent states or excitations).
            dim_n (int): Dimensionality of system outputs (observations or responses).

            hidden_dim (int): Size of the hidden layers, determining the network's capacity.
            num_hidden_layers (int): Depth of the network, i.e., the number of fully connected layers.
            activation_fn (callable): Non-linear activation function applied after each hidden layer.
            process_all_terms_as_features (bool): If True, the network processes ILT terms in aggregate (multi-term mode);
            otherwise, terms are treated sequentially (single-term mode).
        """
        super(TransferFunctionNN, self).__init__()
        self.activation_fn = activation_fn
        self.dim_latent_state = dim_m
        self.dim_response = dim_n
        self.process_all_terms_as_features = process_all_terms_as_features

        # Configure input-output dimensions based on the processing mode of ILT terms.
        if process_all_terms_as_features:
            input_dim = 2 * dim_terms_ilt+state_dim  # Complex-valued grid: real and imaginary axes for all ILT terms.
            output_dim = 2 * (dim_m * dim_n * dim_terms_ilt)  # Complex-valued response, coupling ILT terms with inputs/outputs.

        else:  # Single-term processing mode:
            input_dim = 2 +state_dim
            output_dim = 2  # Return single complex value per term (real + imaginary parts).

        # Define the hidden layers for the fully connected network.
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim)
                ) for i in range(num_hidden_layers)
            ]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Initialize weights using Xavier uniform for stability in complex-valued neural training.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, grid: Tensor,state:Tensor) -> Tensor:
        """
        Forward computation of the Transfer Function Neural Network (H(s) approximation).

        Args:
            grid (Tensor): Complex input grid. Shape:
                              - [batch, time_dim, terms_dim, 2] (real and imaginary components).

        Returns:
            Tensor: Output tensor representing the approximated complex-valued H(s).
        """
        assert len(grid.shape) == 4, "Expecting a 2D grid in dims [ batch, time, ILT terms, 2]."
        batch_dim, time_dim, ilt_terms_dim,_ = grid.shape
        x=grid

        if self.process_all_terms_as_features:  # Multi-term mode.
            x = x.view(batch_dim, time_dim, -1)  # Reshape ILT terms across features for joint processing.

        #concatenate the state
        state=state.unsqueeze(1).expand(-1,time_dim,-1)
        x=torch.cat([state,x],dim=-1)

        # Sequentially apply layers to propagate input through the network.
        for layer in self.hidden_layers:
            x = self.activation_fn(layer(x))
        x = self.output_layer(x)

        real_part, imag_part = torch.chunk(x, 2, dim=-1)
        s_complex = torch.view_as_complex(torch.stack((real_part, imag_part), dim=-1))  # Aggregate as complex output.


        return s_complex.view(batch_dim, time_dim, ilt_terms_dim, self.dim_response, self.dim_latent_state)  # Final shape.