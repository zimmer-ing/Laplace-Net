import torch
import torch.nn as nn
import numpy as np
from src.models.models_base import BaseRegressionModel

#####################################################################################################
# Neural Laplace Operator by Qianying Cao, Somdatta Goswami, George Em Karniadakis
#   @article{cao2023lno,
#   title={Lno: Laplace neural operator for solving differential equations},
#   author={Cao, Qianying and Goswami, Somdatta and Karniadakis, George Em},
#   journal={arXiv preprint arXiv:2303.10528},
#   year={2023}
#   }
# This part is taken from the Laplace-Neural-Operator-main/1D_Duffing_c0/main.py file
# https://github.com/qianyingcao/Laplace-Neural-Operator/blob/0f3d057821bc88aede69bda6c4f387fdb8239cc3/1D_Duffing_c0/main.py
# Auxiliary variables are edited to make the code compatible with the project structure
# ====================================
#  Laplace layer: pole-residue operation is used to calculate the poles and residues of the output
# ====================================
class PR(nn.Module):
    def __init__(self, in_channels, out_channels, modes1,device):
        super(PR, self).__init__()

        self.modes1 = modes1
        self.scale = (1 / (in_channels *out_channels))
        self.weights_pole = nn.Parameter \
            (self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)).to(device)
        self.weights_residue = nn.Parameter \
            (self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)).to(device)
        self.device=device
    def output_PR(self, lambda1 ,alpha, weights_pole, weights_residue):
        Hw =torch.zeros(weights_residue.shape[0] ,weights_residue.shape[0] ,weights_residue.shape[2] ,lambda1.shape[0], device=alpha.device, dtype=torch.cfloat)
        term1 =torch.div(1 ,torch.sub(lambda1 ,weights_pole))
        Hw =weights_residue *term1
        output_residue1 =torch.einsum("bix,xiok->box", alpha, Hw)
        output_residue2 =torch.einsum("bix,xiok->bok", alpha, -Hw)
        return output_residue1 ,output_residue2

    def forward(self, x,grid):
        t=grid.to(self.device)
        # Compute input poles and resudes by FFT
        dt=(t[1]-t[ 0]).item()
        alpha = torch.fft.fft(x)
        lambda0=torch.fft.fftfreq(t.shape[0], dt)*2*np.pi * 1j
        lambda1=lambda0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1=lambda1.to(self.device)

        # Obtain output poles and residues for transient part and steady-state part
        output_residue1,output_residue2= self.output_PR(lambda1, alpha, self.weights_pole, self.weights_residue)

        # Obtain time histories of transient response and steady-state response
        x1 = torch.fft.ifft(output_residue1, n=x.size(-1))
        x1 = torch.real(x1)
        x2=torch.zeros(output_residue2.shape[0],output_residue2.shape[1],t.shape[ 0], device=alpha.device, dtype=torch.cfloat)
        term1=torch.einsum("bix,kz->bixz", self.weights_pole, t.type(torch.complex64).reshape(1,-1))
        term2=torch.exp(term1)
        x2=torch.einsum("bix,ioxz->boz", output_residue2,term2)
        x2=torch.real(x2)
        x2=x2/x.size( - 1)
        return x1+x2

class LNO1d(nn.Module):
    def __init__(self, width,modes,device,activation_fn="sin"):
        super(LNO1d, self).__init__()

        self.width = width
        self.modes1 = modes
        self.fc0 = nn.Linear(1, self.width,).to(device)

        self.conv0 = PR(self.width, self.width, self.modes1,device)
        self.w0 = nn.Conv1d(self.width, self.width, 1).to(device)

        self.fc1 = nn.Linear(self.width, 128).to(device)
        self.fc2 = nn.Linear(128, 1).to(device)
        if activation_fn == "sin":
            self.activation_fn=torch.sin
        elif activation_fn == "tanh":
            self.activation_fn=torch.nn.functional.tanh
        else:
            raise NotImplementedError

    def forward(self,x,grid):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x,grid)
        x2 = self.w0(x)
        x = x1 +x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


#####################################################################################################
class LaplaceNeuralOperator(BaseRegressionModel):
    """

    """

    def __init__(self, config):
        """
        Initializes the LSTMAutoEncoder with the given configuration.

        Args:
            config (dict): Configuration dictionary with model and data parameters.
        """
        super(LaplaceNeuralOperator, self).__init__()

        # Extract and validate configuration parameters
        assert config['dataset']['dim_x'] is not None, "dim_x must be provided in config"
        assert config['dataset']['dim_y'] is not None, "dim_y must be provided in config"
        assert config['model']['width'] is not None, "width must be provided in config"
        assert config['model']['modes'] is not None, "modes must be provided in config"
        assert config['model']['activation_fn'] in ["sin", "tanh"], "activation_fn must be 'sin' or 'tanh'"
        assert config['chosen_device'] is not None, "'chosen_device' must be provided in config"

        # Set parameters based on configuration
        self.input_dim = config['dataset']['dim_x']
        self.output_dim = config['dataset']['dim_y']
        self.device = torch.device(config['chosen_device'])
        self.width = config['model']['width']
        self.modes = config['model']['modes']
        self.activation_fn = config['model']['activation_fn']

        self.LNO=LNO1d(self.width,self.modes,device=self.device,activation_fn=self.activation_fn)

        # Prepare for training by calling the parent class method
        self.prepare_training(config)

    def forward(self,excitation_history: torch.Tensor,
                response_history: torch.Tensor,
                time_history: torch.Tensor,
                excitation_forecast: torch.Tensor,
                time_forecast: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM Autoencoder.

        Args:
            excitation_history (torch.Tensor): Excitation history tensor.
            response_history (torch.Tensor): Response history tensor.
            time_history (torch.Tensor): Time history tensor.
            excitation_forecast (torch.Tensor): Excitation forecast tensor.
            time_forecast (torch.Tensor): Time forecast tensor.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, output_dim).
        """
        # Initialize the output tensor with the same shape as the input sequence
        batch_size, hist_seq_len, _ = excitation_history.shape
        _, fore_seq_len, _ = excitation_forecast.shape


        excitation_total=torch.cat([excitation_history,excitation_forecast],dim=1)
        time_total=torch.cat([time_history,time_forecast],dim=1)[0]
        response_total=self.LNO(excitation_total,grid=time_total)
        response_forecast=response_total[:,hist_seq_len:,:].reshape(batch_size,fore_seq_len,self.output_dim)

        return response_forecast

