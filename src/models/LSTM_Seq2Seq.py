# MIT License, Copyright (c) 2025 Bernd Zimmering
# See LICENSE file for details.
# Please cite: Zimmering, B. et al. (2025), "Breaking Free: Decoupling Forced Systems with Laplace Neural Networks", ECML PKDD, Porto.
from .models_base import BaseRegressionModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMSeq2Seq(BaseRegressionModel):
    """
    LSTM Sequence2Sequence Model for Forecasting.


    """

    def __init__(self, config):
        """
        Initializes the LSTM Sequence2Sequence with the given configuration.

        Args:
            config (dict): Configuration dictionary with model and data parameters.
        """
        super(LSTMSeq2Seq, self).__init__()

        # Extract and validate configuration parameters
        assert config['dataset']['dim_x'] is not None, "dim_x must be provided in config"
        assert config['dataset']['dim_y'] is not None, "dim_y must be provided in config"
        assert config['model']['hidden_size'] is not None, "hidden_size must be provided in config"
        assert config['model']['num_layers'] is not None, "num_layers must be provided in config"
        assert config['chosen_device'] is not None, "'chosen_device' must be provided in config"

        # Set parameters based on configuration
        self.input_dim = config['dataset']['dim_x']
        self.output_dim = config['dataset']['dim_y']
        self.hidden_size = config['model']['hidden_size']
        self.num_layers = config['model']['num_layers']
        self.device = torch.device(config['chosen_device'])
        self.dropout = config.get('model', {}).get('dropout', 0.0)
        if config['model']['num_layers'] == 1:
            self.dropout = config['model']['dropout'] = 0.0 # No dropout for single layer LSTM


        # Initialize encoder and decoder
        self.encoder = Encoder(self.num_layers, self.hidden_size, self.input_dim+self.output_dim+1, self.dropout, self.device)
        self.decoder = Decoder(self.num_layers, self.hidden_size, self.input_dim+1,self.output_dim, self.dropout, self.device)

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

        # Calculate time differences, add feature dimension with unsqueeze(-1), and pad with 0 at the start along the time dimension
        dt_hist = F.pad(time_history.diff(dim=1), (0, 0, 1, 0), mode="constant", value=0)
        input_seq = torch.cat((excitation_history, response_history, dt_hist), dim=-1)

        # Pass the input sequence through the encoder to obtain the final hidden states
        hidden_cell = self.encoder(input_seq)


        dt_fore=torch.cat((time_history[:, -1].unsqueeze(1), time_forecast), dim=1).diff(dim=1)
        input_decoder = torch.cat((excitation_forecast, dt_fore), dim=-1)


        # Get the decoder output and the updated hidden states
        output_decoder, hidden_cell = self.decoder(input_decoder, hidden_cell)

        return output_decoder


class Encoder(nn.Module):
    """
    Encoder component of the LSTM Autoencoder.

    The encoder processes the input sequence and returns the final hidden and cell states,
    which represent the compressed representation of the input.
    """

    def __init__(self, num_layers, hidden_size, input_dim, dropout=0, device=torch.device('cpu')):
        """
        Initializes the Encoder.

        Args:
            num_layers (int): Number of LSTM layers.
            hidden_size (int): Number of features in the hidden state.
            input_dim (int): Number of input features.
            dropout (float, optional): Dropout probability. Defaults to 0.
            device (torch.device, optional): Device for computation. Defaults to CPU.
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # LSTM layer for the encoder
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bias=True).to(device)

    def init_hidden(self, batch_size):
        """
        Initializes hidden and cell states for the encoder.

        Args:
            batch_size (int): Batch size.

        Returns:
            tuple: Initial hidden and cell states.
        """
        return (torch.randn(self.num_layers, batch_size, self.hidden_size, device=self.device),
                torch.randn(self.num_layers, batch_size, self.hidden_size, device=self.device))

    def forward(self, input_seq):
        """
        Forward pass through the encoder.

        Args:
            input_seq (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            tuple: Final hidden and cell states of the encoder.
        """
        # Initialize the hidden states
        hidden = self.init_hidden(input_seq.shape[0])

        # Pass the input sequence through the LSTM
        _, hidden_cell = self.lstm(input_seq, hidden)

        return hidden_cell


class Decoder(nn.Module):
    """
    Decoder component of the LSTM Autoencoder.

    The decoder takes the hidden state from the encoder and sequentially generates
    the output sequence by predicting each step conditioned on the previous output.
    """

    def __init__(self, num_layers, hidden_size,input_dim, output_dim, dropout=0, device=torch.device('cpu')):
        """
        Initializes the Decoder.

        Args:
            num_layers (int): Number of LSTM layers.
            hidden_size (int): Number of features in the hidden state.
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            dropout (float, optional): Dropout probability. Defaults to 0.
            device (torch.device, optional): Device for computation. Defaults to CPU.
        """
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # LSTM layer for the decoder
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bias=True).to(device)

        # Linear layer to map the output of the LSTM to the target feature space
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_dim).to(device)

    def forward(self, input_seq, hidden_cell):
        """
        Forward pass through the decoder.

        Args:
            input_seq (torch.Tensor): Input tensor of shape (batch_size, 1, output_dim).
            hidden_cell (tuple): Hidden and cell states from the encoder.

        Returns:
            tuple: Decoded output tensor of shape (batch_size, 1, output_dim) and updated hidden states.
        """
        # Pass the input and hidden state through the LSTM
        output, hidden_cell = self.lstm(input_seq, hidden_cell)

        # Map the output from hidden_size to output_dim using the linear layer
        output = self.linear(output)

        return output, hidden_cell