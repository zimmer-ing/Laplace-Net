# MIT License, Copyright (c) 2025 Bernd Zimmering
# See LICENSE file for details.
# Please cite: Zimmering, B. et al. (2025), "Breaking Free: Decoupling Forced Systems with Laplace Neural Networks", ECML PKDD, Porto.

import Constants as const

from src.models.models_base import BaseRegressionModel
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

# Import sub elements of the model
from src.models.Laplace_net_utils.InverseLaplaceTransform import ILTFourier
#from src_dev.inverse_laplace.Cartesian_ILT_Class import ILTFourier
from src.models.Laplace_net_utils.LaplaceTransform import discrete_laplace_transform,laplace_from_fft
from src.models.Laplace_net_utils.NN_Models import  InitialStateNN, TransferFunctionNN


from tqdm import tqdm
from typing import Optional
from src.utils.helpers import ensure_sequential_dataloader, concatenate_batches,dl_is_shuffle,dataloader_hash



class LaplaceNet(BaseRegressionModel):
    """
    A modular neural Laplace model (flat configuration version).

    This class implements a Laplace-domain system identification approach using a flat
    configuration structure. It performs the following steps:
      1. Computes a base discrete Laplace transform of the forecast excitation signal.
      2. Computes a learned correction term via an input excitation network.
         The input excitation network processes the concatenation of the forecast excitation
         and its time vector. Its parameters are provided via flat keys.
      3. Evaluates the transfer function H(s) over a generated time grid.
      4. Computes an initial state correction P(s) from historical response data.
      5. Computes Y(s) = H(s) · (X(s) + P(s)) and reconstructs the time-domain output via an ILT.

    Expected flat configuration keys:

      dataset:
        - "dim_x": Input (excitation) dimension.
        - "dim_y": Output (response) dimension.

      model: (flat structure)
        - "ilt_terms": Number of ILT support points.
        - "ilt_alpha": Regularization parameter for ILTFourier.
        - "ilt_scale": Scaling factor for ILTFourier.
        - "ilt_shift": Time (contour) shift for the ILT.
        - "use_laplace_scaling": Boolean indicating whether to apply Laplace scaling.
        - "grid_limits": List specifying the limits for the generated time grid.
        - "tf_hidden_dim": Hidden dimension for the transfer function network.
        - "tf_num_hidden_layers": Number of hidden layers in the transfer function network.
        - "tf_activation": Activation function for the transfer function network.
        - "tf_process_all_terms": Boolean flag for processing all ILT terms as features.
        - "is_dim_history": Dimension of the historical input for the initial state network.
        - "is_order_dim": Output (order) dimension for the initial state network.
        - "is_RNN_hidden_dim": Hidden dimension for the RNN inside the initial state network.
        - "is_RNN_num_layers": Number of RNN layers in the initial state network.
        - "learning_rate": Learning rate for training (required).

      Global:
        - "chosen_device": Device string (e.g., "cpu" or "cuda").
    """

    def __init__(self, config):
        super(LaplaceNet, self).__init__()

        # Validate required configuration parameters
        assert config['dataset']['dim_x'] is not None, "dim_x must be provided in config"
        assert config['dataset']['dim_y'] is not None, "dim_y must be provided in config"
        assert config['chosen_device'] is not None, "'chosen_device' must be provided in config"
        for key in ["ilt_terms", "ilt_alpha", "ilt_scale", "ilt_shift",
                    "tf_hidden_dim", "tf_num_hidden_layers", "tf_activation",
                     "is_order_dim", "is_RNN_hidden_dim", "is_RNN_num_layers",
                    "learning_rate","laplace_scaling_factor","steps"]:
            assert key in config["model"], f"'{key}' must be provided in model config"
        assert config["model"]["LP_trafo_type"] in ["FFLT", "DLT"], "LP_trafo_type must be either 'FFLT' or 'DLT'"
        # Set basic parameters from the dataset configuration
        self.input_dim = config['dataset']["dim_x"]
        self.output_dim = config['dataset']["dim_y"]
        self.device = torch.device(config["chosen_device"])

        # ILT parameters (from flat model config)
        config_model = config["model"]
        self.ilt_terms = config_model["ilt_terms"]
        self.ilt_alpha = config_model["ilt_alpha"]
        self.ilt_scale = config_model["ilt_scale"]
        self.ilt_shift = config_model["ilt_shift"]
        self.LP_trafo_type = config_model["LP_trafo_type"]
        self.use_laplace_scaling = config_model.get("use_laplace_scaling", True)
        self.grid_limits = config_model.get("grid_limits", [-1, 1])
        self.laplace_scaling_factor=config_model["laplace_scaling_factor"]
        self.steps=config_model["steps"]

        self.float_dtype = torch.float32
        self.complex_dtype = torch.complex64
        self.dl_hash_train = -1
        self.dl_hash_val = -1

        # Learning rate is required
        self.learning_rate = config_model["learning_rate"]



        # Initialize the ILT object (for inverse Laplace transformation)
        self.ILT = ILTFourier(
            ilt_reconstruction_terms=self.ilt_terms,
            alpha=self.ilt_alpha,
            scale=self.ilt_scale,
            shift=self.ilt_shift,
            device=self.device
        )

        #network_dims
        self.dim_n=self.input_dim
        self.dim_m=self.output_dim
        self.B = torch.nn.Parameter(torch.randn(self.dim_m ,self.dim_n), requires_grad=True)

        # Initialize the Transfer Function network: computes H(s)
        self.H_s_net = TransferFunctionNN(
            dim_terms_ilt=self.ilt_terms,
            dim_m=self.dim_m,
            dim_n=self.dim_m,
            hidden_dim=config_model["tf_hidden_dim"],
            num_hidden_layers=config_model["tf_num_hidden_layers"],
            state_dim=config_model["is_RNN_hidden_dim"],
            activation_fn=getattr(F, config_model.get("tf_activation", "tanh")),
            process_all_terms_as_features=config_model.get("tf_process_all_terms", True)
        )

        # Initialize the Initial State network: computes P(s) from historical response data
        self.P_s_net = InitialStateNN(
            dim_m=self.dim_m,
            dim_history=self.output_dim,
            order_dim=config_model["is_order_dim"],
            RNN_hidden_dim=config_model["is_RNN_hidden_dim"],
            RNN_num_layers=config_model["is_RNN_num_layers"]
        )



        self.to(self.device)
        self.prepare_training(config)

    def forward(self,
                excitation_history: torch.Tensor,
                response_history: torch.Tensor,
                time_history: torch.Tensor,
                excitation_forecast: torch.Tensor,
                time_forecast: torch.Tensor,
                X_s=None) -> torch.Tensor:
        """
        Forward pass for the Modular Neural Laplace model.

        Args:
            excitation_history (torch.Tensor): Historical excitation signal (batch, hist_seq_len, input_dim).
            response_history (torch.Tensor): Historical response signal (batch, hist_seq_len, output_dim).
            time_history (torch.Tensor): Time stamps for the historical signal (batch, hist_seq_len, 1).
            excitation_forecast (torch.Tensor): Forecast excitation signal (batch, forecast_seq_len, input_dim).
            time_forecast (torch.Tensor): Time stamps for the forecast period (batch, forecast_seq_len, 1).

        Returns:
            torch.Tensor: Reconstructed forecast response signal (batch, forecast_seq_len, output_dim).
        """
        batch_size, forecast_len, _ = excitation_forecast.shape
        stepsize = np.ceil(forecast_len // self.steps)

        # 1. Compute the base discrete Laplace transform (DLT) for the forecast excitation.


        if X_s is None:
            X_s=self.calculate_X_S_steps(excitation_forecast,time_forecast)
                # X_s shape: (batch, forecast_len, ilt_terms)




        response_history_iter=response_history
        output=torch.zeros(batch_size, forecast_len, self.output_dim).to(self.device)
        for i in range(self.steps):
            start = i * stepsize
            end = (i + 1) * stepsize
            if end > forecast_len:
                end = forecast_len

            # 2. Generate a time grid for evaluating H(s)
            grid = self.ILT.generate_time_grid(end-start, self.ilt_terms, self.grid_limits)
            grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # shape: (batch, forecast_len, ilt_terms, 1)

            time_forecast_step = time_forecast[:, start:end, :]
            min_dt = time_forecast_step.diff(1, dim=1).min(dim=1).values.unsqueeze(-1)
            time_forecast_step = time_forecast_step - time_forecast_step[:, 0, :].unsqueeze(1) + min_dt
            act_X_s=X_s[i].to(self.device)
            query_s_forecast_step = self.ILT.get_query_points(time_forecast_step)
            P_s,state = self.P_s_net(query_s_forecast_step, response_history_iter)
            # 3. Compute H(s) using the transfer function network.
            H_s = self.H_s_net(grid,state)
            if self.use_laplace_scaling:
                H_s = self.ILT.inverse_transform_LP_func(H_s, time_forecast_step)/self.laplace_scaling_factor
            # 5. Combine X(s) and P(s)
            B_complex = self.B.to(self.complex_dtype)
            B_Sum_P_X = torch.matmul(B_complex, act_X_s.unsqueeze(-1)) + P_s.to(act_X_s.dtype).unsqueeze(-1)
            # 6. Compute Y(s) = H(s) * (BX(s) + P(s)) via matrix multiplication.
            Y_s = torch.matmul(H_s, B_Sum_P_X).squeeze(-1)
            # Y_s shape: (batch, forecast_len, ilt_terms)

            # 7. Reconstruct the time-domain forecast response from Y(s)
            response_history_iter= self.ILT.reconstruct(Y_s, time_forecast_step)
            output[:, start:end, :] = response_history_iter

        # output shape: (batch, forecast_len, output_dim)
        return output

    def check_input_transform(self,excitation_forecast,time_forecast):
        batch_size, forecast_len, _ = excitation_forecast.shape
        stepsize = np.ceil(forecast_len // self.steps)
        x_t_dlt=torch.zeros(batch_size, forecast_len, self.input_dim).to(self.device)
        x_t_fft=torch.zeros(batch_size, forecast_len, self.input_dim).to(self.device)
        import time
        start_time = time.time()
        for i in range(self.steps):
            start = i * stepsize
            end = (i + 1) * stepsize
            if end > forecast_len:
                end = forecast_len
            time_forecast_step = time_forecast[:, start:end, :]
            # correct offset
            min_dt = time_forecast_step.diff(1, dim=1).min(dim=1).values.unsqueeze(-1)
            time_forecast_step = time_forecast_step - time_forecast_step[:, 0, :].unsqueeze(1) + min_dt
            query_s_forecast_step = self.ILT.get_query_points(time_forecast_step)
            excitation_forecas_step = excitation_forecast[:, start:end, :]
            X_s_step = discrete_laplace_transform(excitation_forecas_step,
                                                  query_s_forecast_step,
                                                  time_forecast_step + self.ilt_shift
                                                  # contour shift needs to be added as ILT also shifts the time
                                                  ).unsqueeze(-1).squeeze(2)
            x_t_step = self.ILT.reconstruct(X_s_step, time_forecast_step)
            x_t_dlt[:, start:end, :] = x_t_step
        print("Time for DLT: ", time.time() - start_time)

        #same for fft
        start_time = time.time()
        for i in range(self.steps):
            start = i * stepsize
            end = (i + 1) * stepsize
            if end > forecast_len:
                end = forecast_len
            time_forecast_step = time_forecast[:, start:end, :]
            # correct offset
            min_dt = time_forecast_step.diff(1, dim=1).min(dim=1).values.unsqueeze(-1)
            time_forecast_step = time_forecast_step - time_forecast_step[:, 0, :].unsqueeze(1) + min_dt
            query_s_forecast_step = self.ILT.get_query_points(time_forecast_step)
            excitation_forecas_step = excitation_forecast[:, start:end, :]
            X_s_step = laplace_from_fft(excitation_forecas_step,
                                        query_s_forecast_step,
                                        time_forecast_step + self.ilt_shift
                                        # contour shift needs to be added as ILT also shifts the time
                                        ).unsqueeze(-1).squeeze(2)
            x_t_step = self.ILT.reconstruct(X_s_step, time_forecast_step)
            x_t_fft[:, start:end, :] = x_t_step
        print("Time for FFT: ", time.time() - start_time)

        return x_t_dlt,x_t_fft

    def calculate_X_S_steps(self, excitation_forecast, time_forecast):
        batch_size, forecast_len, _ = excitation_forecast.shape
        stepsize = np.ceil(forecast_len // self.steps)
        X_s = []
        for i in range(self.steps):
            start = i * stepsize
            end = (i + 1) * stepsize
            if end > forecast_len:
                end = forecast_len
            time_forecast_step = time_forecast[:, start:end, :]
            # correct offset
            min_dt = time_forecast_step.diff(1, dim=1).min(dim=1).values.unsqueeze(-1)
            time_forecast_step = time_forecast_step - time_forecast_step[:, 0, :].unsqueeze(1) + min_dt

            excitation_forecast_step = excitation_forecast[:, start:end, :]
            X_s_step = self.calculate_X_S(excitation_forecast_step, time_forecast_step, LT_type=self.LP_trafo_type,
                                          low_prec=True)
            X_s.append(X_s_step)
        return X_s

    def calculate_X_S(self, excitation_forecast, time_forecast, low_prec=True, LT_type="FFLT"):
        """
        Calculate the Laplace transformed input signal X(s) for the given data.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the input data.
            low_prec (bool, optional): If True, use lower precision for the calculations. Defaults to True.
            LT_type (str, optional): Type of Laplace transform to use. Defaults to "FFLT".

        Returns:
            torch.Tensor: Laplace transformed input signal X(s) for the data.
        """
        assert LT_type in ["FFLT", "DLT"], "LT_type must be either 'FFLT' or 'DLT'"
        if low_prec:
            dtype_float = torch.float32
            dtype_complex = torch.complex64
        else:
            dtype_float = torch.float64
            dtype_complex = torch.complex128

        with torch.no_grad():
            excitation_forecast = excitation_forecast.to(dtype_float)
            time_forecast = time_forecast.to(dtype_float)
            query_s_forecast = self.ILT.get_query_points(time_forecast).to(dtype_complex)
            if LT_type == "FFLT":
                X_s = laplace_from_fft(excitation_forecast,
                                       query_s_forecast,
                                       time_forecast + self.ilt_shift
                                       # contour shift needs to be added as ILT also shifts the time
                                       )
            elif LT_type == "DLT":
                X_s = discrete_laplace_transform(excitation_forecast,
                                                 query_s_forecast,
                                                 time_forecast + self.ilt_shift
                                                 # contour shift needs to be added as ILT also shifts the time
                                                 )
            else:
                raise ValueError("Unknown LT type")

        return X_s.to(self.complex_dtype)

    def train_step(self, data_loader: torch.utils.data.DataLoader, X_S=None, return_loss: bool = False) -> Optional[
        float]:
        """
        Train the model for one epoch on the given data loader.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader with training data.
            return_loss (bool, optional): If True, return the mean loss. Defaults to False.

        Returns:
            Optional[float]: Mean loss if return_loss is True, otherwise None.
        """
        losses = []
        self.train()
        # check if dataloader is in shuffle mode
        if X_S is None:
            if not dl_is_shuffle(data_loader):
                hash = dataloader_hash(data_loader)
                if self.dl_hash_train != hash:
                    self.stored_X_S_train = []
                    # make sure dataloader did not change by hashing from the dataloader
                    # we can precompute the X_S values and store them as they will be the same for each batch during training
                    for batch in data_loader:
                        excitation_forecast = batch['excitation_forecast']
                        time_forecast = batch['time_forecast']
                        X_S_batch = self.calculate_X_S_steps(excitation_forecast, time_forecast)
                        self.stored_X_S_train.append(X_S_batch)
                        X_S = self.stored_X_S_train
                    self.dl_hash_train = hash
                else:
                    X_S = self.stored_X_S_train

            else:
                X_S = [None] * len(data_loader)
        for batch, X_S_batch in zip(data_loader, X_S):
            excitation_history = batch['excitation_history']
            response_history = batch['response_history']
            excitation_forecast = batch['excitation_forecast']
            response_forecast = batch['response_forecast']
            time_history = batch['time_history']
            time_forecast = batch['time_forecast']
            self.optimizer.zero_grad()
            predictions = self(excitation_history, response_history, time_history, excitation_forecast, time_forecast,
                               X_S_batch, )
            loss = self.calculate_loss(predictions, response_forecast.to(self.float_dtype))
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            if return_loss:
                losses.append(loss.item())
            self.optimizer.step()
        if return_loss:
            return sum(losses) / len(losses)

    def validate_step(self, data_loader: torch.utils.data.DataLoader, X_S=None) -> float:
        """
        Validate the model on the given data loader.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader with validation data.

        Returns:
            float: Mean loss over the validation set.
        """
        losses = []
        self.eval()
        if X_S is None:
            if not dl_is_shuffle(data_loader):
                hash = dataloader_hash(data_loader)
                if self.dl_hash_val != hash:
                    self.stored_X_S_val = []
                    # make sure dataloader did not change by hashing from the dataloader
                    # we can precompute the X_S values and store them as they will be the same for each batch during validation
                    for batch in data_loader:
                        excitation_forecast = batch['excitation_forecast']
                        time_forecast = batch['time_forecast']
                        X_S_batch = self.calculate_X_S_steps(excitation_forecast, time_forecast)
                        self.stored_X_S_val.append(X_S_batch)
                        X_S = self.stored_X_S_val
                    self.dl_hash_val = hash
                else:
                    X_S = self.stored_X_S_val

            else:
                X_S = [None] * len(data_loader)
        with torch.no_grad():
            for batch, X_S_batch in zip(data_loader, X_S):
                excitation_history = batch['excitation_history']
                response_history = batch['response_history']
                excitation_forecast = batch['excitation_forecast']
                response_forecast = batch['response_forecast']
                time_history = batch['time_history']
                time_forecast = batch['time_forecast']
                predictions = self(excitation_history, response_history, time_history, excitation_forecast,
                                   time_forecast, X_S_batch)
                loss = self.calculate_loss(predictions, response_forecast)
                losses.append(loss.item())
            if len(losses) == 0:
                raise ValueError("Validation set is empty.")
            return sum(losses) / len(losses)

    def predict(self, data_loader: torch.utils.data.DataLoader, X_S=None, samples_only: bool = False) -> dict:
        """
        Predict on the given data loader in a fully generic way.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader with test data.
            samples_only (bool, optional): If True, return only the predictions and specified subsets. Defaults to False.

        Returns:
            dict: A dictionary containing all batch data, model predictions, and latent variables (if applicable).
        """
        # Initialize an empty dictionary to store each type of data encountered in the batch
        results = {}

        data_loader = ensure_sequential_dataloader(data_loader)
        self.eval()
        if X_S is None:
            X_S = [None] * len(data_loader)
        with torch.no_grad():
            for batch, X_S_batch in zip(data_loader, X_S):
                # Dynamically populate results dictionary with data from the first batch
                if not results:
                    results = {key: [] for key in batch.keys()}
                    results[
                        "predictions"] = []  # Add a key for predictions
                    # results["z_values"] = []  # Add a key for latent variables if used in the model

                # Append batch data to results dictionary
                for key, value in batch.items():
                    results[key].append(value)

                    # Run the model's prediction
                y_hat = self(
                    batch.get("excitation_history"),
                    batch.get("response_history"),
                    batch.get("time_history"),
                    batch.get("excitation_forecast"),
                    batch.get("time_forecast")
                    , X_S_batch
                )
                results["predictions"].append(y_hat)
                # results["z_values"].append(z_batch)

        # If samples_only, return only the predictions and a subset of relevant data
        if samples_only:
            return results

            # Concatenate batches for each entry in the results dictionary
        for key in results.keys():
            results[key] = concatenate_batches(results[key])

        return results

