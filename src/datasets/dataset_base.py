# MIT License, Copyright (c) 2025 Bernd Zimmering
# See LICENSE file for details.
# Please cite: Zimmering, B. et al. (2025), "Breaking Free: Decoupling Forced Systems with Laplace Neural Networks", ECML PKDD, Porto.
import Constants as const
import json
from typing import List, Dict, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings


class MultiFileTimeSeriesDataset(Dataset):
    """
    Dataset class for managing multiple files with consistent scaling for training, validation, or testing.
    """

    def __init__(self, directory: str, files: List[str], config: dict, device: Optional[str] = 'cpu'):
        """
        Initializes the dataset for a specific data split (train, val, or test) and loads scalers if available.

        Args:
            directory (str): Directory containing the dataset files.
            files (List[str]): List of file names.
            config (dict): Configuration dictionary for dataset parameters.
            device (str, optional): Device to store data on ('cpu' or 'cuda').
        """
        self.directory = Path(directory)
        self.config = config
        self.device = device
        self.files = files
        self.scalers = self.load_scalers(self.directory)
        self.timeseries_datasets = self.load_datasets()

        # Calculate cumulative lengths for quick indexing
        self.len_subsets = [len(dataset) for dataset in self.timeseries_datasets]
        self.cumulative_len_subsets = np.cumsum(self.len_subsets)

    def load_scalers(self, scaler_path: Path) -> Dict[str, any]:
        """
        Loads saved scaler parameters from a single JSON file for both 'excitation' and 'response' channels,
        using channel names from the `get_channels` method, and assigns only the relevant channels to each scaler.
        """
        scalers = {}
        try:
            # Load the JSON file containing all scaling parameters
            with open(scaler_path / 'scaler_params.json', 'r') as f:
                scaler_params = json.load(f)

            # Determine the scaler type (StandardScaler or MinMaxScaler)
            scaler_type = scaler_params.get('scaler_type', 'standard')

            # Retrieve channels for excitation and response
            channels = self.get_channels()
            excitation_channels = channels['excitation']
            response_channels = channels['response']

            # Helper function to set scaler parameters for specific channels
            def configure_scaler_for_channels(scaler, channels, params_by_channel):
                # Extract and set the specific parameters for each channel
                scales = []
                mins = []
                means = []
                variances = []
                for channel in channels:
                    channel_params = params_by_channel.get(channel, {})
                    scales.append(channel_params.get('scale_', 1.0))  # Default scale factor
                    mins.append(channel_params.get('min_', 0.0) if scaler_type == 'minmax' else None)
                    means.append(channel_params.get('mean_', 0.0) if scaler_type == 'standard' else None)
                    variances.append(channel_params.get('var_', 1.0) if scaler_type == 'standard' else None)

                # Assign the attributes as arrays for the scaler
                scaler.scale_ = np.array(scales)
                if scaler_type == 'minmax':
                    scaler.min_ = np.array(mins)
                elif scaler_type == 'standard':
                    scaler.mean_ = np.array(means)
                    scaler.var_ = np.array(variances)
                return scaler

            # Create and configure the scaler for 'excitation' channels
            excitation_scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
            excitation_scaler = configure_scaler_for_channels(
                excitation_scaler, excitation_channels, scaler_params['values_by_channel']
            )

            # Create and configure the scaler for 'response' channels
            response_scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
            response_scaler = configure_scaler_for_channels(
                response_scaler, response_channels, scaler_params['values_by_channel']
            )

            # Assign the configured scalers to the corresponding keys
            scalers['excitation'] = excitation_scaler
            scalers['response'] = response_scaler

        except FileNotFoundError:
            if const.DEBUG:
                print(f"Scaler parameters file 'scaler_params.json' not found in {scaler_path}. Skipping as no scalers are available.")
        except Exception as e:
            print(f"Error loading scalers: {e}")

        return scalers


    def load_datasets(self) -> List[Dataset]:
        """
        Loads individual datasets from files in the specified directory.
        """
        datasets = []
        for file_name in self.files:
            data = self.load_file(self.directory / file_name)
            dataset = TimeSeriesForecastingDataset(data, self.config, self.get_channels(), device=self.device)
            datasets.append(dataset)
        return datasets

    def __len__(self):
        """Returns the total number of samples across all datasets."""
        return sum(self.len_subsets)

    def __getitem__(self, index: int):
        """
        Retrieves a sample based on the global index by locating the correct subset and index within it.
        """
        dataset_index = np.argmax(self.cumulative_len_subsets > index)
        if dataset_index == 0:
            sample_index = index
        else:
            sample_index = index - self.cumulative_len_subsets[dataset_index - 1]
        return self.timeseries_datasets[dataset_index][sample_index]

    def inverse_transform(self, excitation, response, return_as_dataframe: bool = True):
        """
        Applies inverse scaling to the provided excitation and response data, restoring them to their original scales.

        Args:
            excitation: Excitation data to be inverse-transformed. Can be a PyTorch tensor, NumPy array, or pandas DataFrame.
            response: Response data to be inverse-transformed. Can be a PyTorch tensor, NumPy array, or pandas DataFrame.
            return_as_dataframe (bool): If True, returns the result as a pandas DataFrame; otherwise, returns as a NumPy array.

        Returns:
            Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray]]: Inverse-transformed excitation and response data.
        """
        # Convert excitation and response to NumPy arrays if they are PyTorch tensors
        if isinstance(excitation, torch.Tensor):
            excitation = excitation.cpu().numpy()
        if isinstance(response, torch.Tensor):
            response = response.cpu().numpy()

        # Apply inverse scaling to excitation channels if a scaler is available
        if 'excitation' in self.scalers:
            excitation = self.scalers['excitation'].inverse_transform(excitation)

        # Apply inverse scaling to response channels if a scaler is available
        if 'response' in self.scalers:
            response = self.scalers['response'].inverse_transform(response)

        # Convert back to pandas DataFrame if requested
        if return_as_dataframe:
            excitation_columns = self.get_channels()['excitation']
            response_columns = self.get_channels()['response']
            excitation = pd.DataFrame(excitation, columns=excitation_columns)
            response = pd.DataFrame(response, columns=response_columns)


        return excitation, response

    def get_channels(self) -> Dict[str, List[str]]:
        """
        Returns the channel names for 'time', 'excitation', 'response', and auxiliary data.
        """
        # Define channels here
        return {
            "time": ["time"],
            "excitation": ["excitation"],
            "response": ["response"],
            "auxiliary": ["aux1", "aux2"]  # Adjust as needed
        }

    def load_file(self, file_path: Path) -> pd.DataFrame:
        """
        Loads a single file. This method can be customized or subclassed as needed.
        """
        return pd.read_csv(file_path)  # Adjust file loading as necessary

    @staticmethod
    def collate_fn(batch):
        """
        Collates a batch of data for DataLoader, stacking historical and forecasted sequences.
        """
        time_hist, time_fore = zip(*[(item["time_history"], item["time_forecast"]) for item in batch])
        excitation_hist, excitation_fore = zip(
            *[(item["excitation_history"], item["excitation_forecast"]) for item in batch])
        response_hist, response_fore = zip(*[(item["response_history"], item["response_forecast"]) for item in batch])
        auxiliary_data = [item["auxiliary"] for item in batch]

        return {
            "time_history": torch.stack(time_hist),
            "time_forecast": torch.stack(time_fore),
            "excitation_history": torch.stack(excitation_hist),
            "excitation_forecast": torch.stack(excitation_fore),
            "response_history": torch.stack(response_hist),
            "response_forecast": torch.stack(response_fore),
            "auxiliary": auxiliary_data
        }

class TimeSeriesForecastingDataset(Dataset):
    """
    Dataset class for creating historical and forecast data subsequences with auxiliary channels.

    Attributes:
        data_sequences (list): List of tuples containing historical and forecast subsequences for main channels
                               and auxiliary data.
    """

    def __init__(self, data: pd.DataFrame, config: dict, channels: Dict[str, List[str]], device: Optional[str] = 'cpu'):
        """
        Initializes the dataset for time series forecasting.

        Args:
            data (pd.DataFrame): Input data for time series.
            config (dict): Configuration for subsequence generation.
            channels (Dict[str, List[str]]): Channel names for time, excitation, response, and auxiliary channels.
            device (str, optional): Device for storing tensors ('cpu' or 'cuda').
        """
        self.config = config
        self.channels = channels
        self.device = device
        self.data_sequences = self.create_subsequences(data, config, channels, device=self.device)

    def __len__(self) -> int:
        """Returns the number of subsequences in the dataset."""
        return len(self.data_sequences)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a specific subsequence by index.

        Args:
            index (int): Index of the subsequence.

        Returns:
            dict: Dictionary with tensors for historical and forecasted time, excitation, response, and auxiliary data.
        """
        time_hist, time_fore, excitation_hist, excitation_fore, response_hist, response_fore, aux_data = self.data_sequences[index]
        return {
            "time_history": time_hist,
            "time_forecast": time_fore,
            "excitation_history": excitation_hist,
            "excitation_forecast": excitation_fore,
            "response_history": response_hist,
            "response_forecast": response_fore,
            "auxiliary": aux_data  # Auxiliary data for analysis
        }

    @staticmethod
    def collate_fn(batch):
        """
        Collates a batch of data for DataLoader.

        This function combines individual data points into a batch, stacking tensors for historical and forecasted
        sequences across time, excitation, response, and auxiliary data.

        Args:
            batch (list): A list of dictionaries containing historical and forecasted sequences.

        Returns:
            dict: A dictionary containing batched tensors for each sequence type:
                - "time_history": Tensor of historical time sequences.
                - "time_forecast": Tensor of forecasted time sequences.
                - "excitation_history": Tensor of historical excitation sequences.
                - "excitation_forecast": Tensor of forecasted excitation sequences.
                - "response_history": Tensor of historical response sequences.
                - "response_forecast": Tensor of forecasted response sequences.
                - "auxiliary": Tensor containing additional auxiliary data.
        """
        time_hist, time_fore = zip(*[(item["time_history"], item["time_forecast"]) for item in batch])
        excitation_hist, excitation_fore = zip(
            *[(item["excitation_history"], item["excitation_forecast"]) for item in batch])
        response_hist, response_fore = zip(*[(item["response_history"], item["response_forecast"]) for item in batch])
        auxiliary_data = [item["auxiliary"] for item in batch]

        return {
            "time_history": torch.stack(time_hist),
            "time_forecast": torch.stack(time_fore),
            "excitation_history": torch.stack(excitation_hist),
            "excitation_forecast": torch.stack(excitation_fore),
            "response_history": torch.stack(response_hist),
            "response_forecast": torch.stack(response_fore),
            "auxiliary": auxiliary_data
        }

    @staticmethod
    def create_subsequences(data: pd.DataFrame, config: dict, channels: Dict[str, List[str]], device: Optional[str] = None):
        """
        Creates subsequences from the data according to the specified configuration.

        This function generates historical and forecast subsequences for the time, excitation, and response channels
        based on the specified history and forecast lengths, and includes auxiliary data.

        Args:
            data (pd.DataFrame): Input time series data.
            config (dict): Configuration containing 'stride', 'history_length', and 'forecast_length'.
            channels (Dict[str, List[str]]): Channel names for "time", "excitation", "response", and auxiliary channels.
            device (str, optional): Device to store the tensors.

        Returns:
            list: A list of tuples containing subsequences for time, excitation, response, and auxiliary data.
        """
        stride = config.get('stride', 1)
        if 'stride' not in config:
            warnings.warn("Default value for 'stride' (1) is being used.", UserWarning)

        history_length = config.get('history_length', 10)
        if 'history_length' not in config:
            warnings.warn("Default value for 'history_length' (10) is being used.", UserWarning)

        forecast_length = config.get('forecast_length', 5)
        if 'forecast_length' not in config:
            warnings.warn("Default value for 'forecast_length' (5) is being used.", UserWarning)

        # Extract time, excitation, response, and auxiliary channel data
        time_data = data[channels["time"]].values
        excitation_data = data[channels["excitation"]].values
        response_data = data[channels["response"]].values
        aux_data = data[channels.get("auxiliary", [])].values

        # Create rolling windows for subsequences
        time_windows = np.lib.stride_tricks.sliding_window_view(time_data, window_shape=history_length + forecast_length)[::stride]
        excitation_windows = np.lib.stride_tricks.sliding_window_view(excitation_data, window_shape=(history_length + forecast_length, excitation_data.shape[1]))[::stride].squeeze()
        response_windows = np.lib.stride_tricks.sliding_window_view(response_data, window_shape=(history_length + forecast_length, response_data.shape[1]))[::stride].squeeze()
        aux_windows = np.lib.stride_tricks.sliding_window_view(aux_data, window_shape=(history_length + forecast_length, aux_data.shape[1]))[::stride].squeeze()

        # Separate historical and forecast parts for each channel
        if len(time_data.shape)==1:
            time_hist_seqs = torch.from_numpy(time_windows[:, :history_length].copy()).float().to(device).unsqueeze(-1)
            time_fore_seqs = torch.from_numpy(time_windows[:, history_length:].copy()).float().to(device).unsqueeze(-1)
        else:
            time_hist_seqs = torch.from_numpy(time_windows[:, :history_length].copy()).float().to(device)
            time_fore_seqs = torch.from_numpy(time_windows[:, history_length:].copy()).float().to(device)

        if len(excitation_windows.shape)==1:
            excitation_windows = excitation_windows.reshape(1,excitation_windows.shape[0],1)
        if len(response_windows.shape)==1:
            response_windows = response_windows.reshape(1,response_windows.shape[0],1)
        if len(aux_windows.shape)<3:
            if aux_windows.size==0:
                aux_windows = np.zeros((excitation_windows.shape[0],excitation_windows.shape[1],0))
            else:
                aux_windows = aux_windows.reshape(1,aux_windows.shape[0],1)



        excitation_hist_seqs = torch.from_numpy(excitation_windows[:, :history_length,:].copy()).float().to(device)
        excitation_fore_seqs = torch.from_numpy(excitation_windows[:, history_length:,:].copy()).float().to(device)
        response_hist_seqs = torch.from_numpy(response_windows[:, :history_length,:].copy()).float().to(device)
        response_fore_seqs = torch.from_numpy(response_windows[:, history_length:,:].copy()).float().to(device)
        #only take aux for the forecasted part
        aux_seqs = aux_windows[:, history_length:,:]

        # Return combined data as a list of tuples
        return [
            (time_hist, time_fore, excitation_hist, excitation_fore, response_hist, response_fore, aux)
            for time_hist, time_fore, excitation_hist, excitation_fore, response_hist, response_fore, aux
            in zip(time_hist_seqs, time_fore_seqs, excitation_hist_seqs, excitation_fore_seqs, response_hist_seqs, response_fore_seqs, aux_seqs)
        ]