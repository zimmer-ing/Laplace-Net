# MIT License, Copyright (c) 2025 Bernd Zimmering
# See LICENSE file for details.
# Please cite: Zimmering, B. et al. (2025), "Breaking Free: Decoupling Forced Systems with Laplace Neural Networks", ECML PKDD, Porto.
from pathlib import Path
from src.datasets.dataset_base import MultiFileTimeSeriesDataset
from typing import Dict, Optional
import Constants as CONST
import torch
from scipy.integrate import odeint
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            # self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

class LNO_Data_base(MultiFileTimeSeriesDataset):
    """
    Dataset class for the 1D Duffing equation with c0=0.

    This class loads data from predefined files, splits it into train, validation, and test sets,
    and organizes it according to the channels for time, excitation, response, and auxiliary data.
    """



    def __init__(self, config: dict, data_split:str='train',device: Optional[str] = 'cpu',path_ds: Optional[str] = '1D_Duffing_c0'):
        """
        Initializes the LeakageAnomalyDataset with specified configuration.

        Args:
            config (dict): Configuration dictionary for dataset parameters.
            scaler_type (str): Type of scaler ('standard' or 'minmax').
            device (Optional[str]): Device for storing tensors ('cpu' or 'cuda').
        """

        path_data = Path(CONST.DATA_PATH, "LNO_Paper", path_ds)


        # check if the data is already generated
        if not Path(path_data, 'train', 'Sample1.csv').exists():
            #make the path if it does not exist
            Path(path_data, 'train').mkdir(parents=True, exist_ok=True)
            Path(path_data, 'val').mkdir(parents=True, exist_ok=True)
            Path(path_data, 'test').mkdir(parents=True, exist_ok=True)
            #load data from matlab and save as CSV to ensure compatibility with the MultiFileTimeSeriesDataset
            reader = MatReader(Path(path_data,"Data","data.mat"),to_torch=False)
            x_train = reader.read_field('f_train')
            y_train = reader.read_field('u_train')
            grid_x_train = reader.read_field('x_train')[:,0]
            #save the data as CSV

            for i in range(x_train.shape[0]):
                df=pd.DataFrame(data={"time":grid_x_train,"forcing":x_train[i],"response":y_train[i]})
                df.to_csv(Path(path_data, 'train', f"Sample{i+1}.csv"),index=False)

            # all samples in a plot
            fig = go.Figure()
            for i in range(x_train.shape[0]):
                fig.add_trace(go.Scatter(x=grid_x_train, y=x_train[i], mode='lines', name='Forcing'))
                fig.add_trace(go.Scatter(x=grid_x_train, y=y_train[i], mode='lines', name='Response'))
            fig.update_layout(title="Train Set")
            #save to html
            fig.write_html(Path(path_data, 'train', "All_samples.html"))

            x_vali = reader.read_field('f_vali')
            y_vali = reader.read_field('u_vali')
            grid_x_vali = reader.read_field('x_vali')[:,0]
            #save the data as CSV

            for i in range(x_vali.shape[0]):
                df=pd.DataFrame(data={"time":grid_x_vali,"forcing":x_vali[i],"response":y_vali[i]})
                df.to_csv(Path(path_data, 'val', f"Sample{i+1}.csv"),index=False)


            # all samples in a plot
            fig = go.Figure()
            for i in range(x_vali.shape[0]):
                fig.add_trace(go.Scatter(x=grid_x_vali, y=x_vali[i], mode='lines', name='Forcing'))
                fig.add_trace(go.Scatter(x=grid_x_vali, y=y_vali[i], mode='lines', name='Response'))
            fig.update_layout(title="Validation Set")
            #save to html
            fig.write_html(Path(path_data, 'val', "All_samples.html"))


            x_test = reader.read_field('f_test')
            y_test = reader.read_field('u_test')
            grid_x_test = reader.read_field('x_test')[:,0]

            #save the data as CSV

            for i in range(x_test.shape[0]):
                df=pd.DataFrame(data={"time":grid_x_test,"forcing":x_test[i],"response":y_test[i]})
                df.to_csv(Path(path_data, 'test', f"Sample{i+1}.csv"),index=False)

            # all samples in a plot
            fig = go.Figure()
            for i in range(x_test.shape[0]):
                fig.add_trace(go.Scatter(x=grid_x_test, y=x_test[i], mode='lines', name='Forcing'))
                fig.add_trace(go.Scatter(x=grid_x_test, y=y_test[i], mode='lines', name='Response'))
            fig.update_layout(title="Test Set")
            #save to html
            fig.write_html(Path(path_data, 'test', "All_samples.html"))





        TRAIN_FILES = list(Path(path_data, 'train').rglob("*.csv"))
        VAL_FILES = list(Path(path_data, 'val').rglob("*.csv"))
        TEST_FILES = list(Path(path_data, 'test').rglob("*.csv"))

        #check if the data split is valid otherwise raise an error
        if data_split not in ['train','val','test']:
            raise ValueError(f"Invalid data split: {data_split}. Must be one of ['train','val','test']")
        # Initialize the dataset with the specified configuration

        super().__init__(directory=path_data,
                         files=TRAIN_FILES if data_split == 'train' else VAL_FILES if data_split == 'val' else TEST_FILES,
                         config=config,
                         device=device)

    def load_file(self, file_path: Path) -> pd.DataFrame:
        """Loads a CSV file as a DataFrame."""
        return pd.read_csv(file_path, delimiter=",")

    @staticmethod
    def get_channels() -> Dict[str, str]:
        """Specifies the channel names for time, excitation, response, and auxiliary data."""
        return {
            "time": "time",
            "excitation": [
                "forcing"
            ],
            "response": [
                "response",
            ],
            "auxiliary": []
        }


class LNO_1D_Duffing_c0(LNO_Data_base):
    path="1D_Duffing_c0"
    def __init__(self, config: dict, data_split:str='train',device: Optional[str] = 'cpu'):
        super().__init__(config, data_split, device, self.path)

class LNO_1D_Duffing_c05(LNO_Data_base):
    path="1D_Duffing_c05"
    def __init__(self, config: dict, data_split:str='train',device: Optional[str] = 'cpu'):
        super().__init__(config, data_split, device, self.path)

class LNO_1D_Lorenz_rho5(LNO_Data_base):
    path="1D_Lorenz_rho5"
    def __init__(self, config: dict, data_split:str='train',device: Optional[str] = 'cpu'):
        super().__init__(config, data_split, device, self.path)

class LNO_1D_Lorenz_rho10(LNO_Data_base):
    path="1D_Lorenz_rho10"
    def __init__(self, config: dict, data_split:str='train',device: Optional[str] = 'cpu'):
        super().__init__(config, data_split, device, self.path)


class LNO_1D_Pendulum_c0(LNO_Data_base):
    path="1D_Pendulum_c0"
    def __init__(self, config: dict, data_split:str='train',device: Optional[str] = 'cpu'):
        super().__init__(config, data_split, device, self.path)

class LNO_1D_Pendulum_c05(LNO_Data_base):
    path="1D_Pendulum_c05"
    def __init__(self, config: dict, data_split:str='train',device: Optional[str] = 'cpu'):
        super().__init__(config, data_split, device, self.path)

if __name__ == "__main__":
    import plotly.graph_objects as go
    import plotly.subplots as sp

    # Configuration parameters
    config = {
        "stride": 10,
        "history_length": 48,
        "forecast_length": 2000,
        "batch_size": 32
    }


    def plot_sample(sample, sample_index=0,name=""):
        """
        Plots a specific sample from the batch, with time as the x-axis, and separate subplots for excitations and responses.

        Args:
            sample (dataset): The dataset object containing the sample data.
            sample_index (int): Index of the sample in the batch to visualize.
        """
        # Extract data for the specified sample
        time_hist = sample[sample_index]["time_history"].numpy().flatten()
        time_fore = sample[sample_index]["time_forecast"].numpy().flatten()
        excitation_hist = sample[sample_index]["excitation_history"].numpy()
        excitation_fore = sample[sample_index]["excitation_forecast"].numpy()
        response_hist = sample[sample_index]["response_history"].numpy()
        response_fore = sample[sample_index]["response_forecast"].numpy()

        # Get channel names from the dataset for labeling
        channels = LNO_Data_base.get_channels()
        excitation_channels = channels["excitation"]
        response_channels = channels["response"]

        # Create subplots: one for excitation and one for response
        fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Excitations", "Responses"))

        # Plot excitations
        for i, channel_name in enumerate(excitation_channels):
            fig.add_trace(
                go.Scatter(x=time_hist, y=excitation_hist[:, i], mode='lines', name=f"{channel_name} History"),
                row=1, col=1)
            fig.add_trace(
                go.Scatter(x=time_fore, y=excitation_fore[:, i], mode='lines', name=f"{channel_name} Forecast"),
                row=1, col=1)

        # Plot responses
        for i, channel_name in enumerate(response_channels):
            fig.add_trace(
                go.Scatter(x=time_hist, y=response_hist[:, i], mode='lines', name=f"{channel_name} History"),
                row=1, col=2)
            fig.add_trace(
                go.Scatter(x=time_fore, y=response_fore[:, i], mode='lines', name=f"{channel_name} Forecast"),
                row=1, col=2)

        # Update layout and axis labels
        fig.update_layout(height=600, width=1200, title_text=f"{name} Dataset - Sample Visualization")
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Excitation Levels", row=1, col=1)
        fig.update_yaxes(title_text="Response Levels", row=1, col=2)

        fig.show()


    # Load dataset splits
    print("LNO 1D Duffing c0")
    train_data = LNO_1D_Duffing_c0(config, data_split='train')
    val_data = LNO_1D_Duffing_c0(config, data_split='val')
    test_data = LNO_1D_Duffing_c0(config, data_split='test')

    # Plot a sample from the training set
    plot_sample(train_data,name="Duffing c0")

    print("LNO 1D Duffing c05")
    train_data = LNO_1D_Duffing_c05(config, data_split='train')
    val_data = LNO_1D_Duffing_c05(config, data_split='val')
    test_data = LNO_1D_Duffing_c05(config, data_split='test')

    # Plot a sample from the training set
    plot_sample(train_data,name="Duffing c05")

    print("LNO 1D Lorenz rho5")
    train_data = LNO_1D_Lorenz_rho5(config, data_split='train')
    val_data = LNO_1D_Lorenz_rho5(config, data_split='val')
    test_data = LNO_1D_Lorenz_rho5(config, data_split='test')

    # Plot a sample from the training set
    plot_sample(train_data,name="Lorenz rho5")

    print("LNO 1D Lorenz rho10")
    train_data = LNO_1D_Lorenz_rho10(config, data_split='train')
    val_data = LNO_1D_Lorenz_rho10(config, data_split='val')
    test_data = LNO_1D_Lorenz_rho10(config, data_split='test')

    # Plot a sample from the training set
    plot_sample(train_data,name="Lorenz rho10")

    print("LNO 1D Pendulum c0")
    train_data = LNO_1D_Pendulum_c0(config, data_split='train')
    val_data = LNO_1D_Pendulum_c0(config, data_split='val')
    test_data = LNO_1D_Pendulum_c0(config, data_split='test')

    # Plot a sample from the training set
    plot_sample(train_data,name="Pendulum c0")


    print("LNO 1D Pendulum c05")
    train_data = LNO_1D_Pendulum_c05(config, data_split='train')
    val_data = LNO_1D_Pendulum_c05(config, data_split='val')
    test_data = LNO_1D_Pendulum_c05(config, data_split='test')

    # Plot a sample from the training set
    plot_sample(train_data,name="Pendulum c05")












