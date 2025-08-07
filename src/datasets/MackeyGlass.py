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
from typing import List, Tuple

def generate_excitation_waveforms(trajectories_to_sample=100, t_nsamples=200, waveform_type="sawtooth", t_end=20.0, freq=0.5):
    """
    Generates various types of waveforms: 'sawtooth', 'sine', 'square', 'triangle', 'step', and more.

    Args:
        trajectories_to_sample (int): Number of trajectories to generate.
        t_nsamples (int): Number of time steps per trajectory.
        waveform_type (str): Type of waveform to generate.
        t_end (float): End time of the simulation.
        freq (float): Frequency of the waveform in Hz.

    Returns:
        torch.Tensor: Tensor of generated trajectories with shape [trajectories_to_sample, t_nsamples, 1].
        torch.Tensor: Time vector with shape [t_nsamples].
    """
    # Define time steps
    t_begin = t_end / t_nsamples
    ti = torch.linspace(t_begin, t_end, t_nsamples)  # Time vector

    # Define offsets x0 for different trajectories
    x0s = torch.linspace(0, 2 * torch.pi, trajectories_to_sample)

    # Define angular frequency
    omega = 2 * torch.pi * freq

    # Define helper functions for different waveforms
    def sawtooth_wave(t_in, x0=0):
        return (omega * t_in + x0) / (2 * torch.pi) - torch.floor((omega * t_in + x0) / (2 * torch.pi))

    def sine_wave(t_in, x0=0):
        return 0.5 * (1 + torch.sin(omega * t_in + x0))

    def square_wave(t_in, x0=0):
        return torch.sign(torch.sin(omega * t_in + x0)) * 0.5 + 0.5

    def triangle_wave(t_in, x0=0):
        return 2 * torch.abs(2 * ((omega * t_in + x0) / (2 * torch.pi) - torch.floor(0.5 + (omega * t_in + x0) / (2 * torch.pi)))) - 1

    def step_wave(t_in, x0=0, step_time=5.0):
        return (t_in >= step_time).float()

    def gaussian_wave(t_in, x0=0, sigma=1.0):
        return torch.exp(-((torch.sin(omega * t_in + x0)) ** 2) / (2 * sigma ** 2))

    def damped_sine_wave(t_in, x0=0, alpha=0.1):
        return torch.exp(-alpha * t_in) * torch.sin(omega * t_in + x0)

    def sigmoid_wave(t_in, x0=0):
        return 1 / (1 + torch.exp(-5 * torch.sin(omega * t_in + x0)))

    def exponential_wave(t_in, x0=0, alpha=0.1):
        return torch.sin(omega * t_in + x0) * torch.exp(-alpha * torch.abs(torch.sin(omega * t_in)))

    def hann_wave(t_in, x0=0):
        return 0.5 * (1 - torch.cos(omega * t_in / torch.pi))

    def blackman_wave(t_in, x0=0):
        return 0.42 - 0.5 * torch.cos(omega * t_in / torch.pi) + 0.08 * torch.cos(2 * omega * t_in / torch.pi)

    # Updated dictionary with waveform functions
    waveform_generators = {
        "sawtooth": sawtooth_wave,
        "sine": sine_wave,
        "square": square_wave,
        "triangle": triangle_wave,
        "step": step_wave,
        "gaussian": gaussian_wave,
        "damped_sine": damped_sine_wave,
        "sigmoid": sigmoid_wave,
        "exponential": exponential_wave,
        "hann": hann_wave,
        "blackman": blackman_wave,
    }

    # Validate the waveform type
    if waveform_type not in waveform_generators:
        raise ValueError(f"Unsupported waveform type: {waveform_type}. Choose from {list(waveform_generators.keys())}.")

    # Select the appropriate generator function
    generator = waveform_generators[waveform_type]

    # Generate trajectories for each offset
    trajs = [generator(ti, x0) for x0 in x0s]

    # Create tensor for trajectories
    y = torch.stack(trajs)
    trajectories_generated = y.view(trajectories_to_sample, -1, 1)

    return trajectories_generated, ti


import numpy as np
import pandas as pd
from typing import List, Tuple

def mackey_glass_euler_interpolated(
    forcing_signal: np.ndarray,
    time_forcing: np.ndarray,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: int = 10,
    tau: float = 2.0,
    dt: float = 0.01,
    T: float = None,
    y0: float = 1.0,
) -> pd.DataFrame:
    """
    Approximates the Mackey–Glass system
        dy/dt = beta * y(t - tau) / [1 + (y(t - tau))^n] - gamma * y(t) + x(t)
    using an explicit Euler scheme on a finer grid, then interpolates the results
    onto the time points in `time_forcing` (up to T).

    Parameters
    ----------
    forcing_signal : np.ndarray
        1D array containing forcing values x(t).
    time_forcing : np.ndarray
        1D array with time stamps corresponding to forcing_signal.
    beta : float, optional
        Mackey–Glass system parameter (default: 0.2).
    gamma : float, optional
        Mackey–Glass system parameter (default: 0.1).
    n : int, optional
        Exponent in the fractional term (default: 10).
    tau : float, optional
        Time delay (default: 2.0).
    dt : float, optional
        Suggested time step for the Euler approximation (default: 0.01).
    T : float, optional
        End time for the simulation. If None or larger than time_forcing[-1],
        we use T = time_forcing[-1].
    y0 : float, optional
        Initial condition y(0) (default: 1.0).

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing columns: "time", "y", "forcing".
        The time column matches the relevant portion of `time_forcing`.
    """

    # -- 0) If T not given or T > time_forcing[-1], use time_forcing[-1]
    if T is None or T > time_forcing[-1]:
        T = time_forcing[-1]

    # -- 1) Ensure dt <= minimal spacing in time_forcing
    min_spacing = np.min(np.diff(time_forcing))
    dt_fine = min(dt, 0.9999 * min_spacing)  # a small factor < 1 to be safe

    # -- 2) Fine time discretization for the Euler scheme
    t_vals_fine = np.arange(0, T + dt_fine, dt_fine)
    if t_vals_fine[-1] < T:  # ensure we cover up to T
        t_vals_fine = np.append(t_vals_fine, T)

    # -- 3) Arrays to store y(t) on the fine grid
    y_vals_fine = np.zeros_like(t_vals_fine)
    y_vals_fine[0] = y0

    # -- 4) Helper to retrieve y(t - tau)
    def y_lag(idx):
        """Return y(t - tau) by nearest index on the fine grid,
           or use y0 if (t - tau) < 0."""
        t_delay = t_vals_fine[idx] - tau
        if t_delay < 0:
            return y0
        else:
            j = int(round(t_delay / dt_fine))
            return y_vals_fine[j]

    # -- 5) Forcing interpolation for arbitrary times on the fine grid
    def forcing_at_time(tt: float) -> float:
        return np.interp(tt, time_forcing, forcing_signal)

    # -- 6) Euler integration on the fine grid
    for i in range(len(t_vals_fine) - 1):
        t_now = t_vals_fine[i]
        forcing_now = forcing_at_time(t_now)
        y_tau = y_lag(i)  # y(t - tau)

        # DGL: dy/dt = ...
        dydt = beta * y_tau/(1.0 + y_tau**n) - gamma * y_vals_fine[i] + forcing_now

        # Forward Euler step
        dt_step = t_vals_fine[i+1] - t_vals_fine[i]
        y_vals_fine[i+1] = y_vals_fine[i] + dt_step * dydt

    # -- 7) Now interpolate solution y(t) back onto the original time_forcing up to T
    #       i.e., keep the portion time_forcing[time_forcing <= T]
    mask_T = time_forcing <= T
    t_final = time_forcing[mask_T]
    # If T < time_forcing[0], we won't have valid times, handle edge case
    if len(t_final) == 0:
        # No overlap with time_forcing: just return the last state or an empty df
        return pd.DataFrame(columns=["time", "y", "forcing"])

    y_final = np.interp(t_final, t_vals_fine, y_vals_fine)
    forcing_final = np.interp(t_final, time_forcing, forcing_signal)

    df = pd.DataFrame({
        "time": t_final,
        "y": y_final,
        "forcing": forcing_final
    })
    return df


def generate_time_series(
    forcing_signal: np.ndarray,
    time_forcing: np.ndarray,
    initial_values: List[float],
    beta: float = 0.2,
    gamma: float = 0.1,
    n: int = 10,
    tau: float = 2.0,
    dt: float = 0.01,
    T: float = None
) -> List[Tuple[int, pd.DataFrame]]:
    """
    Generates multiple time series of the Mackey–Glass system for different initial conditions.
    Each trajectory is computed using an explicit Euler scheme with a fine time step
    and then interpolated onto the given time_forcing grid (up to T).

    Parameters
    ----------
    forcing_signal : np.ndarray
        1D or 2D array of forcing signals.
        If 2D: [num_samples, time_length].
    time_forcing : np.ndarray
        1D array of time stamps corresponding to forcing_signal.
    initial_values : List[float]
        Initial conditions y(0) for different simulations.
    beta, gamma, n, tau : float
        Mackey–Glass system parameters.
    dt : float, optional
        Time step for the internal Euler integration (default: 0.01).
    T : float, optional
        Desired end time. If None or T > time_forcing[-1], we use time_forcing[-1].

    Returns
    -------
    time_series_data : List[Tuple[int, pd.DataFrame]]
        A list of (id, df) tuples. Each `df` has columns ["time", "y", "forcing"].
    """

    # If forcing_signal is 1D, reshape to 2D
    if forcing_signal.ndim == 1:
        forcing_signal = forcing_signal[None, :]
    num_samples = forcing_signal.shape[0]

    time_series_data = []
    idx = 0

    for i in range(num_samples):
        forcing_i = forcing_signal[i]
        for y0 in initial_values:
            df = mackey_glass_euler_interpolated(
                forcing_signal=forcing_i,
                time_forcing=time_forcing,
                beta=beta,
                gamma=gamma,
                n=n,
                tau=tau,
                dt=dt,
                T=T,
                y0=y0
            )
            # Attach metadata if desired
            time_series_data.append((idx, df))
            idx += 1

    return time_series_data
class MackeyGlassSystem(MultiFileTimeSeriesDataset):
    """
    Dataset class for managing time series data specific to the Mackey Glass system.

    This class loads data from predefined files, splits it into train, validation, and test sets,
    and organizes it according to the channels for time, excitation, response, and auxiliary data.
    """



    def __init__(self, config: dict, data_split:str='train',device: Optional[str] = 'cpu'):
        """
        Initializes the LeakageAnomalyDataset with specified configuration.

        Args:
            config (dict): Configuration dictionary for dataset parameters.
            scaler_type (str): Type of scaler ('standard' or 'minmax').
            device (Optional[str]): Device for storing tensors ('cpu' or 'cuda').
        """
        if config["zero_initial_conditions"]:
            path_data = Path(CONST.DATA_PATH, "MG_Dataset_ZeroIC")
        else:
            path_data = Path(CONST.DATA_PATH, "MG_Dataset")
        t_nsamples = 550
        num_inital_conditions = 30
        excitations_num = {'train': 10,
                           'val': 5,
                           'test': 5}
        freqencies = {'train': 0.15,
                      'val': 0.15,
                      'test': 0.15}

        forcing_signals = {'train': "sigmoid",
                           'val': "damped_sine",
                           'test': "triangle"}

        # check if the data is already generated
        if not Path(path_data, 'train', 'Sample1.csv').exists():
            #make the path if it does not exist
            Path(path_data, 'train').mkdir(parents=True, exist_ok=True)
            Path(path_data, 'val').mkdir(parents=True, exist_ok=True)
            Path(path_data, 'test').mkdir(parents=True, exist_ok=True)

            # set seed to ensure reproducibility
            #get current seed
            seed = torch.initial_seed()
            #set seed
            torch.manual_seed(seed)

            for ds_type in ['train', 'val', 'test']:
                waveform = forcing_signals[ds_type]
                freqency = freqencies[ds_type]
                excitation_num = excitations_num[ds_type]
                # Generate forcing signal
                forcing_signal, time_forcing = generate_excitation_waveforms(trajectories_to_sample=excitation_num,
                                                                             t_nsamples=t_nsamples,
                                                                             waveform_type=waveform,
                                                                             freq=freqency)
                forcing_signal = forcing_signal.squeeze().numpy()
                time_forcing = time_forcing.numpy()
                # Define initial conditions
                initial_conditions = []
                if config["zero_initial_conditions"] :
                    initial_conditions=[0]
                else:
                    for _ in range(num_inital_conditions):
                        random_values = torch.rand(1)  # Generate two random values in [0, 1)
                        initial_conditions.append(random_values)
                # Generate time series data
                time_series_data = generate_time_series(forcing_signal, time_forcing, initial_conditions,
                                                        config["beta"],
                                                        config["gamma"],
                                                        config["n"],
                                                        config["tau"],
                                                        config["dt"])
                for idx, df in time_series_data:
                    df.to_csv(Path(path_data, ds_type, f"Sample{idx}.csv"), index=False)
                # put all samples into a plot with plotly
                fig = go.Figure()
                for idx, df in time_series_data:
                    fig.add_trace(go.Scatter(x=df["time"], y=df["y"], mode='lines', name=f"Sample {idx}"))
                    fig.add_trace(go.Scatter(x=df["time"], y=df["forcing"], mode='lines', name=f"Forcing Signal {idx}"))
                fig.update_layout(title="Mackey Glass System - Generated Time Series Data",
                                  xaxis_title="Time")
                # save the plot
                fig.write_html(Path(path_data, ds_type, "Data.html"))

            #set seed back to original value
            torch.manual_seed(seed)

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
                "y",
            ],
            "auxiliary": []
        }

if __name__ == "__main__":
    import plotly.graph_objects as go
    import plotly.subplots as sp

    # Configuration parameters
    config = {
        "stride": 10,
        "history_length": 50,
        "forecast_length": 500,
        "batch_size": 32,
        "zero_initial_conditions": False
    }



    # Load dataset splits
    train_data = MackeyGlassSystem(config, data_split='train')
    val_data = MackeyGlassSystem(config, data_split='val')
    test_data = MackeyGlassSystem(config, data_split='test')

    def plot_sample(sample, sample_index=0):
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
        channels = MackeyGlassSystem.get_channels()
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
        fig.update_layout(height=600, width=1200, title_text="SMD Dataset- Sample Visualization")
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Excitation Levels", row=1, col=1)
        fig.update_yaxes(title_text="Response Levels", row=1, col=2)

        fig.show()


    # Plot a sample from the training set
    plot_sample(train_data)
    print("Train Data")

    #test inverse transform
    sample = train_data[0]
    excitation_fore = sample["excitation_forecast"]
    response_fore = sample["response_forecast"]
    excitation_hist = sample["excitation_history"]
    response_hist = sample["response_history"]


    # Inverse transform the forecast data
    excitation_fore_inv ,response_fore_inv= train_data.inverse_transform(excitation_fore, response_fore)
    # Inverse transform the history data
    excitation_hist_inv,response_hist_inv = train_data.inverse_transform(excitation_hist, response_hist)
    #concatenate the forecast and history data
    excitation = pd.concat((excitation_hist_inv, excitation_fore_inv), axis=0, ignore_index=True)
    response = pd.concat((response_hist_inv, response_fore_inv), axis=0, ignore_index=True)

    #also load the raw file and compare the values
    dir=Path(CONST.DATA_PATH, "MG_Dataset")
    raw_data=pd.read_csv(dir / train_data.files[0])
    excitation_raw=raw_data[MackeyGlassSystem.get_channels()["excitation"]]
    response_raw=raw_data[MackeyGlassSystem.get_channels()["response"]]
    #cut according to the length of the forecast and history
    excitation_raw=excitation_raw.iloc[:config["history_length"]+config["forecast_length"]]
    response_raw=response_raw.iloc[:config["history_length"]+config["forecast_length"]]

    #compare raw and transformed data in a plot
    fig=sp.make_subplots(rows=1, cols=2, subplot_titles=("Excitations", "Responses"))
    for i, channel_name in enumerate(excitation_raw.columns):
        fig.add_trace(go.Scatter(x=excitation.index, y=excitation[channel_name], mode='markers', name=f"{channel_name} Transformed"),row=1, col=1)
        fig.add_trace(go.Scatter(x=excitation.index, y=excitation_raw[channel_name], mode='lines', name=f"{channel_name} Raw"),row=1, col=1)
    for i, channel_name in enumerate(response_raw.columns):
        fig.add_trace(go.Scatter(x=response.index, y=response[channel_name], mode='markers', name=f"{channel_name} Transformed"),row=1, col=2)
        fig.add_trace(go.Scatter
                        (x=response.index, y=response_raw[channel_name], mode='lines', name=f"{channel_name} Raw"),row=1, col=2)
    fig.show()

    # also check the first two samples for verification of the slicing
    sample1 = train_data[0]
    sample2 = train_data[1]
    excitation_fore1 = sample1["excitation_forecast"].numpy()
    response_fore1 = sample1["response_forecast"].numpy()
    excitation_hist1 = sample1["excitation_history"].numpy()
    response_hist1 = sample1["response_history"].numpy()
    excitation_fore2 = sample2["excitation_forecast"].numpy()
    response_fore2 = sample2["response_forecast"].numpy()
    excitation_hist2 = sample2["excitation_history"].numpy()
    response_hist2 = sample2["response_history"].numpy()
    channels = MackeyGlassSystem.get_channels()
    #plot the first two samples in two subplots
    fig = sp.make_subplots(rows=2, cols=2, subplot_titles=("Excitations", "Responses"))
    for i in range(excitation_hist1.shape[1]):
        channel_name = channels["excitation"][i]
        index_array_hist = np.arange(len(excitation_hist1))
        index_array_fore = np.arange(len(excitation_fore1))+len(excitation_hist1)
        fig.add_trace(go.Scatter(x=index_array_hist, y=excitation_hist1[:, i], mode='lines', name=f"{channel_name} History 1"), row=1, col=1)
        fig.add_trace(go.Scatter(x=index_array_fore, y=excitation_fore1[:, i], mode='lines', name=f"{channel_name} Forecast 1"), row=1, col=1)
        fig.add_trace(go.Scatter(x=index_array_hist, y=excitation_hist2[:, i], mode='lines', name=f"{channel_name} History 2"), row=2, col=1)
        fig.add_trace(go.Scatter(x=index_array_fore, y=excitation_fore2[:, i], mode='lines', name=f"{channel_name} Forecast 2"), row=2, col=1)
    for i in range(response_hist1.shape[1]):
        channel_name = channels["response"][i]
        index_array_hist = np.arange(len(response_hist1))
        index_array_fore = np.arange(len(response_fore1)) +len(excitation_hist1)
        fig.add_trace(go.Scatter(x=index_array_hist, y=response_hist1[:, i], mode='lines', name=f"{channel_name} History 1"), row=1, col=2)
        fig.add_trace(go.Scatter(x=index_array_fore, y=response_fore1[:, i], mode='lines', name=f"{channel_name} Forecast 1"), row=1, col=2)
        fig.add_trace(go.Scatter(x=index_array_hist, y=response_hist2[:, i], mode='lines', name=f"{channel_name} History 2"), row=2, col=2)
        fig.add_trace(go.Scatter
                        (x=index_array_fore, y=response_fore2[:, i], mode='lines', name=f"{channel_name} Forecast 2"), row=2, col=2)
    fig.show()







