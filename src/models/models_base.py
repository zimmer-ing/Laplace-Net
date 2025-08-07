# MIT License, Copyright (c) 2025 Bernd Zimmering
# See LICENSE file for details.
# Please cite: Zimmering, B. et al. (2025), "Breaking Free: Decoupling Forced Systems with Laplace Neural Networks", ECML PKDD, Porto.
import torch
import torch.nn as nn
from torch.optim import Adam
from abc import ABC, abstractmethod
import json
import Constants as const
from tqdm import tqdm
from ..utils.helpers import ensure_sequential_dataloader, concatenate_batches
from typing import List, Tuple, Union, Optional
import warnings
import logging

class BaseRegressionModel(nn.Module, ABC):
    """
    Base class for regression models.

    This class provides a template for regression models, including methods for training,
    prediction, validation, and saving/loading the model.

    Attributes:
        config (dict): Configuration dictionary for the model.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        loss_fn (torch.nn.Module): Loss function used for training.
    """

    def __init__(self) -> None:
        """Initialize the base regression model."""
        super(BaseRegressionModel, self).__init__()

    def prepare_training(self, config: dict) -> None:
        """
        Prepare the model for training by initializing the optimizer and loss function.

        Args:
            config (dict): Configuration dictionary for training parameters (e.g., learning rate).
        """
        self.config = config
        self.optimizer = self._initialize_optimizer()
        self.loss_fn = nn.MSELoss()


    @abstractmethod
    def forward(self,excitation_history: torch.Tensor,response_history: torch.Tensor,
                time_history: torch.Tensor,
                excitation_forecast: torch.Tensor,
                time_forecast: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for the forward pass logic. Must be implemented in subclasses.

        Args:
            excitation_history (torch.Tensor): Excitation history tensor.
            response_history (torch.Tensor): Response history tensor.
            time_history (torch.Tensor): Time history tensor.
            excitation_forecast (torch.Tensor): Excitation forecast tensor.
            time_forecast (torch.Tensor): Time forecast tensor.
        """

        pass

    def trainable_parameters(self) -> List[nn.Parameter]:
        """
        Return the list of trainable parameters of the model.

        Returns:
            List[nn.Parameter]: List of trainable model parameters.
        """
        return list(self.parameters())

    def calculate_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss between predictions and targets.

        Args:
            predictions (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth target values.

        Returns:
            torch.Tensor: The calculated loss value.
        """
        return self.loss_fn(predictions, targets)

    def _initialize_optimizer(self) -> Adam:
        """
        Initialize the Adam optimizer with the learning rate from the config.

        Returns:
            Adam: The Adam optimizer.
        """

        lr = self.config.get("model", {}).get("learning_rate")

        if lr is None:
            # If learning rate is not specified, log a warning and set a default value
            warnings.warn("Learning rate not specified in configuration. Using default value of 0.001.", UserWarning)
            lr = 0.001
        return Adam(self.trainable_parameters(), lr=lr)

    def train_step(self, data_loader: torch.utils.data.DataLoader, return_loss: bool = False) -> Optional[float]:
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
        for batch in tqdm(data_loader, desc="Iteration Training Set", disable=not const.VERBOSE_BATCHES):
            excitation_history = batch['excitation_history']
            response_history = batch['response_history']
            excitation_forecast = batch['excitation_forecast']
            response_forecast = batch['response_forecast']
            time_history = batch['time_history']
            time_forecast = batch['time_forecast']
            self.optimizer.zero_grad()
            predictions = self(excitation_history, response_history, time_history, excitation_forecast, time_forecast)
            loss = self.calculate_loss(predictions, response_forecast)
            loss.backward()
            if return_loss:
                losses.append(loss.item())
            self.optimizer.step()
        if return_loss:
            return sum(losses) / len(losses)

    def predict(self, data_loader: torch.utils.data.DataLoader, samples_only: bool = False) -> dict:
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
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Iteration Prediction Set", disable=not const.VERBOSE_BATCHES):
                # Dynamically populate results dictionary with data from the first batch
                if not results:
                    results = {key: [] for key in batch.keys()}
                    results["predictions"] = []  # Add a key for predictions
                    #results["z_values"] = []  # Add a key for latent variables if used in the model

                # Append batch data to results dictionary
                for key, value in batch.items():
                    results[key].append(value)

                # Run the model's prediction
                y_hat= self(
                    batch.get("excitation_history"),
                    batch.get("response_history"),
                    batch.get("time_history"),
                    batch.get("excitation_forecast"),
                    batch.get("time_forecast")
                )
                results["predictions"].append(y_hat)
                #results["z_values"].append(z_batch)

        # If samples_only, return only the predictions and a subset of relevant data
        if samples_only:
            return results

        # Concatenate batches for each entry in the results dictionary
        for key in results.keys():
            results[key] = concatenate_batches(results[key])

        return results

    def validate_step(self, data_loader: torch.utils.data.DataLoader) -> float:
        """
        Validate the model on the given data loader.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader with validation data.

        Returns:
            float: Mean loss over the validation set.
        """
        losses = []
        self.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Iteration Validation Set", disable=not const.VERBOSE_BATCHES):
                excitation_history = batch['excitation_history']
                response_history = batch['response_history']
                excitation_forecast = batch['excitation_forecast']
                response_forecast = batch['response_forecast']
                time_history = batch['time_history']
                time_forecast = batch['time_forecast']
                predictions = self(excitation_history, response_history, time_history, excitation_forecast,
                                   time_forecast)
                loss = self.calculate_loss(predictions, response_forecast)
                losses.append(loss.item())
            if len(losses) == 0:
                raise ValueError("Validation set is empty.")
            return sum(losses) / len(losses)

    def save_model(self, path: str) -> None:
        """
        Save the model to the given path.

        Args:
            path (str): Path to save the model.
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path: str) -> None:
        """
        Load the model from the given path.

        Args:
            path (str): Path to load the model from.
        """
        self.load_state_dict(torch.load(path))
        self.eval()

    @staticmethod
    def save_config(config: dict, path: str) -> None:
        """
        Save the configuration to the given path (as JSON).

        Args:
            config (dict): Configuration dictionary.
            path (str): Path to save the configuration file.
        """
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)

    @staticmethod
    def load_config(path: str) -> dict:
        """
        Load the configuration from the given path (assumes JSON format).

        Args:
            path (str): Path to the configuration file.

        Returns:
            dict: Loaded configuration dictionary.
        """
        with open(path, 'r') as f:
            config = json.load(f)
        return config