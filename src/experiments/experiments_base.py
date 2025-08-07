# MIT License, Copyright (c) 2025 Bernd Zimmering
# See LICENSE file for details.
# Please cite: Zimmering, B. et al. (2025), "Breaking Free: Decoupling Forced Systems with Laplace Neural Networks", ECML PKDD, Porto.
import yaml
import torch
import warnings
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import Constants as CONST
from pathlib import Path
from src.models import MODEL_REGISTRY
from src.datasets import DATASET_REGISTRY
import hashlib
import optuna
import logging
from multiprocessing import cpu_count
from datetime import datetime
import numpy as np
import random
import json
import time
from multiprocessing import Lock, Condition, Manager
import gc
import os
import psutil
import subprocess
from src.utils.helpers import get_pytorch_nvidia_gpu_mapping, get_best_available_gpu,release_gpu

import multiprocessing as mp
import functools
from multiprocessing import Queue
import logging
import logging.handlers
from optuna.trial import TrialState
import sys
import atexit
import pandas as pd
from collections import Counter
import copy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class ConfigManager:
    """
    Manages the loading and validation of configurations for datasets, models,
    experiments, and their respective combination rules.
    """

    def __init__(self):
        """
        Initializes the configuration manager and sets base paths for different config types.
        """
        self.base_path = Path(CONST.CONFIGS_PATH)
        self.dataset_path = self.base_path / 'datasets'
        self.model_path = self.base_path / 'models'
        self.experiment_path = self.base_path / 'experiments'
        self.combination_rules_path = self.base_path / 'combination_rules.yaml'
        self.combination_rules = self._load_combination_rules()


    def _load_config(self, config_path):
        """
        Loads a YAML configuration file.

        Args:
            config_path (Path): Path to the configuration file.

        Returns:
            dict: The parsed YAML configuration file.
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with config_path.open("r") as file:
            return yaml.safe_load(file)

    def load_dataset_config(self, dataset_name):
        """
        Loads the configuration for a specific dataset.

        Args:
            dataset_name (str): The name of the dataset.

        Returns:
            dict: Configuration dictionary for the dataset.
        """
        config_path = Path(self.dataset_path , f"{dataset_name}.yaml")
        return self._load_config(config_path)

    def load_model_config(self, model_name):
        """
        Loads the configuration for a specific model.

        Args:
            model_name (str): The name of the model.

        Returns:
            dict: Configuration dictionary for the model.
        """
        config_path = Path(self.model_path , f"{model_name}.yaml")
        return self._load_config(config_path)

    def load_experiment_config(self, experiment_name):
        """
        Loads the configuration for a specific experiment.

        Args:
            experiment_name (str): The name of the experiment.

        Returns:
            dict: Configuration dictionary for the experiment.
        """
        config_path = Path(self.experiment_path , f"{experiment_name}.yaml")
        return self._load_config(config_path)

    def _load_combination_rules(self):
        """
        Loads the combination rules configuration if it exists.

        Returns:
            dict: Combination rules dictionary.
        """
        if self.combination_rules_path.exists():
            return self._load_config(self.combination_rules_path)
        return {}

    def apply_combination_rules(self, dataset_name, model_name, experiment_name, dataset_config, model_config):
        """
        Applies specific combination rules to modify the dataset or model configurations
        based on the dataset-model-experiment combination.

        Args:
            dataset_name (str): The name of the dataset.
            model_name (str): The name of the model.
            experiment_name (str): The name of the experiment.
            dataset_config (dict): Configuration for the dataset.
            model_config (dict): Configuration for the model.

        Returns:
            tuple: The adjusted configurations for the dataset and model.
        """
        # Check if specific rules exist for the combination of dataset, model, and experiment
        if (dataset_name in self.combination_rules and
                model_name in self.combination_rules[dataset_name] and
                experiment_name in self.combination_rules[dataset_name][model_name]):

            rules = self.combination_rules[dataset_name][model_name][experiment_name]
            if "adjust_x_features" in rules:
                limit = rules["adjust_x_features"]
                dataset_config["x_features"] = dataset_config["x_features"][:limit]
            if "custom_processing" in rules:
                # Placeholder for custom processing logic
                logging.info(f"Applying custom processing for {model_name} on {dataset_name} in experiment {experiment_name}")

        return dataset_config, model_config

    def load_full_config(self, dataset_name, model_name, experiment_name):
        """
        Loads and combines configurations for the specified dataset, model, and experiment.
        Applies any combination rules if applicable.

        Args:
            dataset_name (str): The name of the dataset.
            model_name (str): The name of the model.
            experiment_name (str): The name of the experiment.

        Returns:
            dict: Combined and adjusted configuration dictionary.
        """
        # Load individual configurations
        dataset_config = self.load_dataset_config(dataset_name)
        model_config = self.load_model_config(model_name)
        experiment_config = self.load_experiment_config(experiment_name)

        # Apply combination rules
        adjusted_dataset_config, adjusted_model_config = self.apply_combination_rules(
            dataset_name, model_name, experiment_name, dataset_config, model_config
        )

        # Combine all configurations into a single dictionary
        full_config = {
            "dataset": adjusted_dataset_config,
            "model": adjusted_model_config,
            "experiment": experiment_config
        }
        return full_config


def build_optuna_search_space(config):
    """
    Builds an Optuna-compatible search space from the configuration.


    Returns:
        dict: A dictionary where keys are parameter names and values are functions that
              take an Optuna trial object and return suggested values.
    """

    def create_suggestion_function(key, space):
        """Helper function to create the suggestion lambda based on the type."""
        if space["type"] == "int":
            return lambda trial: trial.suggest_int(key, int(space["low"]), int(space["high"]),
                                                   step=int(space.get("step", 1)))
        elif space["type"] == "float":
            return lambda trial: trial.suggest_float(key, float(space["low"]), float(space["high"]),
                                                     step=space.get("step", None))
        elif space["type"] == "categorical":
            return lambda trial: trial.suggest_categorical(key, space["choices"])
        elif space["type"] == "loguniform":
            return lambda trial: trial.suggest_float(key, float(space["low"]), float(space["high"]), log=True)
        elif space["type"] == "uniform":
            return lambda trial: trial.suggest_uniform(key, float(space["low"]), float(space["high"]))
        else:
            raise ValueError(f"Unknown hyperparameter type: {space['type']}")

    config_model = config['model']
    search_space = {}

    for key, value in config_model.items():
        if "search_space" in value:
            space = value["search_space"]
            search_space[key] = create_suggestion_function(key, space)

    return search_space

class ExperimentBase(ABC):
    """
    Abstract base class for running an experiment for a model. Provides the structure for
    initializing, training, hyperparameter tuning, and evaluating the model.
    """

    def __init__(self, model:str, dataset:str,experiment_name:str,seed=None,hash=None,console_mode=False):
        """
        Initializes the experiment with the necessary configurations, models, and datasets.

        Args:
            models (str): The name of the model for the experiment.
            datasets (str): The name of the dataset for the experiment.
            experiment_name (str): The name of the experiment. (name of the inheriting class)
            seed (int, optional): The random seed to use for reproducibility. If None, the seed from the config is used.
            hash (str, optional): The hash to use for the experiment. If None, a new hash is generated. Can be used to load a specific experiment and perfrom further analysis.
            console_mode (bool, optional): If True, only log to the console and not to a file. Can be used if an experiment is used for manual tests.
        """
        self.model_name = model
        self.dataset_name = dataset
        self.console_mode=console_mode
        self.config_manager = ConfigManager()
        # path with name and actual date and time
        self.config = self.config_manager.load_full_config(dataset, model, experiment_name)
        # set seed if
        if seed is not None:
            self.seed = seed
            self.set_seed(seed)
            warn_msg = "Seed from config (" + str(
                self.config['experiment']['seed']) + ") overwritten by seed from experiment(" + str(seed) + ")"
            warnings.warn(warn_msg)
        elif 'seed' in self.config['experiment']:
            self.seed = self.config['experiment']['seed']
            self.set_seed(self.seed)
        else:
            self.seed = 42
            self.set_seed(self.seed)
            warnings.warn("Seed not found in config for experiment, using default seed 42")
        if hash is None:
            config_hash = self.generate_hash_from_config()
        else:
            config_hash = hash
        self.config_hash=config_hash
        self.path_results = Path(CONST.RESULTS_PATH, experiment_name, model + '_' + dataset,'Seed_'+str(self.seed), config_hash)
        #create the path if it does not exist
        self.path_results.mkdir(parents=True, exist_ok=True)
        #clear timestamps.txt if it exist
        if Path(self.path_results, "timestamps.txt").exists():
            Path(self.path_results, "timestamps.txt").unlink()
        self._log_time("Start Experiment")
        # Create a logging queue for multiprocessing
        self.manager = Manager()
        self.log_queue = self.manager.Queue()

        # Initialize logging (Main Process)
        self.logger,self.queue_listener = self.__setup_logging()
        #make sure logger gets stopped at the end
        atexit.register(self.__stop_logging)


        self.model_class,self.dataset_class=self.initialize(model,dataset)




        #safe the config and start time
        with open(Path(self.path_results, "config_experiment.yaml"), 'w') as file:
            yaml.dump(self.config, file)

    def __setup_logging(self):
        log_file_path = self.path_results / 'experiment.log'

        # Clear all existing handlers to prevent duplicate logs
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)


        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        if self.console_mode:
            #just write to console
            queue_listener = logging.handlers.QueueListener(self.log_queue,  console_handler)
        else:
            queue_listener = logging.handlers.QueueListener(self.log_queue, file_handler,console_handler)
        queue_listener.start()

        logger = logging.getLogger(f"{self.__class__.__name__}_{self.dataset_name}_{self.model_name}_{self.config_hash}")

        # make sure QueueHandler is only added once
        if not any(isinstance(h, logging.handlers.QueueHandler) for h in logger.handlers):
            queue_handler = logging.handlers.QueueHandler(self.log_queue)
            logger.addHandler(queue_handler)

        logging_level = logging.DEBUG if CONST.DEBUG else logging.INFO
        logger.setLevel(logging_level)

        logger.debug(f"Logger initialized in debug mode and writing to {log_file_path}")

        return logger, queue_listener

    def __del__(self):
        self.__stop_logging()

    def finish(self):
        """
        Does everything to shut down the experiment
        :return:
        """
        self.__stop_logging()

    def __stop_logging(self):
        """
        Stops the logging listener when the experiment is finished.
        """
        if hasattr(self, 'queue_listener') and self.queue_listener is not None:
            try:
                self.logger.info("Stopping Logging (QueueListener)...")
                self.queue_listener.stop()
            except Exception as e:
                print(f"Warning: Exception while stopping QueueListener: {e}")

        # Ensure that the log queue exists before sending a stop signal
        if hasattr(self, 'log_queue') and self.log_queue is not None:
            try:
                self.log_queue.put_nowait(None)  # Send a sentinel to unblock the queue
            except Exception as e:
                print(f"Warning: Exception while sending stop signal to log queue: {e}")

        # Explicitly remove attributes to avoid using them after deletion
        self.queue_listener = None
        self.log_queue = None

    def print_experiment_path(self):
        print(f"Experiment path: {self.path_results}")

    def generate_hash_from_config(self) -> str:
        """
        Generates a short and consistent hash from the configuration to identify the experiment.

        Returns:
            str: The hash generated from the configuration.
        """
        # Convert the dictionary to a sorted JSON string to ensure consistency
        hash_string = json.dumps(self.config, sort_keys=True)

        # Generate a SHA-256 hash and return the first 8 characters for brevity
        return hashlib.sha256(hash_string.encode('utf-8')).hexdigest()[:8]
    @staticmethod
    def set_seed(seed):
        set_seed(seed)


    def initialize(self, model_name, dataset_name):
        """
        Initializes the model and dataset for a given seed.

        Args:
            model_name (str): The name of the model.
            dataset_name (str): The name of the dataset.
        Returns:
            tuple: The initialized model and dataset.
        """
        self.logger.info("Initializing model and dataset")
        #check if model and dataset are registered
        assert model_name in MODEL_REGISTRY, f"Model {model_name} not found in registry. Please add to src.model.__init__.py"
        assert dataset_name in DATASET_REGISTRY, f"Dataset {dataset_name} not found in registry. Please add to src.datasets.__init__.py"
        #add device to config
        #check if desired device is available
        assert 'wanted_device' in self.config['experiment'], "device not found in experiment config"
        assert self.config['experiment']['wanted_device'] in ['cuda','cpu','mps'], "device must be either 'cuda', 'cpu' or 'mps'"
        if self.config['experiment']['wanted_device']=='cuda' and torch.cuda.is_available():
            device_obj=torch.device('cuda')
            self.config['chosen_device']='cuda'
            self.logger.info("CUDA available, using GPU")
        elif self.config['experiment']['wanted_device']=='cuda' and not torch.cuda.is_available():
            device_obj=torch.device('cpu')
            self.config['chosen_device']='cpu'
            self.logger.info("CUDA not available, using CPU")
        elif self.config['experiment']['wanted_device']=='mps' and torch.backends.mps.is_available():
            device_obj=torch.device('mps')
            self.config['chosen_device']='mps'
            self.logger.info("MPS available, using MPS")
        elif self.config['experiment']['wanted_device']=='mps' and not torch.backends.mps.is_available():
            device_obj=torch.device('cpu')
            self.config['chosen_device']='cpu'
            self.logger.info("MPS not available, using CPU")
        else:
            device_obj=torch.device('cpu')
            self.config['chosen_device']='cpu'
            self.logger.info("Using CPU")
        self.device=device_obj

        return MODEL_REGISTRY[model_name],DATASET_REGISTRY[dataset_name]

    def check_hyperparameter_search(self, start_dashboard=False):
        """
        Checks the results of the hyperparameter run. Warns when the best configurations are near the border of the search space.
        Evaluates the top 10% trials based on the objective value.
        """
        from optuna.trial import FrozenTrial

        path_db = Path(self.path_results, "hyperparameter_tuning","optuna_study.db")
        study_names = optuna.study.get_all_study_names(storage="sqlite:///" + str(path_db))
        self.logger.info(f"Evaluating Trials for {self.model_name} on {self.dataset_name}")
        self.logger.info(f"Available studies: {study_names}")
        if study_names:
            correct_study_name = study_names[0]
        else:
            raise ValueError("No study found in the database!")

        study = optuna.load_study(
            study_name=correct_study_name,
            storage="sqlite:///" + str(path_db)
        )

        search_space = build_optuna_search_space(self.config)

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        error_trails=[t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        pruned_trails=[t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        # some statistics

        self.logger.info(f"Number of completed trials: {len(completed_trials)} of {len(study.trials)} total trials.")
        self.logger.info(f"Number of pruned trials: {len(pruned_trails)} of {len(study.trials)} total trials.")
        self.logger.info(f"Number of error trials: {len(error_trails)} of {len(study.trials)} total trials.")

        if not completed_trials:
            self.logger.warning("No completed trials found.")
            return

        # Take the top 5% trials (at least 5)
        sorted_trials = sorted(completed_trials, key=lambda t: t.value)
        top_n = max(5, int(len(sorted_trials) * 0.05))
        top_trials = sorted_trials[:top_n]

        self.logger.info(f"Analyzing the top {top_n} trials:")
        self.logger.info(f"The top trials are: {[t.number for t in top_trials]}")
        # Initialize counters for each parameter to track the frequency of values
        param_counts = {param: Counter() for param in search_space.keys()}

        # Dictionary to store warnings for parameters near their boundaries
        warnings_dict = {}

        # Iterate through the top trials to collect statistics
        for trial in top_trials:
            for param_name, param_value in trial.params.items():
                if param_name in search_space:
                    space = self.config['model'][param_name]['search_space']

                    # Check if a numerical comparison is possible
                    if space["type"] in ["int", "float", "loguniform", "uniform"]:
                        low, high = float(space["low"]), float(space["high"])

                    elif space["type"] == "categorical":
                        choices = space["choices"]

                        # Check if all choices are numeric
                        if all(isinstance(v, (int, float)) for v in choices):
                            low, high = float(min(choices)), float(max(choices))
                            space["low"], space["high"] = low, high
                        else:
                            # Skip non-numeric categorical values
                            continue
                    else:
                        continue

                    # Add the value to the frequency statistics
                    param_counts[param_name][param_value] += 1

                    # Check if the parameter value is near the boundaries
                    threshold = (high - low) * 0.10  # 10% of the range as threshold
                    lower_diff = abs(param_value - low)
                    upper_diff = abs(param_value - high)

                    if lower_diff < threshold or upper_diff < threshold:
                        if param_name not in warnings_dict:
                            warnings_dict[param_name] = []

                        warning_msg = f"Trial {trial.number} - Value: {param_value:.5f} (Range: {low} - {high})"
                        if lower_diff < threshold:
                            warning_msg += f" ⚠ Near Lower Bound ({low})"
                        if upper_diff < threshold:
                            warning_msg += f" ⚠ Near Upper Bound ({high})"

                        warnings_dict[param_name].append(warning_msg)

        # **Print summary of best hyperparameter values**
        self.logger.info("\n--- Best Hyperparameter Values ---\n")
        summary_data = []
        for param_name, counts in param_counts.items():
            most_common = counts.most_common(top_n)  # Take the top N most frequent values
            values_str = ", ".join([f"{val} ({count}x)" for val, count in most_common])
            summary_data.append([param_name, values_str])

        # Create a DataFrame for displaying results
        df_summary = pd.DataFrame(summary_data, columns=["Hyperparameter", "Top Values"])

        # Print the summary table
        self.logger.info(df_summary.to_string(index=False))

        # **Print warnings for hyperparameters near their boundaries**
        if warnings_dict:
            self.logger.info("\n⚠ Hyperparameter Warnings ⚠")
            for param, warnings_list in warnings_dict.items():
                self.logger.info(f"\n🔹 {param}:")
                for warning in warnings_list:
                    self.logger.info(f"  {warning}")

        self.logger.info("\nHyperparameter search analysis completed.")

        # Start Optuna Dashboard if requested
        if start_dashboard:
            try:
                subprocess.run(
                    ["optuna-dashboard", f"sqlite:///{path_db}"],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error starting Optuna Dashboard: {e}")

    @staticmethod
    def run_trial(config, model_class_name, dataset_class_name, default_config, study_db_path, gpu_mapping,
                  lock, completed_trials, first_trial_nr,log_queue,seed,gpu_lock,gpu_status):
        """
        Runs a single Optuna trial in an isolated process.

        Args:
            config (dict): Experiment configuration.
            model_class_name (str): Model name.
            dataset_class_name (str): Dataset name.
            default_config (dict): Default model configuration.
            study_db_path (str): Path to the Optuna SQLite database.
            gpu_mapping (dict): PyTorch-to-NVIDIA GPU mapping.
            lock (multiprocessing.Lock): Synchronization lock for GPU selection.
            queue (multiprocessing.Queue): Queue for trial synchronization.
            completed_trials (multiprocessing.Manager().list()): Shared list of completed trials.
            first_trial_nr (int): The number of the first trial in the study.
            log_queue (multiprocessing.Queue): Queue for logging.
            seed (int): Random seed for the trial.
            gpu_lock (multiprocessing.Lock): Lock for GPU selection.
            gpu_status (multiprocessing.Manager().dict()): Shared dictionary for GPU status.

        Returns:
            float: Validation loss of the trial.
        """
        # Set up logging
        # Create a logger for the worker process
        logger = logging.getLogger("WorkerProcess")
        # Set logging level (inherits from main process)
        if CONST.DEBUG:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO
        logger.setLevel(log_level)

        # Attach the QueueHandler so logs are forwarded to the main process
        if not any(isinstance(h, logging.handlers.QueueHandler) for h in logger.handlers):
            queue_handler = logging.handlers.QueueHandler(log_queue)
            logger.addHandler(queue_handler)

        try:
            #set seed

            hyperparameter_config = config['experiment']['hyperparameter_tuning']
            if hyperparameter_config['use_pruner']:
                pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=hyperparameter_config['min_epochs'],
                                                                reduction_factor=hyperparameter_config[
                                                                    'reduction_factor'])
            else:
                pruner = optuna.pruners.NopPruner()
            # Load Optuna study (each process must reconnect separately)
            study = optuna.load_study(
                study_name=f"Hyperparameter Optimization for {model_class_name} on {dataset_class_name}",
                storage=f"sqlite:///{study_db_path}",
                pruner=pruner,
            )

            with lock:
                trial = study.ask()
            trial_id = trial.number
            study.sampler = optuna.samplers.TPESampler(seed=seed+trial_id)
            set_seed(seed+trial_id)
            logger.debug(f"Starting trial {trial_id} seed {seed+trial_id}")
            logger.info(f"Starting trial {trial_id} process with PID {os.getpid()}")


            # Select GPU dynamically if CUDA is available
            if torch.cuda.is_available() and default_config['chosen_device'] == 'cuda':
                nv_gpu_id = get_best_available_gpu(gpu_lock,gpu_status,logger)
                pt_gpu_id = gpu_mapping.get(nv_gpu_id, 0)  # Default to GPU 0 if mapping fails
                torch.cuda.set_device(pt_gpu_id)
                device = torch.device(f"cuda:{pt_gpu_id}")
                logger.debug(f"Trial {trial_id} running on NVIDIA-SMI GPU {nv_gpu_id} (PyTorch GPU cuda:{pt_gpu_id})")
            else:
                device = torch.device('cpu')

            if default_config['chosen_device'] == 'mps':
                device = torch.device('mps')


            if default_config['chosen_device'] == 'cuda':
                logger.info(f"Trial {trial_id} running on device: {device} which is NVIDIA-SMI GPU {nv_gpu_id}")
            else:
                logger.info(f"Trial {trial_id} running on device: {device}")



            # Sample hyperparameters from the search space
            search_space = build_optuna_search_space(config)
            hyperparameters = {key: suggest_fn(trial) for key, suggest_fn in search_space.items()}
            logger.debug(f"Trial {trial_id} sampled hyperparameters: {hyperparameters}")

            # Update model configuration with sampled hyperparameters
            config = copy.deepcopy(default_config) #make sure we have a copy in any case
            config['model'].update(hyperparameters)

            # Initialize dataset and model
            model_class = MODEL_REGISTRY[model_class_name]
            dataset_class = DATASET_REGISTRY[dataset_class_name]
            model = model_class(config)

            # Load training and validation datasets
            dataset_train = dataset_class(config['dataset'], data_split='train', device=device)
            dataset_val = dataset_class(config['dataset'], data_split='val', device=device)

            batch_size = config['experiment']['batch_size']
            if batch_size<len(dataset_train):
                shuffle=True
            else:
                shuffle=False
            train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle,
                                      collate_fn=dataset_class.collate_fn)
            val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=dataset_class.collate_fn)

            total_epochs = config['experiment']['hyperparameter_tuning']['num_epochs']

            for epoch in range(total_epochs):
                model.train_step(train_loader)
                val_loss = model.validate_step(val_loader)
                if epoch % max(1, total_epochs // 50) == 0 or epoch == total_epochs - 1:
                    with lock:
                        trial.report(val_loss, epoch)

                #log every 10% of the epochs
                if epoch % max(1, total_epochs // 10) == 0 or epoch == total_epochs - 1:
                    logger.debug(f"Trail {trial_id} in Epoch {epoch + 1}/{total_epochs}: Validation Loss: {val_loss:.5f}")


                # Check for pruning condition
                if hyperparameter_config['use_pruner']:
                    if trial.should_prune():

                        logger.debug(f"Trial {trial_id} pruned at epoch {epoch}.")
                        if CONST.DEBUG and trial.number - 1 not in completed_trials and trial.number > first_trial_nr:
                            logger.debug(f"Trial {trial_id} waiting for Trial {trial_id - 1}...")
                        # Wait until the previous trial finishes before proceeding
                        while trial.number - 1 not in completed_trials and trial.number > first_trial_nr:
                            if CONST.DEBUG:
                                time.sleep(3)
                                logger.debug(f"Trial {trial_id} still waiting for Trial {trial_id - 1}...")
                            time.sleep(0.5)

                            # Report final result to Optuna
                        logger.info(f"Trial {trial_id} pruned with val_loss: {val_loss}")
                        if torch.cuda.is_available() and default_config['chosen_device'] == 'cuda':
                            release_gpu(gpu_lock,gpu_status,nv_gpu_id)
                        with lock:
                            study.tell(trial, state=TrialState.PRUNED)
                            completed_trials.append(trial.number)
                        return



            # Ensure trials finish in the correct order, accounting for first_trial_nr
            logger.debug(f"Trial {trial_id} is ready now. Check if prior trails are ready")

            if CONST.DEBUG and trial.number - 1 not in completed_trials and trial.number > first_trial_nr:
                logger.debug(f"Trial {trial_id} waiting for Trial {trial_id - 1}...")
            # Wait until the previous trial finishes before proceeding
            while trial.number - 1 not in completed_trials and trial.number > first_trial_nr:
                if CONST.DEBUG:
                    time.sleep(3)
                    logger.debug(f"Trial {trial_id} still waiting for Trial {trial_id - 1}...")
                time.sleep(0.5)

            # Report final result to Optuna
            logger.info(f"Trial {trial_id} completed successfully with val_loss: {val_loss}")
            if torch.cuda.is_available() and default_config['chosen_device'] == 'cuda':
                release_gpu(gpu_lock, gpu_status, nv_gpu_id)
            with lock:
                study.tell(trial, val_loss)
                completed_trials.append(trial.number)
            return val_loss

        except Exception as e:
            logger.error(f"Trial {trial_id} with hyperparameters {hyperparameters} failed with exception: {e}", exc_info=True)
            time.sleep(1)
            if CONST.DEBUG and trial.number - 1 not in completed_trials and trial.number > first_trial_nr:
                logger.debug(f"Trial {trial_id} waiting for Trial {trial_id - 1}...")
            while trial.number - 1 not in completed_trials and trial.number > first_trial_nr:
                if CONST.DEBUG:
                    time.sleep(3)
                    logger.debug(f"Trial {trial_id} still waiting for Trial {trial_id - 1}...")
                time.sleep(0.5)  # Wait until the previous trial finishes before proceeding
            if torch.cuda.is_available() and default_config['chosen_device'] == 'cuda':
                release_gpu(gpu_lock, gpu_status, nv_gpu_id)
            with lock:
                study.tell(trial, state=TrialState.FAIL)
                completed_trials.append(trial.number)
            logger.error(f"Trial {trial_id} failed and process ended.")
            return np.nan
        finally:
            logger.debug(f"Stopping QueueListener in trial {trial_id}")
            logging.getLogger().handlers.clear()

    def hyperparameter_tuning(self):
        """
        Runs hyperparameter tuning using Optuna with multiprocessing.

        Ensures:
        - Each trial runs in a separate process to prevent GPU memory conflicts.
        - The least utilized GPU is selected dynamically for each trial.

        Returns:
            dict: The best hyperparameters found during tuning.
        """

        hyperparameter_config = self.config['experiment']['hyperparameter_tuning']
        n_trials = hyperparameter_config['n_trials']
        if self.device.type=='cuda':
            jobs_per_GPU=hyperparameter_config.get('jobs_per_gpu',2)
            max_jobs=jobs_per_GPU*torch.cuda.device_count()
            if hyperparameter_config.get('n_jobs', torch.cuda.device_count()) > max_jobs:
                self.logger.info(f"Reducing n_jobs from {hyperparameter_config.get('n_jobs', torch.cuda.device_count())} to {max_jobs} to avoid GPU memory conflicts")
                n_jobs=max_jobs
            elif hyperparameter_config.get('n_jobs', torch.cuda.device_count()) ==0:
                self.logger.error("n_jobs is set to 0, setting it to 1")
                hyperparameter_config['n_jobs']=1
                n_jobs=1
            elif hyperparameter_config.get('n_jobs', torch.cuda.device_count()) <0:
                self.logger.info(f"Performing max trials per GPU which is {jobs_per_GPU} and in total {max_jobs}")
                n_jobs = max_jobs
        else:
            if hyperparameter_config.get('n_jobs', cpu_count()) <0:
                self.logger.info(f"Performing max trials per CPU which is {cpu_count()}")
                n_jobs = cpu_count()
            elif hyperparameter_config.get('n_jobs', cpu_count()) ==0:
                self.logger.error("n_jobs is set to 0, setting it to 1")
                hyperparameter_config['n_jobs']=1
                n_jobs=1
            else:
                n_jobs = min(hyperparameter_config.get('n_jobs', cpu_count()), cpu_count())
        self.logger.info(f"Using {n_jobs} parallel jobs for hyperparameter tuning")

        #load the dataset once to make sure it exists / is generated
        _=self.dataset_class(self.config['dataset'], data_split='train', device=self.device)

        # Set up paths
        base_path = Path(self.path_results, 'hyperparameter_tuning')
        base_path.mkdir(parents=True, exist_ok=True)
        db_path = Path(base_path, "optuna_study.db")


        self.logger.info(f"Starting hyperparameter tuning with {n_trials} trials and {n_jobs} parallel jobs")
        self._log_time("Start Hyperparameter Tuning")

        # Get PyTorch-to-NVIDIA GPU mapping
        gpu_mapping = get_pytorch_nvidia_gpu_mapping() if torch.cuda.is_available() else {}

        if hyperparameter_config['use_pruner']:
            pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=hyperparameter_config['min_epochs'],
                                                            reduction_factor=hyperparameter_config['reduction_factor'])
        else:
            pruner = optuna.pruners.NopPruner()

        # Initialize the Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            study_name=f"Hyperparameter Optimization for {self.model_name} on {self.dataset_name}",
            storage=f"sqlite:///{db_path}",
            load_if_exists=True,
            pruner=pruner
        )

        # Identify the next trial number in case the study is restarted mid-way
        # If the study is resumed, we need to determine the first available trial number.
        existing_trials = study.get_trials()
        first_trial_number = max([trial.number for trial in existing_trials] or [0]) + 1

        # Use a Manager to create shared memory resources across multiple processes.
        # This ensures synchronization between different trials running in parallel.
        with Manager() as manager:
            completed_trials = manager.list([-1])  # Shared list tracking completed trials.
            lock = manager.Lock()  # Lock for synchronizing GPU selection across processes.
            jobs_per_gpu = hyperparameter_config.get('jobs_per_gpu', 2)
            gpu_status = manager.dict({i: jobs_per_gpu for i in range(torch.cuda.device_count())})
            gpu_lock = manager.Lock()

            # Create a partial function with fixed arguments for running a trial.
            # This allows each process to execute `self.run_trial` with the necessary parameters.
            run_trial_partial = functools.partial(
                self.run_trial,
                config=self.config,
                model_class_name=self.model_name,
                dataset_class_name=self.dataset_name,
                default_config=self.get_model_defaults(),
                study_db_path=db_path,
                gpu_mapping=gpu_mapping,
                lock=lock,
                completed_trials=completed_trials,
                first_trial_nr=first_trial_number,
                log_queue=self.log_queue,
                seed=self.seed,
                gpu_lock=gpu_lock,
                gpu_status=gpu_status

            )

            # Create a multiprocessing pool with the 'spawn' context.
            # Each trial runs in a separate process to prevent GPU memory conflicts.
            with mp.get_context('spawn').Pool(processes=n_jobs, maxtasksperchild=1) as pool:
                results = []
                num_new_trials = n_trials - len(study.trials)  # Calculate remaining trials.

                # Start new trials one by one with a short delay to avoid GPU congestion.
                for i in range(num_new_trials):
                    time.sleep(5)  # Introduce a small delay to avoid overloading resources.
                    self.logger.debug(f"Main process is starting trial index {i}")

                    # Submit a new trial to the process pool.
                    # This launches a new worker process for each trial (up to n_jobs in parallel).
                    r = pool.apply_async(run_trial_partial)

                    # Store the results to keep track of running processes.
                    results.append(r)

                # Wait for all trials to complete.
                for r in results:
                    # .get() blocks execution until the respective trial process is finished.
                    val_loss = r.get()
                    self.logger.debug(f"Main process got result val_loss={val_loss}")

            # Clean up the multiprocessing pool.
            pool.close()  # Prevents new processes from being submitted.
            pool.join()  # Ensures all processes complete before continuing.

        # Save best hyperparameters
        self.logger.info("Best hyperparameters found: ", study.best_params)
        with open(Path(base_path, "best_hyperparameters.yaml"), 'w') as file:
            yaml.dump(study.best_params, file)

        self._log_time("End Hyperparameter Tuning")
        trials_df = study.trials_dataframe()
        trials_df.to_csv(Path(base_path, "hyperparameter_search.csv"), index=False)

        return study.best_params
    def _log_time(self, name):
        with open(Path(self.path_results, "timestamps.txt"), 'a') as file:
            file.write(f'\n{name} : {datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

    def load_best_hparams(self):
        if not Path(self.path_results, "hyperparameter_tuning", "best_hyperparameters.yaml").exists():
            raise FileNotFoundError(f"Best hyperparameters not found at {Path(self.path_results, 'hyperparameter_tuning', 'best_hyperparameters.yaml')}")
        with open(Path(self.path_results, "hyperparameter_tuning", "best_hyperparameters.yaml"), 'r') as file:
            best_hyperparameters = yaml.safe_load(file)
        return best_hyperparameters

    def get_untrained_model(self, best_hyperparameters=None):
        """
        Returns an untrained model with the best hyperparameters found during tuning.

        Args:
            best_hyperparameters (dict, optional): The best hyperparameters found during tuning.

        Returns:
            model: An untrained model with the best hyperparameters.
            config: The configuration used to initialize the model.
        """
        config = self.get_model_defaults()
        if best_hyperparameters is not None:
            self.logger.info("Using best hyperparameters found during tuning")
            config['model'].update(best_hyperparameters)
        return self.model_class(config),config

    def final_training(self, best_hyperparameters=None,seed=None):
        """
        Trains the model using the best hyperparameters and tracks the training and validation loss across epochs.
        Saves the model with the best validation loss.

        Args:
            best_hyperparameters (dict, optional): The best hyperparameters found during tuning.
            seed (int, optional): The random seed to use for reproducibility. If None, the seed from the config is used.

        Returns:
            model: The trained model with the best validation performance.
        """
        self.logger.info("Starting final training")
        self._log_time('Final Training Start')

        if seed is not None:
            self.set_seed(seed)
            warn_msg = "Seed from config (" + str(
                self.config['experiment']['seed']) + ") overwritten by seed from experiment(" + str(seed) + ")"
            warnings.warn(warn_msg)
            self.logger.info(f"Using seed {seed} for final training")

        # Load the data in training and validation splits
        dataloaders = self.load_data(shuffle_train=False)
        self.model,config = self.get_untrained_model(best_hyperparameters)

        #store the config that has been used for the model
        with open(Path(self.path_results, "model_config.yaml"), 'w') as file:
            yaml.dump(config, file)

        # Initialize lists for tracking losses and storing model state_dicts
        train_losses = []
        val_losses = []
        model_checkpoints = []  # Stores (state_dict, val_loss, epoch) tuples

        total_epochs = self.config['experiment']['num_epochs']
        best_val_loss = float("inf")
        best_checkpoint = (copy.deepcopy(self.model.state_dict()), float("inf"), 0)
        for epoch in range(total_epochs):
            # Perform training step and log training loss
            train_loss = self.model.train_step(dataloaders['train'], return_loss=True)
            train_losses.append(train_loss)

            # Perform validation step and log validation loss
            val_loss = self.model.validate_step(dataloaders['val'])
            val_losses.append(val_loss)

            # Save best model state_dict with associated validation loss and epoch number
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint=(copy.deepcopy(self.model.state_dict()), val_loss, epoch + 1)
            if (epoch+1)%10 == 0 or epoch == total_epochs - 1:
                self.logger.info(f"Epoch {epoch + 1}/{total_epochs}: Train Loss: {train_loss:.3e}, Val Loss: {val_loss:.3e}")


        best_state_dict, best_val_loss, best_epoch = best_checkpoint
        self.model.load_state_dict(best_state_dict)

        # Save the best model in the model checkpoints folder
        best_model_path = Path(self.path_results, 'model_checkpoints', 'best_model.pth')
        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), best_model_path)

        # Log the epoch number of the best model for later reference
        with open(Path(self.path_results,'model_checkpoints', "best_model_info.txt"), 'w') as file:
            file.write(f"Best model found at epoch: {best_epoch}\n")
            file.write(f"Validation loss at best epoch: {best_val_loss:.4e}\n")

        # Also save the last model
        last_model_path = Path(self.path_results, 'model_checkpoints', 'last_epoch.pth')
        torch.save(self.model.state_dict(), last_model_path)


        # Save the training and validation losses for later analysis
        losses_path = Path(self.path_results, 'training_logs')
        losses_path.mkdir(parents=True, exist_ok=True)
        with open(losses_path / 'train_losses.yaml', 'w') as file:
            yaml.dump(train_losses, file)
        with open(losses_path / 'val_losses.yaml', 'w') as file:
            yaml.dump(val_losses, file)

        self.logger.info("Final training completed. Best model saved.")
        self._log_time('Final Training End')
        return self.model

    def save_model(self, model, filename):
        model_path = Path(self.path_results, 'model_checkpoints', filename)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)


    def load_best_model(self,return_best_epoch=False):
        """
        Loads the best model if it has been trained and stored
        :param defaults:
        :return:
        """
        #check if model exists
        best_model_path = Path(self.path_results, 'model_checkpoints', 'best_model.pth')
        if not best_model_path.exists():
            raise FileNotFoundError(f"Best model not found at {best_model_path}")
        if not Path(self.path_results, "model_config.yaml").exists():
            raise FileNotFoundError(f"Model config not found at {Path(self.path_results, 'model_config.yaml')}")
        #load the model config
        with open(Path(self.path_results, "model_config.yaml"), 'r') as file:
            config = yaml.safe_load(file)
        #adjust chosen device
        config['chosen_device']=self.device

        #init model and load state dict
        self.model = self.model_class(config)
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.model.device, weights_only=True))


        if return_best_epoch:
            with open(Path(self.path_results,'model_checkpoints', "best_model_info.txt"), 'r') as file:
                best_epoch=int(file.readline().split(":")[1].strip())
            return self.model,best_epoch

        return self.model

    def performance_measurement(self, model, dataset):
        """
        Measures the performance of the model.

        Args:
            model: The trained model.
            dataset: The dataset for evaluation.

        Returns:
            y_pred: The model's predictions on the dataset.
        """
        self.logger.info("Measuring performance")
        y_pred = model.predict(dataset)
        return y_pred

    def write_results(self, results: dict, filename: str):
        """
        Writes the results to a JSON file in a readable format.

        Args:
            results (dict): The results to write.
            filename (str): The filename to write to.
        """

        # Convert numpy types to native Python types for JSON compatibility
        def convert_to_python(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert arrays to lists
            elif isinstance(obj, (np.generic, np.number)):
                return obj.item()  # Convert numpy scalars to Python scalars
            elif isinstance(obj, dict):
                return {k: convert_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python(v) for v in obj]
            else:
                return obj

        readable_results = convert_to_python(results)

        # Write the results to a JSON file
        with open(Path(self.path_results, filename+'.json'), 'w') as file:
            json.dump(readable_results, file, indent=4)

    def load_data(self, return_dataloader=True,shuffle_train=True,batch_size=None):
        """
        Loads the data for the experiment.
        Args:
            return_dataloader (bool, optional): If True, return the DataLoader. Defaults to True.
            shuffle_train (bool, optional): If True, shuffle the training data. Defaults to True.
            batch_size (int, optional): The batch size to use for the DataLoader. If None, the default batch size of the experiment is used.
        Returns:
            dict: Dictionary containing the data.
        """
        train_dataset = self.dataset_class(self.config['dataset'], data_split='train',device=self.device)
        val_dataset = self.dataset_class(self.config['dataset'], data_split='val',device=self.device)
        test_dataset = self.dataset_class(self.config['dataset'], data_split='test',device=self.device)


        if return_dataloader:
            if batch_size is None:
                batch_size=self.config['experiment']['batch_size']
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=shuffle_train,
                                      collate_fn=self.dataset_class.collate_fn)
            val_loader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    collate_fn=self.dataset_class.collate_fn)
            test_loader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     collate_fn=self.dataset_class.collate_fn)
            return {"train": train_loader, "val": val_loader, "test": test_loader}
        return {"train": train_dataset, "val": val_dataset, "test": test_dataset}



    def get_model_defaults(self):
        """
        Extracts the default values from a nested configuration dictionary.

        This function recursively traverses the 'model' section of self.config and
        replaces any dictionary that contains a "default" key with its default value,
        while preserving the overall nested structure.

        Returns:
            dict: A configuration dictionary with only the default values.
        """

        def recursive_defaults(d):
            # If d is a dictionary, iterate over its items.
            if isinstance(d, dict):
                out = {}
                for key, value in d.items():
                    # If the value is a dict with a "default" key, use that.
                    if isinstance(value, dict) and "default" in value:
                        out[key] = value["default"]
                    else:
                        # Otherwise, recurse into the value.
                        out[key] = recursive_defaults(value)
                return out
            else:
                # If d is not a dictionary, return it unchanged.
                return d

        config = self.config.copy()
        # Replace the "model" subdictionary with its defaults
        if "model" in config:
            config["model"] = recursive_defaults(config["model"])
        return config






    def performance_evaluation(self, y_true, y_pred, metrics):
        """
        Evaluates the performance of the model using specified metrics.

        Args:
            y_true: Ground truth labels.
            y_pred: Predictions from the model.
            metrics (list): List of metric functions to use for evaluation.

        Returns:
            dict: The evaluation results for each metric.
        """
        self.logger.info("Evaluating performance")
        return self.scoring(y_true, y_pred, metrics)

    @abstractmethod
    def run_experiment(self, default_parameters=False):
        """
        Abstract method to be implemented by derived classes.
        This method defines the main flow for running the experiment.
        """
        pass

    def store_best_model(self, seed, model):
        """
        Stores the best model for a given seed.

        Args:
            seed (int): The random seed used during training.
            model: The best model obtained for this seed.
        """
        if seed not in self.best_models:
            self.best_models[seed] = []
        self.best_models[seed].append(model)


    def load_train_losses(self):
        """
        Load the lossed from the final training
        :return: train_losses, val_losses
        """
        #check if the files exists
        train_losses_path = Path(self.path_results, 'training_logs', 'train_losses.yaml')
        val_losses_path = Path(self.path_results, 'training_logs', 'val_losses.yaml')
        if not train_losses_path.exists():
            raise FileNotFoundError(f"Train losses not found at {train_losses_path}")

        if not val_losses_path.exists():
            raise FileNotFoundError(f"Val losses not found at {val_losses_path}")

        #load the losses
        with open(train_losses_path, 'r') as file:
            train_losses = yaml.safe_load(file)

        with open(val_losses_path, 'r') as file:
            val_losses = yaml.safe_load(file)

        return train_losses, val_losses