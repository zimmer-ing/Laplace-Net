import Constants as const
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import random
import os
import subprocess
import time
import hashlib


def dataloader_hash(dataloader):
    """
    Computes a hash for a DataLoader to detect changes in its data.

    - Handles batches containing Tensors, Strings, and Dictionaries.
    - Converts Tensors to byte format for hashing.
    - Encodes Strings and Dictionaries to ensure consistency.

    Args:
        dataloader (DataLoader): A PyTorch DataLoader.

    Returns:
        str: A SHA-256 hash representing the data in the DataLoader.
    """
    hasher = hashlib.sha256()

    for batch in dataloader:
        batch_tensors = []

        # Handle different batch formats (Tuple, List, Dict, Tensor)
        if isinstance(batch, torch.Tensor):
            batch_tensors.append(batch.flatten())
        elif isinstance(batch, (tuple, list)):  # If batch is a tuple/list
            for item in batch:
                if isinstance(item, torch.Tensor):
                    batch_tensors.append(item.flatten())
                elif isinstance(item, str):  # Convert strings to bytes for hashing
                    hasher.update(item.encode())
                elif isinstance(item, dict):  # Convert dicts to sorted string for consistent hashing
                    hasher.update(str(sorted(item.items())).encode())

        # If batch contains tensors, convert them to bytes and update hash
        if batch_tensors:
            batch_data = torch.cat(batch_tensors).numpy().tobytes()
            hasher.update(batch_data)

    return hasher.hexdigest()

def dl_is_shuffle(dataloader: DataLoader) -> bool:
    """
    Check if a DataLoader is shuffled or not.

    Args:
        dataloader (DataLoader): The DataLoader to check.

    Returns:
        bool: True if the DataLoader is shuffled, False otherwise.
    """
    if isinstance(dataloader.sampler, RandomSampler):
        return True
    elif isinstance(dataloader.sampler, SequentialSampler):
        return False
    elif hasattr(dataloader, 'batch_sampler') and isinstance(dataloader.batch_sampler.sampler, RandomSampler):
        return True
    else:
        return False


def change_to_sequential_and_return_new(dataloader: DataLoader) -> DataLoader:
    """
    Change a DataLoader to use SequentialSampler and return a new DataLoader.

    Args:
        dataloader (DataLoader): The DataLoader to change.

    Returns:
        DataLoader: A new DataLoader with SequentialSampler.
    """
    new_dataloader = DataLoader(
        dataset=dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        sampler=SequentialSampler(dataloader.dataset),
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        generator=dataloader.generator,
        prefetch_factor=dataloader.prefetch_factor,
        persistent_workers=dataloader.persistent_workers
    )

    if not dl_is_shuffle(new_dataloader):
        print("Successfully created a new DataLoader in sequential mode.")
    else:
        print("Failed to create a new DataLoader in sequential mode.")

    return new_dataloader


def ensure_sequential_dataloader(dataloader: DataLoader) -> DataLoader:
    """
    Ensure that a DataLoader is in sequential mode by creating a new DataLoader with SequentialSampler.

    Args:
        dataloader (DataLoader): The DataLoader to check and possibly change.

    Returns:
        DataLoader: A new DataLoader with SequentialSampler if the original was shuffled, otherwise the original.
    """
    if not dl_is_shuffle(dataloader):
        return dataloader
    print("Dataloader was in shuffle mode for predictions. Changing to sequential mode.")

    new_dataloader = change_to_sequential_and_return_new(dataloader)
    return new_dataloader


from typing import List, Union
import torch

def concatenate_batches(batches: List[Union[torch.Tensor, list, dict]]) -> Union[torch.Tensor, list, dict]:
    """
    Concatenate a list of batched tensors and handle auxiliary (non-tensor) data.

    Args:
        batches (List[Union[torch.Tensor, list, dict]]): A list containing tensors, lists, or dictionaries.

    Returns:
        Union[torch.Tensor, list, dict]: Concatenated tensor if tensor data is provided, otherwise
                                         concatenated lists or merged dictionaries as appropriate.
    """
    if all(isinstance(batch, torch.Tensor) for batch in batches):
        # Concatenate tensor data
        concatenated = torch.cat(batches, dim=0)
        total_batches_times_batch, time, features = concatenated.shape
        return concatenated.view(-1, features)

    elif all(isinstance(batch, np.ndarray) for batch in batches):
        # Concatenate numpy data
        concatenated = np.concatenate(batches, axis=0)
        return concatenated

    elif all(isinstance(batch, list) for batch in batches):
        # Handle lists by concatenating them into a single list
        batches=sum(batches, [])  # Flatten the list of lists
        if all(isinstance(item, torch.Tensor) for item in batches):
            return torch.cat(batches, dim=0)
        elif all(isinstance(item, np.ndarray) for item in batches):
            return np.concatenate(batches, axis=0)
        return batches  # Flatten the list of lists

    elif all(isinstance(batch, dict) for batch in batches):
        # Handle dictionaries by concatenating each key's values
        result = {}
        for key in batches[0].keys():
            result[key] = concatenate_batches([batch[key] for batch in batches])
        return result

    else:
        raise TypeError("All items in batches must be of the same type (torch.Tensor, list, or dict).")

def get_pytorch_gpu_uuids():
    """
    Retrieves GPU UUIDs and their corresponding PyTorch CUDA indices.

    This function uses PyTorch's `torch.cuda.get_device_properties()` to extract
    the UUIDs of all available CUDA devices.

    Returns:
        dict: A dictionary mapping GPU UUIDs to PyTorch device IDs.
              Example: {"UUID_1": 0, "UUID_2": 1, ...}
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
        return {}

    return {str(torch.cuda.get_device_properties(i).uuid): i for i in range(torch.cuda.device_count())}


def get_nvml_gpu_uuids():
    """
    Retrieves GPU UUIDs from NVIDIA-SMI and removes formatting artifacts.

    This function runs `nvidia-smi` via subprocess to obtain GPU UUIDs and their corresponding
    NVIDIA device IDs, then cleans the output by removing "GPU-" prefixes and unwanted characters.

    Returns:
        dict: A dictionary mapping GPU UUIDs to NVIDIA GPU device IDs.
              Example: {"UUID_1": 0, "UUID_2": 1, ...}
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,nounits,noheader"],
            stdout=subprocess.PIPE, text=True, check=True
        )
    except FileNotFoundError:
        print("NVIDIA-SMI not found. Ensure you have NVIDIA drivers installed.")
        return {}

    return {gpu[1].replace("GPU-", "").strip("'").strip(): int(gpu[0]) for gpu in
            [line.split(", ") for line in result.stdout.strip().split("\n")]}


def get_pytorch_nvidia_gpu_mapping():
    """
    Creates a mapping between NVIDIA GPU IDs and PyTorch CUDA IDs based on UUIDs.

    Since PyTorch and NVIDIA-SMI may assign different device indices, this function
    ensures a correct mapping between the two by comparing GPU UUIDs.

    Returns:
        dict: A dictionary mapping NVIDIA GPU IDs to their corresponding PyTorch CUDA IDs.
              Example: {0: 2, 1: 3, 2: 4, ...}
    """
    nvml_uuids = get_nvml_gpu_uuids()  # Retrieve NVIDIA GPU UUIDs
    pytorch_uuids = get_pytorch_gpu_uuids()  # Retrieve PyTorch GPU UUIDs

    mapping = {}
    for uuid, nv_id in nvml_uuids.items():
        if uuid in pytorch_uuids:
            mapping[nv_id] = pytorch_uuids[uuid]

    # Print the mapping for debugging
    if const.DEBUG:
        print("\n--- GPU Mapping (UUID-based) ---")
        for nv_id, pt_id in mapping.items():
            print(f"PyTorch ID {pt_id} -> NVIDIA GPU ID {nv_id} ({torch.cuda.get_device_name(pt_id)})")

    return mapping


def get_least_used_gpu(logger=None):
    """
    Identifies the GPU with the most available memory.

    Uses NVML to query the free memory on all GPUs and selects the GPU with the highest free memory.
    If NVML fails, it defaults to PyTorch GPU ID 0.

    Returns:
        int: The PyTorch GPU ID with the most available memory.
    """
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown

        nvmlInit()  # Initialize NVML before making any calls

        free_memory = {}
        for i in range(torch.cuda.device_count()):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            free_memory[i] = info.free

        # Select the GPU with the highest free memory
        best_gpu = max(free_memory, key=free_memory.get)
        if const.VERBOSE:
            log_message(f"Selected GPU: {best_gpu} (Free memory: {free_memory[best_gpu] / 1e6:.2f} MB)",logger)
        return best_gpu

    except Exception as e:
        print(f"Error in NVML: {e}")
        return 0  # Fallback to GPU 0 if NVML fails

    finally:
        try:
            nvmlShutdown()  # Always shut down NVML when done
        except:
            pass


def get_gpu_status(logger):
    """
    Retrieves GPU utilization and free memory using NVML.

    Returns:
        list of tuples: (GPU ID, utilization percentage, free memory in MB).
    """
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlShutdown

        nvmlInit()
        gpu_status = []
        for i in range(torch.cuda.device_count()):
            handle = nvmlDeviceGetHandleByIndex(i)
            utilization = nvmlDeviceGetUtilizationRates(handle).gpu  # GPU utilization percentage
            free_memory = nvmlDeviceGetMemoryInfo(handle).free / 1e6  # Convert to MB
            gpu_status.append((i, utilization, free_memory))

        nvmlShutdown()
        return gpu_status

    except Exception as e:
        log_message(f"Error retrieving GPU status: {e}",logger)
        return []

def release_gpu(gpu_lock, gpu_status, gpu_id):
    with gpu_lock:
        gpu_status[gpu_id] += 1

def get_best_available_gpu(gpu_lock, gpu_status, logger=None):
    """
    Selects the best available GPU based on the least utilization and most available memory.
    :param gpu_lock:
    :param gpu_status:
    :param logger:
    :return:
    """
    with gpu_lock:
        free_gpus = {gpu: slots for gpu, slots in gpu_status.items() if slots > 0}
        if not free_gpus:
            if logger:
                logger.warning("No free GPU found. Waiting for a slot to become available.")
            time.sleep(5)
            return get_best_available_gpu(gpu_lock, gpu_status, logger)
        best_gpu = min(free_gpus, key=free_gpus.get)
        gpu_status[best_gpu] -= 1
    return best_gpu

def get_best_available_gpu_random(gpu_lock,logger=None):
    """
    Selects the best GPU based on the least utilization and most available memory.
    Uses a multiprocessing lock to prevent multiple processes from selecting the same GPU.

    Args:
        gpu_lock (multiprocessing.Lock): A lock to prevent concurrent selection of the same GPU.

    Returns:
        int: PyTorch GPU ID of the selected GPU.
    """
    with gpu_lock:  # Ensure only one process at a time checks the GPUs
        gpu_status = get_gpu_status(logger)

        if not gpu_status:
            log_message("No valid GPU status retrieved. Defaulting to GPU 0.",logger)
            return 0  # Default to first GPU

        # Sort by utilization first, then by free memory
        sorted_gpus = sorted(gpu_status, key=lambda x: (x[1], -x[2]))
        log_message(f"GPU status: {sorted_gpus}",logger,"debug")
        # Select the GPU with the lowest utilization
        best_gpu = sorted_gpus[0][0]
        best_utilization = sorted_gpus[0][1]
        best_memory = sorted_gpus[0][2]

        # If multiple GPUs have the same utilization, choose randomly among them
        candidate_gpus = [g[0] for g in sorted_gpus if g[1] == best_utilization]
        log_message(f"Candidate GPUs: {candidate_gpus}",logger,"debug")
        if len(candidate_gpus) > 1:

            best_gpu = random.choice(candidate_gpus)
            if const.DEBUG:
                log_message(f"More than one GPU is best. GPU {best_gpu} was randomly selected from {candidate_gpus}.",logger,"debug")
        if const.VERBOSE:
            log_message(f"Selected GPU: {best_gpu} (Utilization: {best_utilization}%, Free Memory: {best_memory:.2f} MB)",logger,"debug")
        return best_gpu

def log_message(message, logger=None, level="info"):
    """
    Logs a message using the provided logger or prints it if no logger is given.

    Args:
        message (str): The message to log or print.
        logger (logging.Logger, optional): The logger to use. Defaults to None.
        level (str): Logging level, e.g., "info", "debug", "warning". Defaults to "info".
    """
    if logger:
        log_method = getattr(logger, level, logger.info)
        log_method(message)
    else:
        print(message)

if __name__ == "__main__":

    # Test the functions
    dataset = torch.rand(100, 2)
    dataloader_shuffled = DataLoader(dataset, batch_size=10, shuffle=True)
    dataloader_sequential = DataLoader(dataset, batch_size=10, shuffle=False)

    print("Shuffled DataLoader:", dl_is_shuffle(dataloader_shuffled))
    print("Sequential DataLoader:", dl_is_shuffle(dataloader_sequential))

    # Example usage
    print("DataLoader is shuffle before :", dl_is_shuffle(dataloader_shuffled))
    dataloader_sequential = ensure_sequential_dataloader(dataloader_shuffled)
    print("DataLoader is shuffle after :", dl_is_shuffle(dataloader_sequential))

    # Test concatenate_batches function
    batch1 = torch.randn(1, 5, 3)
    batch2 = torch.randn(2, 5, 3)

    result_single = concatenate_batches([batch1])
    result_multiple = concatenate_batches([batch1, batch2])

    print("Single batch result shape:", result_single.shape)
    print("Multiple batches result shape:", result_multiple.shape)


    ##### gpu mapping
    # Initialize GPU mappings to ensure correct device assignments
    gpu_mapping = get_pytorch_nvidia_gpu_mapping()

    # Ensure all GPUs are visible to PyTorch
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(torch.cuda.device_count()))
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Select the least used GPU based on available memory
    import multiprocessing
    lock = multiprocessing.Lock()
    best_nv_gpu = get_best_available_gpu(lock)

    if best_nv_gpu in gpu_mapping:
        best_pytorch_gpu = gpu_mapping[best_nv_gpu]  # Map NVIDIA GPU ID to PyTorch ID
        torch.cuda.set_device(best_pytorch_gpu)  # Assign the correct GPU in PyTorch
        device = torch.device(f"cuda:{best_pytorch_gpu}")
        print(f"Using CUDA device: {device} (Mapped from NVIDIA GPU {best_nv_gpu})")
    else:
        device = torch.device("cpu")
        print("No GPU available, running on CPU.")

    # Verify the selected GPU in PyTorch
    print(
        f"PyTorch reports using: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")

