"""This file contains functions for loading the dataset"""

import math
import os
import pickle
import subprocess
from typing import List, Tuple, Any, Optional, Sequence
import h5py
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from dataset import TabularDataset, TextDataset, load_agnews, TabularDatasetTabPFN
from trainers.fast_train import get_batches, load_cifar10_data


META_KEYS = ("chunks", "compression", "compression_opts", "shuffle", "fletcher32")


def _copy_sampling_metadata(src_dataset):
    """Copy dataset creation kwargs (compression/chunks) to preserve layout."""
    kwargs = {}
    for key in META_KEYS:
        value = getattr(src_dataset, key, None)
        if value is not None:
            kwargs[key] = value
    return kwargs


def _sanitize_chunks(chunks, target_shape):
    if not chunks:
        return None
    sanitized = []
    for dim_len, chunk_len in zip(target_shape, chunks):
        if dim_len == 0:
            return None
        sanitized.append(max(1, min(int(dim_len), int(chunk_len))))
    return tuple(sanitized)


def _write_rows(dst_dataset, dest_slice, rows_slice):
    if len(dest_slice) == 0:
        return
    dest_slice = np.asarray(dest_slice, dtype=np.int64)
    if len(dest_slice) == 1:
        dst_dataset[int(dest_slice[0])] = rows_slice[0]
        return
    if np.all(dest_slice[1:] == dest_slice[:-1] + 1):
        start = int(dest_slice[0])
        end = int(dest_slice[-1]) + 1
        dst_dataset[start:end] = rows_slice
        return
    scatter_order = np.argsort(dest_slice, kind="mergesort")
    dest_sorted = dest_slice[scatter_order]
    rows_sorted = rows_slice[scatter_order]
    dst_dataset[dest_sorted] = rows_sorted


def _stream_copy_rows(src_dataset, dst_dataset, sorted_indices, dest_order):
    if len(sorted_indices) == 0:
        return

    num_rows = src_dataset.shape[0]
    chunk_hint = getattr(src_dataset, "chunks", None)
    if chunk_hint and len(chunk_hint) > 0 and chunk_hint[0]:
        chunk_len = int(chunk_hint[0])
    else:
        chunk_len = max(1, min(1 << 15, num_rows))
    chunk_len = max(1, min(chunk_len, num_rows))
    max_span = max(chunk_len, min(num_rows, chunk_len * 8))

    ptr = 0
    total = len(sorted_indices)
    while ptr < total:
        idx = int(sorted_indices[ptr])
        chunk_start = (idx // chunk_len) * chunk_len
        chunk_end = min(chunk_start + chunk_len, num_rows)

        chunk_stop = ptr
        while True:
            while chunk_stop < total and sorted_indices[chunk_stop] < chunk_end:
                chunk_stop += 1

            if chunk_stop == total:
                break

            span = sorted_indices[chunk_stop] - chunk_start
            if span < max_span and chunk_end < num_rows:
                chunk_end = min(chunk_end + chunk_len, num_rows)
                continue
            break

        data_chunk = src_dataset[chunk_start:chunk_end]
        relative_indices = sorted_indices[ptr:chunk_stop] - chunk_start
        sampled_rows = data_chunk[relative_indices]
        dest_slice = dest_order[ptr:chunk_stop]
        _write_rows(dst_dataset, dest_slice, sampled_rows)

        ptr = chunk_stop


def save_sampled(src_path, dst_path, indices):
    indices = np.asarray(indices)
    if indices.ndim != 1:
        raise ValueError("indices must be a 1-D sequence")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("indices must be integers")
    indices = indices.astype(np.int64, copy=False)

    sort_order = np.argsort(indices, kind="mergesort")
    sorted_indices = indices[sort_order]

    with h5py.File(src_path, "r") as fin, h5py.File(dst_path, "w") as fout:
        if "X" not in fin:
            raise KeyError("Source file must contain dataset 'X' to infer sample size")
        num_rows = fin["X"].shape[0]

        if np.any(indices < 0) or np.any(indices >= num_rows):
            raise IndexError("indices contain values outside the dataset range")

        for key, data in fin.items():
            if data.shape and data.shape[0] == num_rows:
                target_shape = (len(indices),) + data.shape[1:]
                creation_kwargs = _copy_sampling_metadata(data)
                chunks = creation_kwargs.get("chunks")
                if chunks:
                    sanitized = _sanitize_chunks(chunks, target_shape)
                    if sanitized:
                        creation_kwargs["chunks"] = sanitized
                    else:
                        creation_kwargs.pop("chunks", None)
                if target_shape[0] == 0:
                    creation_kwargs.pop("chunks", None)
                dst_dataset = fout.create_dataset(
                    key,
                    shape=target_shape,
                    dtype=data.dtype,
                    **creation_kwargs,
                )
                _stream_copy_rows(data, dst_dataset, sorted_indices, sort_order)
            else:
                fout.create_dataset(key, data=data[()])

        fout.create_dataset("sampled_indices", data=indices)


def verify_sampled_files(src_path, sampled_paths, batch_size=1024):
    """
    Ensure that sampled HDF5 files store data identical to the source rows referenced
    by their respective 'sampled_indices' datasets.
    """
    if isinstance(sampled_paths, (str, bytes, os.PathLike)):
        sampled_paths = [sampled_paths]

    with h5py.File(src_path, "r") as src_file:
        if "X" not in src_file:
            raise KeyError("Source file must contain dataset 'X' to infer row count.")
        num_rows = src_file["X"].shape[0]

        for sampled_path in sampled_paths:
            with h5py.File(sampled_path, "r") as sampled_file:
                if "sampled_indices" not in sampled_file:
                    raise KeyError(
                        f"'sampled_indices' missing from sampled file {sampled_path}"
                    )
                indices = np.asarray(
                    sampled_file["sampled_indices"][()], dtype=np.int64
                )
                if indices.ndim != 1:
                    raise ValueError(
                        f"'sampled_indices' must be 1-D in sampled file {sampled_path}"
                    )
                if np.any(indices < 0) or np.any(indices >= num_rows):
                    raise ValueError(
                        f"Indices in {sampled_path} fall outside the source dataset."
                    )

                effective_batch = max(1, min(int(batch_size), len(indices)))

                for key, sampled_dataset in sampled_file.items():
                    if key == "sampled_indices":
                        continue
                    if key not in src_file:
                        raise KeyError(
                            f"Key '{key}' exists in {sampled_path} but not in source file."
                        )
                    src_dataset = src_file[key]
                    if (
                        src_dataset.shape
                        and src_dataset.shape[0] == num_rows
                        and sampled_dataset.shape
                        and sampled_dataset.shape[0] == len(indices)
                    ):
                        _assert_axis_samples_match(
                            src_dataset,
                            sampled_dataset,
                            indices,
                            effective_batch,
                            key,
                            sampled_path,
                        )
                    else:
                        src_value = src_dataset[()]
                        sampled_value = sampled_dataset[()]
                        if not _arrays_equal(src_value, sampled_value):
                            raise AssertionError(
                                f"Dataset '{key}' mismatch between source and {sampled_path}"
                            )


def _arrays_equal(lhs, rhs):
    lhs_arr = np.asarray(lhs)
    rhs_arr = np.asarray(rhs)
    if lhs_arr.dtype.kind in ("f", "c") or rhs_arr.dtype.kind in ("f", "c"):
        return np.array_equal(lhs_arr, rhs_arr, equal_nan=True)
    try:
        return np.array_equal(lhs_arr, rhs_arr)
    except TypeError:
        return lhs == rhs


def _assert_axis_samples_match(
    src_dataset, sampled_dataset, indices, batch_size, key, sampled_path
):
    total = len(indices)
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch_indices = indices[start:end]
        src_slice = _fetch_rows(src_dataset, batch_indices)
        sampled_slice = sampled_dataset[start:end]
        if not _arrays_equal(src_slice, sampled_slice):
            raise AssertionError(
                f"Dataset '{key}' mismatch in rows {start}:{end} for {sampled_path}"
            )


def _fetch_rows(dataset, index_batch):
    if len(index_batch) == 0:
        return dataset[0:0]
    index_batch = np.asarray(index_batch, dtype=np.int64)
    if np.all(index_batch[1:] >= index_batch[:-1]):
        return dataset[index_batch]
    sorted_order = np.argsort(index_batch, kind="mergesort")
    sorted_rows = dataset[index_batch[sorted_order]]
    inverse_order = np.empty_like(sorted_order)
    inverse_order[sorted_order] = np.arange(len(sorted_order))
    return sorted_rows[inverse_order]


class InfinitelyIndexableDataset(Dataset):
    """
    A PyTorch Dataset that is able to index the given dataset infinitely.
    This is a helper class to allow easier and more efficient computation later when repeatedly indexing the dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be indexed repeatedly.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        # If the index is out of range, wrap it around
        return self.dataset[idx % len(self.dataset)]


def get_dataset(dataset_name: str, data_dir: str, logger: Any, **kwargs: Any) -> Any:
    """
    Function to load the dataset from the pickle file or download it from the internet.

    Args:
        dataset_name (str): Dataset name.
        data_dir (str): Indicate the log directory for loading the dataset.
        logger (logging.Logger): Logger object for the current run.

    Raises:
        NotImplementedError: If the dataset is not implemented.

    Returns:
        Any: Loaded dataset.
    """
    path = f"{data_dir}/{dataset_name}"
    if os.path.exists(f"{path}.pkl"):
        with open(f"{path}.pkl", "rb") as file:
            all_data = pickle.load(file)
        logger.info(f"Data loaded from {path}.pkl")
        if os.path.exists(f"{path}_population.pkl"):
            with open(f"{path}_population.pkl", "rb") as file:
                test_data = pickle.load(file)
            logger.info(f"Population data loaded from {path}_population.pkl")
    else:
        if dataset_name == "cifar10":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            all_data = torchvision.datasets.CIFAR10(
                root=path, train=True, download=True, transform=transform
            )
            test_data = torchvision.datasets.CIFAR10(
                root=path, train=False, download=True, transform=transform
            )
            # all_features = np.concatenate([all_data.data, test_data.data], axis=0)
            # all_targets = np.concatenate([all_data.targets, test_data.targets], axis=0)
            # all_data.data = all_features
            # all_data.targets = all_targets
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            logger.info(f"Save data to {path}.pkl")
            with open(f"{path}_population.pkl", "wb") as file:
                pickle.dump(test_data, file)
            logger.info(f"Save population data to {path}_population.pkl")
        elif dataset_name == "cifar10_canary":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            all_data = torchvision.datasets.CIFAR10(
                root=path.replace("cifar10_canary", "cifar10"),
                train=True,
                download=True,
                transform=transform,
            )
            labels = np.random.randint(10, size = len(all_data))
            all_data.targets = labels.tolist()
            test_data = torchvision.datasets.CIFAR10(
                root=path.replace("cifar10_canary", "cifar10"),
                train=False,
                download=True,
                transform=transform,
            )
            # all_features = np.concatenate([all_data.data, test_data.data], axis=0)
            # all_targets = np.concatenate([all_data.targets, test_data.targets], axis=0)
            # all_data.data = all_features
            # all_data.targets = all_targets
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            logger.info(f"Save data to {path}.pkl")
            with open(f"{path}_population.pkl", "wb") as file:
                pickle.dump(test_data, file)
            logger.info(f"Save population data to {path}_population.pkl")
        elif dataset_name == "cifar100":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            all_data = torchvision.datasets.CIFAR100(
                root=path, train=True, download=True, transform=transform
            )
            test_data = torchvision.datasets.CIFAR100(
                root=path, train=False, download=True, transform=transform
            )
            # all_features = np.concatenate([all_data.data, test_data.data], axis=0)
            # all_targets = np.concatenate([all_data.targets, test_data.targets], axis=0)
            # all_data.data = all_features
            # all_data.targets = all_targets
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            logger.info(f"Save data to {path}.pkl")
            with open(f"{path}_population.pkl", "wb") as file:
                pickle.dump(test_data, file)
            logger.info(f"Save population data to {path}_population.pkl")

        elif dataset_name == "300k_150x5_2":
            if not os.path.exists(f"{data_dir}/300k_150x5_2.h5"):
                logger.info(
                    f"{dataset_name} not found in /{data_dir}. Downloading data from https://github.com/automl/nanoTabPFN to /{data_dir}..."
                )
                try:
                    # Download the dataset to /data
                    subprocess.run(
                        [
                            "wget",
                            "http://ml.informatik.uni-freiburg.de/research-artifacts/nanoTabPFN/300k_150x5_2.h5",
                            "-P",
                            f"./{data_dir}",
                        ],
                        check=True,
                    )
                    logger.info(
                        "Dataset downloaded and extracted to /data successfully."
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error during download or extraction: {e}")
                    raise RuntimeError("Failed to download or extract the dataset.")
            with h5py.File(f"{data_dir}/300k_150x5_2.h5", "r") as f:
                X_all = f["X"]
                y_all = f["y"]
                single_eval_pos_all = f["single_eval_pos"]
                num_of_datasets = X_all.shape[0]
                training_size = int(
                    len(y_all) * 0.75
                ) 
                idxs = sorted(np.random.choice(num_of_datasets, size=training_size, replace=False))
                idxs_test = sorted(np.setdiff1d(np.arange(num_of_datasets), idxs))
                all_data = []
                for i in idxs:
                    X_i = np.array(X_all[i], dtype=np.float32)
                    y_i = np.array(y_all[i])
                    y_i = y_i.astype(np.int64, copy=False)
                    single_eval_pos_i = np.array(single_eval_pos_all[i], dtype=np.int8)
                    all_data.append(TabularDatasetTabPFN(X_i, y_i, single_eval_pos_i))
                test_data = []
                for i in idxs_test:
                    X_i = np.array(X_all[i], dtype=np.float32)
                    y_i = np.array(y_all[i])
                    y_i = y_i.astype(np.int64, copy=False)
                    single_eval_pos_i = np.array(single_eval_pos_all[i], dtype=np.int8)
                    test_data.append(TabularDatasetTabPFN(X_i, y_i, single_eval_pos_i))
            save_sampled(
                f"{data_dir}/300k_150x5_2.h5",
                f"{data_dir}/300k_150x5_2_axis0_all.h5",
                idxs,
            )
            logger.info("Save data to 300k_150x5_2_axis0_all.h5")
            save_sampled(
                f"{data_dir}/300k_150x5_2.h5",
                f"{data_dir}/300k_150x5_2_axis0_test.h5",
                idxs_test,
            )
            logger.info("Save population data to 300k_150x5_2_axis0_test.h5")
            verify_sampled_files(
                f"{data_dir}/300k_150x5_2.h5",
                [
                    f"{data_dir}/300k_150x5_2_axis0_all.h5",
                    f"{data_dir}/300k_150x5_2_axis0_test.h5",
                ],
            )
            logger.info("Verified sampled HDF5 slices for training and test splits")

        elif dataset_name == "purchase100":
            if not os.path.exists(f"{data_dir}/dataset_purchase"):
                logger.info(
                    f"{dataset_name} not found in {data_dir}/dataset_purchase. Downloading to /data..."
                )
                try:
                    # Download the dataset to /data
                    subprocess.run(
                        [
                            "wget",
                            "https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz",
                            "-P",
                            f"./{data_dir}",
                        ],
                        check=True,
                    )
                    # Extract the dataset to /data
                    subprocess.run(
                        [
                            "tar",
                            "-xf",
                            f"./{data_dir}/dataset_purchase.tgz",
                            "-C",
                            f"./{data_dir}",
                        ],
                        check=True,
                    )
                    logger.info(
                        "Dataset downloaded and extracted to /data successfully."
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error during download or extraction: {e}")
                    raise RuntimeError("Failed to download or extract the dataset.")

            df = pd.read_csv(
                f"{data_dir}/dataset_purchase", header=None, encoding="utf-8"
            ).to_numpy()
            y = df[:, 0] - 1
            X = df[:, 1:].astype(np.float32)
            training_size = int(
                len(y) * 0.75
            )  # Splitting to create a population dataset
            all_data = TabularDataset(X[:training_size], y[:training_size])
            test_data = TabularDataset(X[training_size:], y[training_size:])
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            logger.info(f"Save data to {path}.pkl")
            with open(f"{path}_population.pkl", "wb") as file:
                pickle.dump(test_data, file)
            logger.info(f"Save population data to {path}_population.pkl")
        elif dataset_name == "texas100":
            if not os.path.exists(f"{data_dir}/dataset_texas/feats"):
                logger.info(
                    f"{dataset_name} not found in {data_dir}/dataset_purchase. Downloading to /data..."
                )
                try:
                    # Download the dataset to /data
                    subprocess.run(
                        [
                            "wget",
                            "https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz",
                            "-P",
                            f"./{data_dir}",
                        ],
                        check=True,
                    )
                    # Extract the dataset to /data
                    subprocess.run(
                        [
                            "tar",
                            "-xf",
                            f"./{data_dir}/dataset_texas.tgz",
                            "-C",
                            "./data",
                        ],
                        check=True,
                    )
                    logger.info(
                        "Dataset downloaded and extracted to /data successfully."
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error during download or extraction: {e}")
                    raise RuntimeError("Failed to download or extract the dataset.")

            X = (
                pd.read_csv(
                    f"{data_dir}/dataset_texas/feats", header=None, encoding="utf-8"
                )
                .to_numpy()
                .astype(np.float32)
            )
            y = (
                pd.read_csv(
                    f"{data_dir}/dataset_texas/labels",
                    header=None,
                    encoding="utf-8",
                )
                .to_numpy()
                .reshape(-1)
                - 1
            )
            training_size = int(
                len(y) * 0.75
            )  # Splitting to create a population dataset
            all_data = TabularDataset(X[:training_size], y[:training_size])
            test_data = TabularDataset(X[training_size:], y[training_size:])
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            logger.info(f"Save data to {path}.pkl")
            with open(f"{path}_population.pkl", "wb") as file:
                pickle.dump(test_data, file)
            logger.info(f"Save population data to {path}_population.pkl")
        elif dataset_name == "agnews":
            tokenizer = kwargs.get("tokenizer")
            if tokenizer is None:
                agnews = load_agnews(tokenize=False)
                agnews_test = load_agnews(split="test", tokenize=False)
            else:
                agnews = load_agnews(
                    tokenize=True,
                    tokenizer=AutoTokenizer.from_pretrained(
                        tokenizer, clean_up_tokenization_spaces=True
                    ),
                )
                agnews_test = load_agnews(
                    split="test",
                    tokenize=True,
                    tokenizer=AutoTokenizer.from_pretrained(
                        tokenizer, clean_up_tokenization_spaces=True
                    ),
                )
            all_data = TextDataset(agnews, target_column="labels", text_column="text")
            test_data = TextDataset(
                agnews_test, target_column="labels", text_column="text"
            )
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            logger.info(f"Save data to {path}.pkl")
            with open(f"{path}_population.pkl", "wb") as file:
                pickle.dump(test_data, file)
            logger.info(f"Save population data to {path}_population.pkl")
        else:
            raise NotImplementedError(f"{dataset_name} is not implemented")

    logger.info(f"The whole dataset size: {len(all_data)}")
    return all_data, test_data


def load_dataset_subsets(
    dataset: torchvision.datasets,
    index: List[int],
    model_type: str,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Function to divide dataset into subsets and load them into device (GPU).

    Args:
        dataset (torchvision.datasets): The whole dataset.
        index (List[int]): List of sample indices.
        model_type (str): Type of the model.
        batch_size (int): Batch size for getting signals.
        device (str): Device used for loading models.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Loaded samples and their labels.
    """
    assert max(index) < len(dataset) and min(index) >= 0, "Index out of range"
    input_list = []
    targets_list = []
    if model_type != "speedyresnet":
        if batch_size == 1:
            # This happens with range dataset. Need to set num_workers to 0 to avoid CUDA error
            data_loader = get_dataloader(
                torch.utils.data.Subset(dataset, index),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
        else:
            data_loader = get_dataloader(
                torch.utils.data.Subset(dataset, index),
                batch_size=batch_size,
                shuffle=False,
            )
        for inputs, targets in data_loader:
            input_list.append(inputs)
            targets_list.append(targets)
        inputs = torch.cat(input_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
    else:
        data = load_cifar10_data(dataset, index[:1], index, device=device)
        size = len(index)
        list_divisors = list(
            set(
                factor
                for i in range(1, int(math.sqrt(size)) + 1)
                if size % i == 0
                for factor in (i, size // i)
                if factor < batch_size
            )
        )
        batch_size = max(list_divisors)

        for inputs, targets in get_batches(
            data, key="eval", batchsize=batch_size, shuffle=False, device=device
        ):
            input_list.append(inputs)
            targets_list.append(targets)
        inputs = torch.cat(input_list, dim=0)
        targets = torch.cat(targets_list, dim=0).max(dim=1)[1]
    return inputs, targets


def get_dataloader(
    dataset: torchvision.datasets,
    batch_size: int,
    loader_type: str = "torch",
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Function to get DataLoader.

    Args:
        dataset (torchvision.datasets): The whole dataset.
        batch_size (int): Batch size for getting signals.
        loader_type (str): Loader type.
        shuffle (bool): Whether to shuffle dataset or not.

    Returns:
        DataLoader: DataLoader object.
    """
    if loader_type == "torch":
        repeated_data = InfinitelyIndexableDataset(dataset)
        return DataLoader(
            repeated_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=16 if num_workers > 0 else None,
        )
    else:
        raise NotImplementedError(f"{loader_type} is not supported")
