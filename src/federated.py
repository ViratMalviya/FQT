# =============================================================================
# src/federated.py
# Federated Learning client and server aggregation logic using Flower (FLWR).
# =============================================================================
# This module implements the full FL pipeline:
#
#   - FlowerClient: implements the NumPyClient interface so each simulated
#     edge device can perform local training and share parameter updates.
#
#   - local_train(): runs multiple SGD epochs on a client's local dataset
#     and returns updated model parameters + training metrics.
#
#   - federated_average(): implements the FedAvg aggregation algorithm,
#     averaging parameter updates from all participating clients weighted
#     by their dataset sizes.
#
#   - client_fn(): factory function called by Flower's simulation engine
#     to instantiate a FlowerClient for a given client ID.
#
# Usage:
#   Run via main.py which calls fl.simulation.start_simulation().
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import flwr as fl
from flwr.common import NDArrays, Scalar

from models import QuantumTrainGenerator, TargetCNN
from preprocessing import load_local_dataset


# =============================================================================
# LOCAL TRAINING LOOP
# =============================================================================

def local_train(
    model: nn.Module,
    data_loader,
    epochs: int = 3,
    learning_rate: float = 0.01,
    device: torch.device = torch.device("cpu")
) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
    """
    Perform local SGD training on a single client's private dataset.

    The raw audio data (MFCCs) never leaves the local device — only the
    updated model parameters are returned for federated aggregation.

    Args:
        model (nn.Module):      The quantum-classical model to train locally.
        data_loader (DataLoader): Client's local training DataLoader.
        epochs (int):           Number of local SGD epochs per FL round.
        learning_rate (float):  SGD learning rate.
        device (torch.device):  CPU or CUDA device to train on.

    Returns:
        Tuple of:
            - parameters (List[np.ndarray]): Updated model weights as numpy arrays.
            - num_samples (int): Total samples seen during training.
            - metrics (Dict):    Loss and accuracy over all epochs.
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0

        for mfcc_batch, label_batch in data_loader:
            mfcc_batch  = mfcc_batch.to(device)
            label_batch = label_batch.to(device)

            # Forward pass: quantum weight generation → CNN → logits
            logits = model(mfcc_batch)
            loss   = criterion(logits, label_batch)

            # Backward pass: gradients flow through MLP heads and VQC params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            predictions     = logits.argmax(dim=1)
            batch_correct   = (predictions == label_batch).sum().item()
            epoch_correct  += batch_correct
            epoch_loss     += loss.item() * len(label_batch)

        epoch_samples   = len(data_loader.dataset)
        total_loss     += epoch_loss
        total_correct  += epoch_correct
        total_samples   = epoch_samples  # Same dataset each epoch

    avg_loss     = total_loss / (epochs * total_samples + 1e-9)
    avg_accuracy = total_correct / (epochs * total_samples + 1e-9)

    # Extract model parameters as flat numpy arrays (for Flower transmission)
    parameters = [
        val.cpu().detach().numpy()
        for val in model.state_dict().values()
    ]

    metrics = {"loss": avg_loss, "accuracy": avg_accuracy}
    return parameters, total_samples, metrics


# =============================================================================
# FEDERATED AVERAGING (FedAvg)
# =============================================================================

def federated_average(
    results: List[Tuple[List[np.ndarray], int]]
) -> List[np.ndarray]:
    """
    Aggregate model parameter updates using the FedAvg algorithm.

    FedAvg computes a weighted average of client model parameters,
    where each client's contribution is proportional to its dataset size.
    This ensures that clients with more data have a larger influence on
    the global model, which is statistically principled.

    Reference: McMahan et al., "Communication-Efficient Learning of Deep
    Networks from Decentralized Data", AISTATS 2017.

    Args:
        results: List of (parameters, num_samples) tuples from each client.
                 parameters: List of numpy arrays (one per model layer).
                 num_samples: Number of training samples on that client.

    Returns:
        List[np.ndarray]: Aggregated (averaged) global model parameters.
    """
    total_samples = sum(num_samples for _, num_samples in results)

    # Weighted average across clients
    aggregated = [
        np.zeros_like(params[0]) for params in [r[0] for r in results]
    ]

    for parameters, num_samples in results:
        weight = num_samples / total_samples
        for i, param in enumerate(parameters):
            aggregated[i] += weight * param

    return aggregated


# =============================================================================
# FLOWER CLIENT
# =============================================================================

class FlowerClient(fl.client.NumPyClient):
    """
    Flower NumPyClient implementation for the FQT framework.

    Each FlowerClient instance represents one simulated edge device.
    It holds a local copy of the quantum-classical model and a private
    DataLoader. During each federated round, it:
        1. Receives the latest global model parameters from the server.
        2. Performs local_train() on its private dataset.
        3. Returns the updated local parameters + training metrics.

    The critical privacy guarantee: raw MFCC features (voice data) are
    never transmitted — only floating-point model parameter deltas.

    Args:
        client_id (int):           Unique identifier for this client.
        dataset_path (str):        Path to this client's local dataset.
        config (dict):             Loaded config.yaml dictionary.
        device (torch.device):     Training device.
    """

    def __init__(
        self,
        client_id: int,
        dataset_path: str,
        config: dict,
        device: torch.device
    ):
        self.client_id   = client_id
        self.device      = device
        self.config      = config

        # Build the quantum-classical model for this client
        self.model = QuantumTrainGenerator(TargetCNN())

        # Load this client's local private dataset
        self.train_loader = load_local_dataset(
            dataset_path=dataset_path,
            batch_size=config["training"]["batch_size"],
            n_mfcc=config["model"]["n_mfcc"],
            max_len=config["model"]["max_len"],
            sample_rate=config["data"]["sample_rate"]
        )

    def get_parameters(self, config: dict) -> NDArrays:
        """Return current local model parameters as a list of numpy arrays."""
        return [
            val.cpu().detach().numpy()
            for val in self.model.state_dict().values()
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Load a set of numpy parameter arrays into the local model."""
        state_dict = OrderedDict(
            {
                k: torch.tensor(v)
                for k, v in zip(self.model.state_dict().keys(), parameters)
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: NDArrays,
        config: dict
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Receive global parameters, train locally, return updated parameters.

        Args:
            parameters: Global model parameters from the aggregation server.
            config:     Server-side config passed for this round.

        Returns:
            (updated_params, num_samples, metrics)
        """
        # Load the latest global model weights
        self.set_parameters(parameters)

        # Perform local training on private client data
        updated_params, num_samples, metrics = local_train(
            model=self.model,
            data_loader=self.train_loader,
            epochs=self.config["training"]["local_epochs"],
            learning_rate=self.config["training"]["learning_rate"],
            device=self.device
        )

        print(
            f"[Client {self.client_id}] "
            f"Loss: {metrics['loss']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Samples: {num_samples}"
        )

        return updated_params, num_samples, metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate the global model on the local client dataset.

        Args:
            parameters: Global model parameters to evaluate.
            config:     Server-side evaluation config.

        Returns:
            (loss, num_samples, metrics)
        """
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss    = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for mfcc_batch, label_batch in self.train_loader:
                mfcc_batch  = mfcc_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                logits = self.model(mfcc_batch)
                loss   = criterion(logits, label_batch)

                predictions    = logits.argmax(dim=1)
                total_correct += (predictions == label_batch).sum().item()
                total_loss    += loss.item() * len(label_batch)
                total_samples += len(label_batch)

        avg_loss     = total_loss / (total_samples + 1e-9)
        avg_accuracy = total_correct / (total_samples + 1e-9)

        return avg_loss, total_samples, {"accuracy": avg_accuracy}
