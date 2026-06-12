# =============================================================================
# src/main.py
# Entry point for the FQT Federated Quantum-Train simulation.
# =============================================================================
# Launches a Flower federated learning simulation with NUM_CLIENTS edge
# devices, each training a QuantumTrainGenerator locally on its private
# audio dataset, then aggregating via FedAvg over NUM_ROUNDS rounds.
#
# Usage:
#   python src/main.py
#   python src/main.py --config config.yaml
# =============================================================================

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import NDArrays, Scalar
from flwr.server.strategy import FedAvg

# Add src/ to path so imports resolve from the project root
sys.path.insert(0, os.path.dirname(__file__))

from models import QuantumTrainGenerator, TargetCNN
from federated import FlowerClient


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FQT: Federated Quantum-Train Audio Deepfake Detector"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load and return the YAML configuration as a Python dictionary."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Make sure config.yaml is in the project root directory."
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# FLOWER CLIENT FACTORY
# =============================================================================

def client_fn(client_id: str, cfg: dict, device: torch.device):
    """
    Factory function called by Flower's simulation engine to instantiate
    a FlowerClient for a given client ID string.

    Each client is assigned its own dataset subdirectory:
        <dataset_path>/client_<id>/

    Args:
        client_id (str):      Flower-assigned client ID string (e.g. "0", "1").
        cfg (dict):           Loaded configuration dictionary.
        device (torch.device): Training device.

    Returns:
        FlowerClient: Instantiated and ready-to-train client.
    """
    # Each client has its own local data partition
    client_dataset_path = os.path.join(
        cfg["data"]["dataset_path"],
        f"client_{client_id}"
    )

    if not os.path.isdir(client_dataset_path):
        # Fallback: use shared dataset path (for single-machine demo)
        client_dataset_path = cfg["data"]["dataset_path"]

    return FlowerClient(
        client_id=int(client_id),
        dataset_path=client_dataset_path,
        config=cfg,
        device=device
    )


# =============================================================================
# TRAINING HISTORY VISUALIZATION
# =============================================================================

def plot_training_curves(
    loss_history: List[float],
    accuracy_history: List[float],
    save_path: str = "assets/training_curve.png"
) -> None:
    """
    Plot the federated training loss and test accuracy per round.

    Creates a dual-axis figure showing:
        - Left Y-axis: Average training loss (decreasing)
        - Right Y-axis: Test accuracy percentage (increasing)

    Args:
        loss_history (List[float]):     Average loss per federated round.
        accuracy_history (List[float]): Test accuracy per federated round.
        save_path (str):                File path to save the figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rounds = list(range(1, len(loss_history) + 1))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "FQT Federated Training: Loss & Accuracy per Round",
        fontsize=13, fontweight="bold"
    )

    # Left axis — Loss
    color_loss = "#E74C3C"
    ax1.set_xlabel("Federated Round", fontsize=11)
    ax1.set_ylabel("Average Loss", color=color_loss, fontsize=11)
    ax1.plot(
        rounds, loss_history,
        color=color_loss, marker="o", linewidth=2, markersize=7,
        label="Avg Loss"
    )
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    # Right axis — Accuracy
    ax2 = ax1.twinx()
    color_acc = "#2980B9"
    ax2.set_ylabel("Test Accuracy (%)", color=color_acc, fontsize=11)
    ax2.plot(
        rounds, [a * 100 for a in accuracy_history],
        color=color_acc, marker="s", linewidth=2, markersize=7,
        linestyle="--", label="Test Accuracy"
    )
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.set_ylim(0, 100)

    # Annotate final accuracy
    if accuracy_history:
        final_acc = accuracy_history[-1] * 100
        ax2.annotate(
            f"{final_acc:.2f}%",
            xy=(rounds[-1], final_acc),
            xytext=(-40, 10), textcoords="offset points",
            fontsize=10, color=color_acc, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color_acc)
        )

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Training curve saved to: {save_path}")
    plt.close()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed everything for reproducibility
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("  FQT — Federated Quantum-Train Audio Deepfake Detector")
    print("=" * 60)
    print(f"  Device       : {device}")
    print(f"  Clients      : {cfg['federated']['num_clients']}")
    print(f"  FL Rounds    : {cfg['federated']['num_rounds']}")
    print(f"  Local Epochs : {cfg['training']['local_epochs']}")
    print(f"  Learning Rate: {cfg['training']['learning_rate']}")
    print("=" * 60)

    # --- Track metrics across rounds ---
    loss_history     = []
    accuracy_history = []

    # --- FedAvg strategy with custom metric aggregation ---
    def evaluate_metrics_aggregation(metrics):
        """Aggregate accuracy from all clients (weighted average)."""
        total_samples = sum(num for num, _ in metrics)
        aggregated_accuracy = sum(
            num * m.get("accuracy", 0.0) for num, m in metrics
        ) / (total_samples + 1e-9)
        accuracy_history.append(aggregated_accuracy)
        return {"accuracy": aggregated_accuracy}

    def fit_metrics_aggregation(metrics):
        """Aggregate loss from all clients (weighted average)."""
        total_samples = sum(num for num, _ in metrics)
        aggregated_loss = sum(
            num * m.get("loss", 0.0) for num, m in metrics
        ) / (total_samples + 1e-9)
        loss_history.append(aggregated_loss)
        return {"loss": aggregated_loss}

    strategy = FedAvg(
        fraction_fit=cfg["federated"]["fraction_fit"],
        fraction_evaluate=cfg["federated"]["fraction_evaluate"],
        min_fit_clients=cfg["federated"]["num_clients"],
        min_evaluate_clients=cfg["federated"]["num_clients"],
        min_available_clients=cfg["federated"]["num_clients"],
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
    )

    # --- Launch Flower simulation ---
    fl.simulation.start_simulation(
        client_fn=lambda cid: client_fn(cid, cfg, device),
        num_clients=cfg["federated"]["num_clients"],
        config=fl.server.ServerConfig(
            num_rounds=cfg["federated"]["num_rounds"]
        ),
        strategy=strategy,
    )

    # --- Save the final global model ---
    print("\n[INFO] Saving final global model...")
    global_model = QuantumTrainGenerator(TargetCNN())
    torch.save(global_model.state_dict(), cfg["output"]["model_path"])
    print(f"[INFO] Global model saved to: {cfg['output']['model_path']}")

    # --- Plot training curves ---
    if loss_history and accuracy_history:
        plot_training_curves(
            loss_history=loss_history,
            accuracy_history=accuracy_history,
            save_path=os.path.join(cfg["output"]["assets_dir"], "training_curve.png")
        )

    print("\n[DONE] Federated training complete.")
    if accuracy_history:
        print(f"       Final Test Accuracy: {accuracy_history[-1]*100:.2f}%")


if __name__ == "__main__":
    main()
