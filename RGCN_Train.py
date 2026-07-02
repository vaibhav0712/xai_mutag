from torch_geometric.nn import FastRGCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import numpy as np
import random
import torch
import os
import matplotlib.pyplot as plt

from config import (
    PYG_DATASET_PATH,
    TRAINING_SET_PATH,
    TEST_SET_PATH,
    MODEL_DIR,
    MODEL_CHECKPOINT_PATH,
    RESULTS_DIR,
)

if not MODEL_DIR.exists():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


set_seed(132)


# Model Definition
class MutagenicityGNN(torch.nn.Module):
    def __init__(self, in_channels, num_relations, hidden_channels=64, num_classes=2):
        super().__init__()
        self.conv1 = FastRGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = FastRGCNConv(hidden_channels, hidden_channels, num_relations)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_type, batch):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# Training and Evaluation Functions
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_type, data.batch)
        loss = criterion(out, data.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_type, data.batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y.squeeze()).sum().item()
    return correct / len(loader.dataset)


# Main Execution Block
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 32
    HIDDEN_CHANNELS = 64
    EPOCHS = 50
    LR = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

    print("[INFO] Loading processed dataset...")
    pyg_dataset = torch.load(PYG_DATASET_PATH, weights_only=False)
    train_df = pd.read_csv(TRAINING_SET_PATH, sep="\t")
    test_df = pd.read_csv(TEST_SET_PATH, sep="\t")

    print("[INFO] Mapping training and testing molecule IDs to dataset indices...")
    id_to_idx = {data.molecule_id: i for i, data in enumerate(pyg_dataset)}

    train_ids = sorted(train_df["bond"].apply(lambda x: x.split("#")[-1]))
    test_ids = sorted(test_df["bond"].apply(lambda x: x.split("#")[-1]))

    train_idx = [id_to_idx[mol_id] for mol_id in train_ids if mol_id in id_to_idx]
    test_idx = [id_to_idx[mol_id] for mol_id in test_ids if mol_id in id_to_idx]
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

    print("[INFO] Constructing train, test and validation DataLoader")
    train_loader = DataLoader(
        [pyg_dataset[i] for i in train_idx],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        [pyg_dataset[i] for i in val_idx],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        [pyg_dataset[i] for i in test_idx],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Dynamically extract dimensions from the loaded data
    in_channels = pyg_dataset[0].x.size(1)

    # Extract total number of unique edge types across the dataset
    num_relations = int(max(data.edge_type.max().item() for data in pyg_dataset) + 1)

    model = MutagenicityGNN(
        in_channels=in_channels,
        num_relations=num_relations,
        hidden_channels=HIDDEN_CHANNELS,
        num_classes=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    epochs_list = []
    loss_list = []
    train_acc_list = []
    val_acc_list = []

    print("[INFO] Starting training...")
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)

        epochs_list.append(epoch)
        loss_list.append(loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:02d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
            )

    test_acc = evaluate(model, test_loader, device)
    print(f"[INFO] Final Test Accuracy: {test_acc:.4f}")

    # Generate and save plots in results folder
    print("[INFO] Generating training performance plots...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Style configuration
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 0.8

    # 1. Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_list, loss_list, color='#1F77B4', linewidth=2.0, label='Training Loss')
    plt.title('Training Loss vs. Epochs', fontsize=13, pad=15, fontweight='bold', color='#333333')
    plt.xlabel('Epoch', fontsize=11, labelpad=8, color='#333333')
    plt.ylabel('Loss', fontsize=11, labelpad=8, color='#333333')
    plt.grid(True, linestyle='--', alpha=0.5, color='#DDDDDD')
    plt.xlim(1, EPOCHS)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    loss_plot_path = RESULTS_DIR / "loss_plot.png"
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Loss plot saved to {loss_plot_path}")

    # 2. Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_list, train_acc_list, color='#2CA02C', linewidth=2.0, label='Train Accuracy')
    plt.plot(epochs_list, val_acc_list, color='#FF7F0E', linewidth=2.0, linestyle='--', label='Val Accuracy')
    plt.title('Model Accuracy vs. Epochs', fontsize=13, pad=15, fontweight='bold', color='#333333')
    plt.xlabel('Epoch', fontsize=11, labelpad=8, color='#333333')
    plt.ylabel('Accuracy', fontsize=11, labelpad=8, color='#333333')
    plt.grid(True, linestyle='--', alpha=0.5, color='#DDDDDD')
    plt.xlim(1, EPOCHS)
    plt.ylim(0, 1.0)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    accuracy_plot_path = RESULTS_DIR / "accuracy_plot.png"
    plt.savefig(accuracy_plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Accuracy plot saved to {accuracy_plot_path}")

    print("[INFO] Saving model checkpoint...")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "hyperparameters": {
            "in_channels": in_channels,
            "num_relations": num_relations,
            "hidden_channels": HIDDEN_CHANNELS,
            "num_classes": 2,
        },
    }

    torch.save(checkpoint, MODEL_CHECKPOINT_PATH)
    print(f"[INFO] Saved successfully as {MODEL_CHECKPOINT_PATH}")
