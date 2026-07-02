from RGCN_Train import MutagenicityGNN, train_epoch, evaluate, set_seed
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import PYG_DATASET_PATH, RESULTS_DIR


def run_kfold(
    pyg_dataset,
    n_splits,
    hidden_channels,
    epochs,
    lr,
    batch_size,
    seed,
    device,
):

    # Pull labels out so StratifiedKFold can balance the folds
    labels = [data.y.item() for data in pyg_dataset]
    indices = list(range(len(pyg_dataset)))

    # Infer architecture dimensions from the saved dataset
    in_channels = pyg_dataset[0].x.size(1)
    num_relations = int(max(data.edge_type.max().item() for data in pyg_dataset) + 1)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels), start=1):

        print(f"\n{'─'*52}")
        print(
            f"  Fold {fold:2d} / {n_splits}   "
            f"(train={len(train_idx)}, test={len(test_idx)})"
        )
        print(f"{'─'*52}")

        # Re-seed before every fold so model init is deterministic but
        # different folds still see different weight initialisations when
        # seed + fold_number vary together.
        set_seed(seed + fold)

        train_loader = DataLoader(
            [pyg_dataset[i] for i in train_idx],
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            [pyg_dataset[i] for i in test_idx],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        model = MutagenicityGNN(
            in_channels=in_channels,
            num_relations=num_relations,
            hidden_channels=hidden_channels,
            num_classes=2,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in tqdm(range(1, epochs + 1)):
            train_epoch(model, train_loader, optimizer, criterion, device)

        test_acc = evaluate(model, test_loader, device)
        fold_accs.append(test_acc)
        print(f"\n✓ Fold {fold} Test Accuracy: {test_acc:.4f}")

    # Final summary
    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))

    print(f"\n{'═'*52}")
    print(f"{n_splits}-Fold Cross-Validation Summary")
    print(f"{'═'*52}")
    for i, acc in enumerate(fold_accs, start=1):
        bar = " " * int(acc * 20)
        print(f"  Fold {i:2d}: {acc:.4f}  {bar}")
    print(f"{'─'*52}")
    print(f"Mean Accuracy: {mean_acc:.4f}")
    print(f"Std Deviation: {std_acc:.4f}")
    print(f"95% CI: [{mean_acc - 2*std_acc:.4f},  {mean_acc + 2*std_acc:.4f}]")
    print(f"{'═'*52}")

    return fold_accs, mean_acc, std_acc


if __name__ == "__main__":
    # setting seed for deterministic execution
    set_seed(42)

    # same hyper parameters as train file
    BATCH_SIZE = 32
    HIDDEN_CHANNELS = 64
    EPOCHS = 50
    LR = 0.001

    print("[INFO] Loading pre-processed PyG dataset...")
    pyg_dataset = torch.load(
        PYG_DATASET_PATH,
        weights_only=False,
    )
    print(f"[INFO] Total molecules loaded: {len(pyg_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    fold_accs, mean_acc, std_acc = run_kfold(
        pyg_dataset=pyg_dataset,
        n_splits=10,
        hidden_channels=HIDDEN_CHANNELS,
        epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        seed=42,
        device=device,
    )

    # Generate and save box plot in results folder
    print("[INFO] Generating K-Fold Cross-Validation accuracy box plot...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Style configuration
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 0.8

    plt.figure(figsize=(6, 5))
    
    # Customizing boxplot elements for nice aesthetics
    box = plt.boxplot(
        fold_accs, 
        patch_artist=True, 
        tick_labels=['10-Fold CV'],
        boxprops=dict(facecolor='#EBF3F9', color='#1F77B4', linewidth=1.5),
        capprops=dict(color='#1F77B4', linewidth=1.5),
        whiskerprops=dict(color='#1F77B4', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='#FF7F0E', markersize=6, linestyle='none', markeredgecolor='#FF7F0E'),
        medianprops=dict(color='#D62728', linewidth=2.0)
    )

    # Add individual data points for better transparency of results (swarm-like overlay)
    x = np.random.normal(1, 0.04, size=len(fold_accs))
    plt.scatter(x, fold_accs, alpha=0.7, color='#1F77B4', edgecolor='none', zorder=3, label='Fold Accuracy')

    plt.title(f'10-Fold Cross-Validation Accuracy\n(Mean: {mean_acc:.4f} ± {std_acc:.4f})', fontsize=12, pad=15, fontweight='bold', color='#333333')
    plt.ylabel('Test Accuracy', fontsize=11, labelpad=8, color='#333333')
    plt.grid(True, linestyle='--', alpha=0.5, color='#DDDDDD')
    plt.ylim(min(0.5, min(fold_accs) - 0.05), 1.0)
    
    plt.tight_layout()
    boxplot_path = RESULTS_DIR / "kfold_accuracy_boxplot.png"
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    print(f"[INFO] K-Fold accuracy boxplot saved to {boxplot_path}")
