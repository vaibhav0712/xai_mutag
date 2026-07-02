import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# external imports
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.metric import unfaithfulness
from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import json

# internal imports
from RGCN_Train import MutagenicityGNN, set_seed
from config import (
    PYG_DATASET_PATH,
    TRAINING_SET_PATH,
    TEST_SET_PATH,
    MODEL_CHECKPOINT_PATH,
    EXPLANATION_EVALUATION_RESULTS_PATH,
)

SEED = 42


# custom sparsity function
def compute_sparsity(explanation, data, topk=10):
    edge_mask = explanation.edge_mask
    num_edges = edge_mask.numel()
    num_nodes = data.num_nodes

    if num_edges == 0:
        return 1.0

    k = min(topk, num_edges)
    _, top_edge_indices = edge_mask.topk(k)
    selected_edges = torch.zeros(num_edges, device=edge_mask.device)
    selected_edges[top_edge_indices] = 1.0

    src = data.edge_index[0]
    dst = data.edge_index[1]
    incident_nodes = set()
    for ei in top_edge_indices.tolist():
        incident_nodes.add(src[ei].item())
        incident_nodes.add(dst[ei].item())

    num_selected_edges = int(selected_edges.sum().item())
    num_selected_nodes = len(incident_nodes)

    total_elements = num_nodes + num_edges
    selected_elements = num_selected_nodes + num_selected_edges

    sparsity = 1.0 - (selected_elements / total_elements)
    return sparsity


# custom fiedlity function. reason for using this is mentioned in report.
def compute_fidelity(explainer, explanation, data, device):
    node_mask = explanation.node_mask
    edge_mask = explanation.edge_mask
    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        original_logits = explainer.model(
            data.x, data.edge_index, data.edge_type, batch
        )
        original_probs = F.softmax(original_logits, dim=-1).squeeze()
        pred_class = original_logits.argmax(dim=-1).item()
        original_prob = original_probs[pred_class].item()

    with torch.no_grad():
        if node_mask is not None:
            x_complement = (1.0 - node_mask) * data.x
        else:
            x_complement = data.x

        if edge_mask is not None:
            complement_mask = 1.0 - edge_mask
            keep = complement_mask > 0.5
            masked_edge_index = data.edge_index[:, keep]
            masked_edge_type = data.edge_type[keep]
        else:
            masked_edge_index = data.edge_index
            masked_edge_type = data.edge_type

        if masked_edge_index.size(1) == 0:
            fidelity_pos = original_prob
        else:
            batch_c = torch.zeros(x_complement.size(0), dtype=torch.long, device=device)
            complement_logits = explainer.model(
                x_complement, masked_edge_index, masked_edge_type, batch_c
            )
            complement_probs = F.softmax(complement_logits, dim=-1).squeeze()
            complement_prob = complement_probs[pred_class].item()
            fidelity_pos = max(original_prob - complement_prob, 0.0)

    with torch.no_grad():
        if node_mask is not None:
            x_subgraph = node_mask * data.x
        else:
            x_subgraph = data.x

        if edge_mask is not None:
            keep = edge_mask > 0.5
            masked_edge_index = data.edge_index[:, keep]
            masked_edge_type = data.edge_type[keep]
        else:
            masked_edge_index = data.edge_index
            masked_edge_type = data.edge_type

        if masked_edge_index.size(1) == 0:
            fidelity_neg = original_prob
        else:
            batch_s = torch.zeros(x_subgraph.size(0), dtype=torch.long, device=device)
            subgraph_logits = explainer.model(
                x_subgraph, masked_edge_index, masked_edge_type, batch_s
            )
            subgraph_probs = F.softmax(subgraph_logits, dim=-1).squeeze()
            subgraph_prob = subgraph_probs[pred_class].item()
            fidelity_neg = max(original_prob - subgraph_prob, 0.0)

    return fidelity_pos, fidelity_neg


if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading dataset...")
    pyg_dataset = torch.load(PYG_DATASET_PATH, weights_only=False)
    print(f"[INFO] Total molecules in dataset: {len(pyg_dataset)}")

    print("[INFO] Loading train/test split...")
    train_df = pd.read_csv(TRAINING_SET_PATH, sep="\t")
    test_df = pd.read_csv(TEST_SET_PATH, sep="\t")

    id_to_idx = {data.molecule_id: i for i, data in enumerate(pyg_dataset)}
    test_ids = sorted(test_df["bond"].apply(lambda x: x.split("#")[-1]))
    test_idx = [id_to_idx[mol_id] for mol_id in test_ids if mol_id in id_to_idx]
    print(f"[INFO] Test set size: {len(test_idx)}")

    print("[INFO] Loading model checkpoint...")
    checkpoint = torch.load(
        MODEL_CHECKPOINT_PATH, map_location=device, weights_only=False
    )
    hparams = checkpoint["hyperparameters"]

    base_model = MutagenicityGNN(
        in_channels=hparams["in_channels"],
        num_relations=hparams["num_relations"],
        hidden_channels=hparams["hidden_channels"],
        num_classes=hparams["num_classes"],
    ).to(device)
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.eval()
    print("[INFO] Model loaded and set to eval mode.")

    print("[INFO] Initializing GNNExplainer...")
    explainer = Explainer(
        model=base_model,
        algorithm=GNNExplainer(epochs=200, lr=0.01),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
    )
    print("[INFO] Explainer ready.")

    results = []

    correctly_classified = 0
    total_test = len(test_idx)

    # Dictionary mappings and accumulators for tracking top important node/edge types
    id_to_bond = {0: "Bond-1", 1: "Bond-2", 2: "Bond-3", 3: "Bond-7"}
    relation_importance = defaultdict(float)
    relation_frequency = defaultdict(int)
    node_type_importance = defaultdict(float)
    node_type_frequency = defaultdict(int)

    print(f"[INFO] Evaluating explanations for {total_test} test molecules...")
    print("=" * 60)

    for i, idx in tqdm(enumerate(test_idx), total=len(test_idx)):
        data = pyg_dataset[idx].to(device)
        mol_id = data.molecule_id
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        with torch.no_grad():
            logits = base_model(data.x, data.edge_index, data.edge_type, batch)
            pred_class = logits.argmax(dim=1).item()
            true_class = data.y.item()

        is_correct = pred_class == true_class

        if not is_correct:
            continue

        correctly_classified += 1

        set_seed(SEED)

        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            batch=batch,
        )

        unfaith_score = unfaithfulness(explainer=explainer, explanation=explanation)
        fid_pos, fid_neg = compute_fidelity(explainer, explanation, data, device)
        sparsity_score = compute_sparsity(explanation, data, topk=10)

        # Accumulate edge and node type importance based on top-k edges
        edge_mask = explanation.edge_mask.cpu().numpy()
        topk_edges_val = 10
        top_edge_indices = edge_mask.argsort()[-topk_edges_val:][::-1]
        for idx_edge in top_edge_indices:
            weight = float(edge_mask[idx_edge])
            if weight <= 0.0:
                continue

            rel_id = int(data.edge_type[idx_edge].item())
            rel_str = id_to_bond.get(rel_id, f"Bond-{rel_id}")
            relation_importance[rel_str] += weight
            relation_frequency[rel_str] += 1

            src_id = int(data.edge_index[0, idx_edge].item())
            dst_id = int(data.edge_index[1, idx_edge].item())

            for n_id in (src_id, dst_id):
                node_type = (
                    data.atom_info[n_id]["type"]
                    if (data.atom_info is not None and n_id < len(data.atom_info))
                    else "Unknown"
                )
                if explanation.node_mask is not None:
                    node_w = float(explanation.node_mask[n_id].sum().item())
                else:
                    node_w = weight
                node_type_importance[node_type] += node_w
                node_type_frequency[node_type] += 1

        result = {
            "molecule_id": mol_id,
            "predicted_class": pred_class,
            "true_class": true_class,
            "fidelity_pos": fid_pos,
            "fidelity_neg": fid_neg,
            "unfaithfulness": unfaith_score,
            "sparsity": sparsity_score,
        }
        results.append(result)

    print("=" * 60)
    print(f"[INFO] Correctly classified: {correctly_classified} / {total_test}")

    if len(results) == 0:
        print(
            "[WARNING] No correctly classified molecules found. No averages to report."
        )
    else:
        avg_fid_pos = np.mean([r["fidelity_pos"] for r in results])
        avg_fid_neg = np.mean([r["fidelity_neg"] for r in results])
        avg_unfaith = np.mean([r["unfaithfulness"] for r in results])
        avg_sparsity = np.mean([r["sparsity"] for r in results])

        print(f"\n{'='*60}")
        print(f"  Average Explanation Metrics ({len(results)} correctly classified)")
        print(f"{'='*60}")
        print(f"  Avg Fidelity (+)     : {avg_fid_pos:.4f}")
        print(f"  Avg Fidelity (-)      : {avg_fid_neg:.4f}")
        print(f"  Avg Unfaithfulness    : {avg_unfaith:.4f}  (lower is better)")
        print(f"  Avg Sparsity          : {avg_sparsity:.4f}  (higher is better)")
        print(f"{'='*60}")

        # Print top relations
        print(f"\n{'='*60}")
        print("Top Most Influential Relation Types (Top 10 edges)")
        print(f"{'='*60}")
        sorted_relations = sorted(relation_importance.items(), key=lambda x: -x[1])
        print(f"{'Relation Type':<30} | {'Accumulated Weight':<20} | {'Frequency':<10}")
        print("-" * 68)
        for rel_str, weight in sorted_relations[:10]:
            freq = relation_frequency[rel_str]
            print(f"{rel_str:<30} | {weight:<20.4f} | {freq:<10}")

        # Print top node types
        print(f"\n{'='*60}")
        print("Top Most Influential Node Types (from Top 10 edges)")
        print(f"{'='*60}")
        sorted_nodes = sorted(node_type_importance.items(), key=lambda x: -x[1])
        print(
            f"{'Node Class/Type':<30} | {'Accumulated Weight':<20} | {'Frequency':<10}"
        )
        print("-" * 68)
        for node_type, weight in sorted_nodes[:10]:
            freq = node_type_frequency[node_type]
            print(f"{node_type:<30} | {weight:<20.4f} | {freq:<10}")
        print(f"{'='*60}")

        output_path = EXPLANATION_EVALUATION_RESULTS_PATH

        serialized_relations = [
            {
                "relation_type": rel,
                "accumulated_weight": float(weight),
                "frequency": relation_frequency[rel],
            }
            for rel, weight in sorted_relations
        ]
        serialized_nodes = [
            {
                "node_type": node,
                "accumulated_weight": float(weight),
                "frequency": node_type_frequency[node],
            }
            for node, weight in sorted_nodes
        ]

        data_to_save = (
            {
                "per_molecule_results": results,
                "averages": {
                    "fidelity_pos": avg_fid_pos,
                    "fidelity_neg": avg_fid_neg,
                    "unfaithfulness": avg_unfaith,
                    "sparsity": avg_sparsity,
                },
                "top_influential_relations": serialized_relations,
                "top_influential_nodes": serialized_nodes,
                "num_correctly_classified": len(results),
                "total_test": total_test,
                "seed": SEED,
            },
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=4)

        print(f"[INFO] Results saved to {output_path}")
