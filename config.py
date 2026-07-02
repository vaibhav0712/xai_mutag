from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset directory and files
DATASET_DIR = PROJECT_ROOT / "mutag-hetero"
RDF_PATH = DATASET_DIR / "mutag_stripped.nt"
TRAINING_SET_PATH = DATASET_DIR / "trainingSet.tsv"
TEST_SET_PATH = DATASET_DIR / "testSet.tsv"
PYG_DATASET_PATH = DATASET_DIR / "pyg_dataset.pt"

# Model directory and checkpoint
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_CHECKPOINT_PATH = MODEL_DIR / "rgnn_model_checkpoint.pt"

# Results directory and files
RESULTS_DIR = PROJECT_ROOT / "results"
EXPLANATION_EVALUATION_RESULTS_PATH = RESULTS_DIR / "explanation_evaluation_results.json"
PAYLOAD_PATH = RESULTS_DIR / "payload.json"
