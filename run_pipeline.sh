#!/bin/bash

# Exit immediately if a command exits with a non-zero status
# If you are running it manually Ignore the code block below.
set -e

# If you are running it manually Ignore the code block below.
# Detect Python command (use python3 if python is not available/mapped)
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        echo "Error: Python is not installed or not in the PATH." >&2
        exit 1
    fi
fi

# START FROM HERE IF YOU WANT TO RUN IT MANUALLY EACH COMMAND BELOW
echo "=================================================="
echo " Starting Mutagenicity XAI GNN Pipeline"
echo "=================================================="
echo "Using python command: $PYTHON_CMD"

# 1. Preprocessing
echo ""
echo "=================================================="
echo "[1/5] Preprocessing the dataset..."
echo "=================================================="
$PYTHON_CMD data_preprocessing.py

# 2. Training
echo ""
echo "=================================================="
echo "[2/5] Training the RGCN model..."
echo "=================================================="
$PYTHON_CMD RGCN_Train.py

# 3. K-Fold Cross Validation
echo ""
echo "=================================================="
echo "[3/5] Evaluating performance with 10-fold cross-validation..."
echo "=================================================="
$PYTHON_CMD RGCN_Kfold.py

# 4. GNN Explainer
echo ""
echo "=================================================="
echo "[4/5] Running GNNExplainer for molecule explanation..."
echo "=================================================="
$PYTHON_CMD GNNExplainer.py

# 5. Evaluation of explanations
echo ""
echo "=================================================="
echo "[5/5] Evaluating GNNExplainer metrics..."
echo "=================================================="
$PYTHON_CMD Evaluate_GNNExplainer.py

echo ""
echo "=================================================="
echo " Pipeline completed successfully!"
echo "=================================================="
