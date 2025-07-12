#!/bin/bash

# Unified evaluation script for VibeRL
# Usage: ./examples/eval.sh [algorithm] [environment] [model_path] [episodes]

ALG=${1:-reinforce}
ENV=${2:-snake}
MODEL_PATH=${3}
EPISODES=${4:-10}

# Check if model path is provided
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: Model path is required"
    echo "Usage: $0 [algorithm] [environment] [model_path] [episodes]"
    echo "Example: $0 dqn snake experiments/dqn_snake_20250712/final_model.pth 10"
    exit 1
fi

# Check if file exists
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Error: Model file '$MODEL_PATH' not found"
    exit 1
fi

# Check if algorithm is valid
if [[ "$ALG" != "reinforce" && "$ALG" != "dqn" ]]; then
    echo "Error: Algorithm must be 'reinforce' or 'dqn'"
    echo "Usage: $0 [reinforce|dqn] [env] [model_path] [episodes]"
    exit 1
fi

echo "Starting evaluation with:"
echo "  Algorithm: $ALG"
echo "  Environment: $ENV"
echo "  Model: $MODEL_PATH"
echo "  Episodes: $EPISODES"
echo ""

# Run the evaluation
python -m viberl.cli eval --alg "$ALG" --env "$ENV" --model-path "$MODEL_PATH" --episodes "$EPISODES" "$@:5}"

echo ""
echo "Evaluation completed!"}
