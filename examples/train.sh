#!/bin/bash

# Unified training script for VibeRL
# Usage: ./examples/train.sh [algorithm] [environment] [episodes] [additional_args...]

ALG=${1:-reinforce}
ENV=${2:-snake}
EPISODES=${3:-1000}

# Check if algorithm is valid
if [[ "$ALG" != "reinforce" && "$ALG" != "dqn" ]]; then
    echo "Error: Algorithm must be 'reinforce' or 'dqn'"
    echo "Usage: $0 [reinforce|dqn] [env] [episodes] [additional_args...]"
    exit 1
fi

# Check if environment is valid
if [[ "$ENV" != "snake" ]]; then
    echo "Error: Environment must be 'snake'"
    echo "Usage: $0 [reinforce|dqn] [snake] [episodes] [additional_args...]"
    exit 1
fi

echo "Starting training with:"
echo "  Algorithm: $ALG"
echo "  Environment: $ENV"
echo "  Episodes: $EPISODES"
echo "  Additional args: ${@:4}"
echo ""

# Run the training
python -m viberl.cli train --alg "$ALG" --env "$ENV" --episodes "$EPISODES" "${@:4}"

echo ""
echo "Training completed!"
echo "Check experiments/ directory for results"
