#!/bin/bash

# Demo script for VibeRL
# Usage: ./examples/demo.sh [algorithm] [environment] [episodes]

ALG=${1:-random}
ENV=${2:-snake}
EPISODES=${3:-5}

echo "Running demo with:"
echo "  Algorithm: $ALG"
echo "  Environment: $ENV"
echo "  Episodes: $EPISODES"
echo ""

if [[ "$ALG" == "random" ]]; then
    # Run demo with random actions
    python -m viberl.cli demo --env "$ENV" --episodes "$EPISODES"
else
    echo "Currently only 'random' algorithm is supported for demo mode"
    echo "Use: ./examples/demo.sh random snake 5"
fi
