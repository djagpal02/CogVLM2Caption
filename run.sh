#!/bin/bash
set -e

# Get the absolute path of the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Set the paths for the cache and data directories relative to the script directory
CACHE_DIR="$SCRIPT_DIR/cache"
DATA_DIR="$SCRIPT_DIR/data"

# Print the paths for verification
echo "Cache Directory: $CACHE_DIR"
echo "Data Directory: $DATA_DIR"

# Initialize default values
SKIP_BUILD=false
GPUS="all"

# Check for the --skip-build or --sb flag and set GPU variable if provided
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-build|--sb) SKIP_BUILD=true ;;
        *) GPUS="$1" ;;
    esac
    shift
done

# Create directories if they don't exist
mkdir -p "$CACHE_DIR"
mkdir -p "$DATA_DIR"

# Verify directories exist (after attempting to create them)
if [ ! -d "$CACHE_DIR" ]; then
    echo "Cache directory does not exist: $CACHE_DIR"
    exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
    echo "Data directory does not exist: $DATA_DIR"
    exit 1
fi

# Build the Docker image if the flag is not set
if [ "$SKIP_BUILD" == "false" ]; then
    echo "Building the Docker image..."
    hare build -t dkjj20/cogvlm2caption .
fi
 
# Run the Docker container with volume mappings
echo "Running Docker container with GPUs: $GPUS..."
hare run -it --rm --gpus "device=$GPUS" \
    -v "$CACHE_DIR:/app/cache:z" \
    -v "$DATA_DIR:/app/data:z" \
    dkjj20/cogvlm2caption