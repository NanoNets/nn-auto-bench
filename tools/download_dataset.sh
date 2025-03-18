#!/bin/bash

# Dataset information
dataset_id="nanonets/nn-auto-bench-ds"
download_path="data/"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null
then
    echo "Error: huggingface-cli is not installed."
    echo "Install it using: pip install huggingface_hub"
    exit 1
fi

# Handle existing download directory
if [ -d "$download_path" ]; then
  echo "Directory '$download_path' already exists."
  echo -n "Remove and re-download? (y/n) " # Use echo -n for prompt (no newline)
  read reply_line # Read the whole line of input (no -n 1)
  reply=$(echo "$reply_line" | head -c 1) # Extract first char
  reply=$(echo "$reply" | tr -d '[:space:]') # Trim whitespace
  echo # Add a newline after user input

  case "$reply" in # Use case statement for comparison
    [yY]) # Match 'y' or 'Y'
      rm -rf "$download_path"
      echo "Existing directory '$download_path' removed."
      ;;
    *) # Default case (anything else)
      echo "Using existing dataset in '$download_path'."
      exit 0
      ;;
  esac
fi

# Create directory and download
mkdir -p "$download_path"
echo "Downloading dataset '$dataset_id' to '$download_path'..."

huggingface-cli download "$dataset_id" \
    --repo-type dataset \
    --local-dir "$download_path" \
    --local-dir-use-symlinks False

if [ $? -eq 0 ]; then
    echo "Dataset downloaded successfully to: $download_path"
else
    echo "Error: Failed to download dataset '$dataset_id' to '$download_path'."
    exit 1
fi