#!/bin/bash
# download_frames_data.sh

# Exit immediately if a command exits with a non-zero status
set -e

DATA_DIR="./frames"
ZIP_URL="https://github.com/Igkho/ZeroHostCopyInference/releases/download/v1.0-frames-data/frames.zip"
ZIP_FILE="./frames.zip"

echo "[Data Fetcher] Initializing input data environment..."

mkdir -p "$DATA_DIR"

# Check if the directory already contains the JPEGs
if [ "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    echo "[Data Fetcher] Input frames already exist in $DATA_DIR. Skipping download."
    exit 0
fi

echo "[Data Fetcher] Downloading input frames from release..."
wget -q -O "$ZIP_FILE" "$ZIP_URL"

echo "[Data Fetcher] Extracting frames..."
# Assumes the zip contains the flat JPEGs or unzips them directly into the target folder
unzip -q -j "$ZIP_FILE" -d "$DATA_DIR"

echo "[Data Fetcher] Cleaning up..."
rm "$ZIP_FILE"

echo "[Data Fetcher] Success. Frames are ready for ZeroCopyInference tool in $DATA_DIR"