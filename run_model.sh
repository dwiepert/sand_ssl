#!/bin/bash
REPO_URL="https://github.com/dwiepert/sand_ssl.git"
REPO_DIR="sand_ssl"
PYTHON_SCRIPT="test.py"
DATA_FOLDER="/u/dwiepert/Documents/task1_train"
EPOCHS="1"
BATCH_SZ="2"
MODEL_TYPE="wavlm-large"
OUTPUT_FOLDER="/u/dwiepert/Documents/sand_results"

#clone repo
git clone "$REPO_URL"

# Check if cloning was successful
if [ $? -ne 0 ]; then
    echo "Error: Git clone failed. Exiting."
    exit 1
fi

#Go to repo dir
cd "$REPO_DIR"

pip install .
if [ $? -ne 0 ]; then
    echo "Error: pip install failed. Exiting."
    exit 1
fi

python "$PYTHON_SCRIPT" --data_dir="$DATA_FOLDER" --epochs="$EPOCHS" --model_type="$MODEL_TYPE" --batch_sz="$BATCH_SZ" --output_path="$OUTPUT_FOLDER"
if [ $? -ne 0 ]; then
    echo "Error: Python script execution failed."
    exit 1
fi