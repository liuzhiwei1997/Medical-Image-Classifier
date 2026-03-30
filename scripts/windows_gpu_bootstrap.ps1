# Windows + NVIDIA GPU bootstrap script for Medical-Image-Classifier
# Run in PowerShell from repository root:
#   powershell -ExecutionPolicy Bypass -File .\scripts\windows_gpu_bootstrap.ps1

$ErrorActionPreference = "Stop"

Write-Host "1) Create Conda environment: class (Python 3.10)"
conda create -n class python=3.10 -y

Write-Host "2) Activate Conda environment: class"
conda activate class

Write-Host "3) Install CUDA runtime dependencies from conda-forge"
conda install -y -c conda-forge cudatoolkit=11.2 cudnn=8.1

Write-Host "4) Upgrade pip tooling"
python -m pip install --upgrade pip setuptools wheel

Write-Host "5) Install Windows GPU dependencies"
pip install -r requirements-win-gpu.txt

Write-Host "6) Verify TensorFlow and GPU visibility"
python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

Write-Host "7) Verify project CLI"
python .\main.py --help

Write-Host "8) Start training"
python .\main.py --phase TRAIN --config_json .\configs\config.json --gpu 0

Write-Host "Done."
