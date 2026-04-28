#!/bin/bash

#SBATCH --account=ai
#SBATCH --job-name=aerturk23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH -c 4
#SBATCH --output=output/job_%j/output.log
#SBATCH --error=output/job_%j/output.err

# 1. Load Cluster Modules
module load anaconda3/2025.06
module load cuda/12.1.1
module load cudnn/9.10.2
module load git/2.9.5

# 2. Activate Conda Environment
ENV_PATH="/home/aerturk23/.conda/envs/bach_qnlp"

echo "--- Path ---"
echo "Python: $ENV_PATH/bin/python"
$ENV_PATH/bin/python -c "import torch; print('Torch:', torch.__version__)"
echo "--------------------------"

echo "--- Starting Classical Training ---"
$ENV_PATH/bin/python bach_lstm.py --data_path ./bach_measure_dataset.json --epochs 100 --loss_dir output/job_$SLURM_JOB_ID
$ENV_PATH/bin/python bach_lstm_generator.py

# echo "--- Starting Quantum Training (12 Qubits) ---"
# $ENV_PATH/bin/python bach_qnlp.py --data_path ./bach_measure_dataset.json --epochs 100 --output_dir output/job_$SLURM_JOB_ID

echo "Job finished at: $(date)"