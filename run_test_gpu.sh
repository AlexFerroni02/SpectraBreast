#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --output=test_gpu_%j.out
#SBATCH --error=test_gpu_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=4G

module purge
# Modulo CESVIMA con PyTorch/1.10.0-foss-2021a-CUDA-11.3.1
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1

cd notebooks
echo "Checking GPU..."
srun python3 test_gpu.py
echo "Check GPU done!"