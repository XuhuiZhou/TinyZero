#!/bin/bash
#SBATCH --job-name=train_qwen        # Job name
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=1                   # Run a single task		
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --mem=64G                    # Job memory request
#SBATCH --time=36:00:00             # Time limit hrs:min:sec
#SBATCH --output=log_%j.out          # Standard output log
#SBATCH --error=log_%j.err           # Standard error log
#SBATCH --partition=preempt         # Add partition name
#SBATCH --gres=gpu:A100_80GB:2      # Request 2 A100 GPUs
##SBATCH --gres=gpu:h100:2          # Uncomment this and comment above line for H100s

# Load any necessary modules here
module purge
module load cuda/11.8

# Set any environment variables if needed
export CUDA_VISIBLE_DEVICES=0,1

# Run the training script
bash run.sh 
