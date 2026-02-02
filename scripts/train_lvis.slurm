#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for job submission on our clusters.
# ---------------------------------------------------------------------
#SBATCH --job-name=train_LVIS
#SBATCH --account=def-vislearn
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00
#SBATCH --mem-per-cpu=50G
#SBATCH --mail-user=hang18@ualberta.ca
#SBATCH --mail-type=ALL

# ---------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
# Run your inference step here...
python run_train.py \
    --resume_path 'lightning_logs/version_30/checkpoints/epoch=0-step=34160.ckpt' \
    --batch_size 8 \
    --logger_freq 1000 \
    --dataset_name lvis
