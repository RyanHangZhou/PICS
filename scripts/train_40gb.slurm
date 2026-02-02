#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for job submission on our clusters.
# ---------------------------------------------------------------------
#SBATCH --job-name=train_all
#SBATCH --account=rrg-vislearn
#SBATCH --nodes=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4-00:00
#SBATCH --mem-per-cpu=60G
#SBATCH --mail-user=hang18@ualberta.ca
#SBATCH --mail-type=ALL

# ---------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
# Run your inference step here...
python run_train.py \
    --resume_path 'lightning_logs/version_82/checkpoints/epoch=1-step=123588.ckpt' \
    --batch_size 4 \
    --logger_freq 1000 \
    --is_joint
