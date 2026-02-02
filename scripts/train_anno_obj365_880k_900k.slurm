#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for job submission on our clusters.
# ---------------------------------------------------------------------
#SBATCH --job-name=train_gen_obj365_anno_val_880k_900k
#SBATCH --account=def-vislearn
#SBATCH --nodes=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH --mail-user=hang18@ualberta.ca
#SBATCH --mail-type=ALL

# ---------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
echo "train_gen_obj365_anno_val_880k_900k"
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
# Run your inference step here...
python -u scripts/annotate_sam.py --index_low 880000 --index_high 900000 --is_train
