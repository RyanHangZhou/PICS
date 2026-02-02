#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for job submission on our clusters.
# ---------------------------------------------------------------------
#SBATCH --job-name=test_gen_obj365_anno_val_0_15k
#SBATCH --account=def-vislearn
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH --mail-user=hang18@ualberta.ca
#SBATCH --mail-type=ALL

# ---------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
echo "test_gen_obj365_anno_val_0_15k"
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
# Run your inference step here...
python scripts/annotate_sam.py --index_low 0 --index_high 15000
