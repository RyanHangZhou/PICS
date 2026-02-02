#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for job submission on our clusters.
# ---------------------------------------------------------------------
#SBATCH --job-name=annotate_mapillary_vistas_train
#SBATCH --account=def-vislearn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH --mail-user=hang18@ualberta.ca
#SBATCH --mail-type=ALL

# ---------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"
echo "annotate_mapillary_vistas_train"
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
# Run your inference step here...
python -u -m datasets.mapillary_vistas \
    --dataset_dir "/home/hang18/links/scratch/data" \
    --construct_dataset_dir "data/train/MapillaryVistas" \
    --area_ratio 0.02 \
    --is_build_data \
    --is_train
