# LVIS train set
python -m datasets.lvis \
    --dataset_dir "/path/to/raw_data" \
    --construct_dataset_dir "data/train/LVIS" \
    --area_ratio 0.02 \
    --is_build_data \
    --is_train

# LVIS test set
python -m datasets.lvis \
    --dataset_dir "/path/to/raw_data" \
    --construct_dataset_dir "data/test/LVIS" \
    --area_ratio 0.02 \
    --is_build_data

