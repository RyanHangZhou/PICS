# train on the whole dataset
python run_train.py \
    --root_dir 'LOGS/all_data' \
    --batch_size 1 \
    --limit_train_batches 1 \
    --logger_freq 1000 \
    --is_joint

python run_train.py \
    --root_dir 'LOGS/lvis' \
    --batch_size 1 \
    --limit_train_batches 1 \
    --logger_freq 1000 \
    --dataset_name lvis
