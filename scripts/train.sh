CUDA_VISIBLE_DEVICES=0 python run_train.py \
    --resume_path 'LOGS/LVIS_two_obj/lightning_logs/version_9/checkpoints/epoch=1-step=68319.ckpt' \
    --root_dir 'LOGS/LVIS_two_obj' \
    --batch_size 1 \
    --limit_train_batches 1 \
    --logger_freq 10 \
    --dataset_name lvis

CUDA_VISIBLE_DEVICES=0 python run_train_v2.py \
    --resume_path 'LOGS/LVIS_two_obj/lightning_logs/version_9/checkpoints/epoch=1-step=68319.ckpt' \
    --root_dir 'LOGS/LVIS_two_obj' \
    --batch_size 1 \
    --limit_train_batches 1 \
    --logger_freq 10 \
    --is_joint
