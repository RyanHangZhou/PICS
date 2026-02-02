# visualize data
python -m datasets.lvis

# get large-intersection pair (test)
python -m datasets.lvis \
    --dataset_dir "/data/hang/customization/data" \
    --construct_dataset_dir "data/test/LVIS" \
    --area_ratio 0.02 \
    --is_build_data

# get large-intersection pair (train)
python -m datasets.lvis \
    --dataset_dir "/data/hang/customization/data" \
    --construct_dataset_dir "data/train/LVIS" \
    --area_ratio 0.02 \
    --is_build_data \
    --is_train

# verify dataloader (LVIS)
python -m datasets.lvis \
    --dataset_dir "/data/hang/customization/data" \
    --construct_dataset_dir "data/test/LVIS" \
    --area_ratio 0.02

# train
CUDA_VISIBLE_DEVICES=3,0 python run_train_coin.py
CUDA_VISIBLE_DEVICES=3,0,1,2 python run_train_coin.py






# Alliance-Rorqual

# LVIS
# get large-intersection pair (LVIS test)
python -m datasets.lvis \
    --dataset_dir "/home/hang18/links/scratch/data" \
    --construct_dataset_dir "data/test/LVIS" \
    --area_ratio 0.02 \
    --is_build_data

# get large-intersection pair (LVIS train)
python -m datasets.lvis \
    --dataset_dir "/home/hang18/links/scratch/data" \
    --construct_dataset_dir "data/train/LVIS" \
    --area_ratio 0.02 \
    --is_build_data \
    --is_train

# verify dataloader (LVIS)
python -m datasets.lvis \
    --dataset_dir "/data/hang/customization/data" \
    --construct_dataset_dir "data/test/LVIS" \
    --area_ratio 0.02

# visualize data (LVIS)
python -m visualize.visualize_two_objects_relation \
    --input_root "data/test/LVIS" \
    --output_dir "data/test/LVIS_concat"

# VITON-HD
# get large-intersection pair (VITON-HD test)
python -m datasets.vitonhd \
    --dataset_dir "/home/hang18/links/scratch/data" \
    --construct_dataset_dir "data/test/VITONHD" \
    --area_ratio 0.02 \
    --is_build_data

# get large-intersection pair (VITON-HD train)
python -m datasets.vitonhd \
    --dataset_dir "/home/hang18/links/scratch/data" \
    --construct_dataset_dir "data/train/VITONHD" \
    --area_ratio 0.02 \
    --is_build_data \
    --is_train

# visualize data (VITON-HD)
python -m visualize.visualize_two_objects_relation \
    --input_root "data/test/VITONHD" \
    --output_dir "data/test/VITONHD_concat"

# get large-intersection pair (Objects365 test)
python -m datasets.objects365 \
    --dataset_dir "/home/hang18/links/scratch/data" \
    --construct_dataset_dir "data/test/Objects365" \
    --area_ratio 0.02 \
    --is_build_data

# get large-intersection pair (Objects365 train)
python -m datasets.objects365 \
    --dataset_dir "/home/hang18/links/scratch/data" \
    --construct_dataset_dir "data/train/Objects365" \
    --area_ratio 0.02 \
    --is_build_data \
    --is_train

# train
python run_train_coin.py




# segment Objects365 dataset (train)
python scripts/annotate_sam.py --is_train
python scripts/annotate_sam.py --is_train # TODO
# segment Objects365 dataset (test)
python scripts/annotate_sam.py
python scripts/annotate_sam.py --index_low 0 --index_high 15000 # 10 hours on nvidia_h100_80gb_hbm3_1g.10gb:1




# data transfer
scp /Users/hangzhou/Downloads/Forest2Seq-main.zip hang18@rorqual.alliancecan.ca:/home/hang18/links/scratch/data






#### rebuttal
# 2-object LVIS testing data prep
python -m datasets.lvis \
    --dataset_dir "/data/hang/customization/data" \
    --construct_dataset_dir "data/test/LVIS" \
    --area_ratio 0.02 \
    --is_build_data

# 3-object LVIS testing data prep
python -m datasets.lvis \
    --dataset_dir "/data/hang/customization/data" \
    --construct_dataset_dir "data/test/LVIS_o3" \
    --area_ratio 0.02 \
    --is_build_data

# 2-object LVIS training data prep
python -m datasets.lvis \
    --dataset_dir "/data/hang/customization/data" \
    --construct_dataset_dir "data/train/LVIS" \
    --area_ratio 0.02 \
    --is_build_data \
    --is_train

# 3-object LVIS training data prep
python -m datasets.lvis \
    --dataset_dir "/data/hang/customization/data" \
    --construct_dataset_dir "data/train/LVIS_o3" \
    --area_ratio 0.02 \
    --is_build_data \
    --is_train

# train pics
CUDA_VISIBLE_DEVICES=3 python run_train.py \
    --resume_path '/data/hang/customization/AnyDoor/checkpoints/epoch=1-step=8687.ckpt' \
    --root_dir 'LOGS/LVIS_two_obj' \
    --batch_size 1 \
    --limit_train_batches 1 \
    --logger_freq 1000 \
    --dataset_name lvis

CUDA_VISIBLE_DEVICES=0 python run_train.py \
    --resume_path '/data/hang/customization/AnyDoor/checkpoints/epoch=1-step=8687.ckpt' \
    --root_dir 'LOGS/LVIS_two_obj' \
    --batch_size 1 \
    --limit_train_batches 1 \
    --logger_freq 1000 \
    --dataset_name lvis

# Nibi
python run_train.py \
    --resume_path 'checkpoints/epoch=1-step=8687.ckpt' \
    --root_dir 'LOGS/LVIS_two_obj' \
    --batch_size 1 \
    --limit_train_batches 1 \
    --logger_freq 1000 \
    --dataset_name lvis

# trillium
python run_train.py \
    --resume_path 'checkpoints/control_sd21_ini.ckpt' \
    --root_dir 'LOGS/LVIS_two_obj' \
    --batch_size 16 \
    --limit_train_batches 1 \
    --logger_freq 1000 \
    --dataset_name lvis


python run_train.py \
    --resume_path 'LOGS/LVIS_two_obj/lightning_logs/version_98541/checkpoints/epoch=0-step=4269.ckpt' \
    --root_dir 'LOGS/LVIS_two_obj' \
    --batch_size 8 \
    --limit_train_batches 1 \
    --logger_freq 1000 \
    --dataset_name lvis

python run_train.py \
    --resume_path 'LOGS/LVIS_two_obj/lightning_logs/version_4837708/checkpoints/epoch=0-step=4269.ckpt' \
    --root_dir 'LOGS/LVIS_two_obj' \
    --batch_size 8 \
    --limit_train_batches 1 \
    --logger_freq 1000 \
    --dataset_name lvis










