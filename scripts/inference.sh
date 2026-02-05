CUDA_VISIBLE_DEVICES=2 python run_test.py \
    --input "/data/hang/customization/data/Wild" \
    --output "results/LVIS_again3" \
    --obj_thr 2

CUDA_VISIBLE_DEVICES=2 python run_test.py \
    --input "/data/hang/customization/data/dreambooth_reorganized" \
    --output "results/dreambooth" \
    --obj_thr 2