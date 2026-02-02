
CUDA_VISIBLE_DEVICES=3 python run_single_3D.py


CUDA_VISIBLE_DEVICES=3 python run_single_3D_wild_rebuttal.py


# python


## FID
# python -m pytorch_fid /public/zhouhang/customization/GrabCompose/results/COCO_batch_multiobj_obj365/composed_resize /public/zhouhang/customization/GrabCompose/results/COCO_batch_multiobj_obj365/source_resize
# python -m pytorch_fid /public/zhouhang/customization/AnyDoor/results/COCO_batch_provide_ckpt_all/composed_resize /public/zhouhang/customization/GrabCompose/results/COCO_batch_multiobj_obj365/source_resize

