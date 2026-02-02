import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.lvis import LVISDataset
from datasets.viton_hd import VITONHDDataset
from datasets.objects365 import Objects365Dataset
from datasets.cityscapes import CityscapesDataset
from datasets.mapillary_vistas import MapillaryVistasDataset
from datasets.bdd100k import BDD100KDataset

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from torch.utils.data import ConcatDataset
from cldm.hack import disable_verbosity, enable_sliced_attention
from omegaconf import OmegaConf
import torch

print("Number of GPUs available: ", torch.cuda.device_count())
print("Current device: ", torch.cuda.current_device())
print("Device name: ", torch.cuda.get_device_name(0))

def get_args_parser():
    parser = argparse.ArgumentParser('COIN Training Script', add_help=False)

    parser.add_argument('--resume_path', required=True, type=str)
    parser.add_argument('--root_dir', required=True, type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--limit_train_batches', default=1, type=float)
    parser.add_argument('--logger_freq', default=1000, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--is_joint', action='store_true', help="Joint/Seprate training")
    parser.add_argument("--dataset_name", type=str, default='lvis', help="Dataset name")

    return parser

def main(args):
    save_memory = False
    disable_verbosity()
    if save_memory:
        enable_sliced_attention()
    
    sd_locked = False
    only_mid_control = False
    accumulate_grad_batches = 1
    obj_thr = {'obj_thr': 2}

    model = create_model('./configs/grabcompose.yaml').cpu()

    
    checkpoint = load_state_dict(args.resume_path, location='cpu')
    model.load_state_dict(checkpoint, strict=False)


    # --- load & prune cond_stage weights from old ckpt (e.g., DINOv2-G/14) ---
    # ckpt = load_state_dict(args.resume_path, location='cpu')

    # # 有的 ckpt 外面包了一层 'state_dict'，统一展开
    # state = ckpt.get('state_dict', ckpt)

    # # 去掉 DDP 前缀（如 'module.'）
    # state = {k.replace('module.', ''): v for k, v in state.items()}

    # # 丢弃旧的 cond stage 参数，避免 1536↔1280 维度冲突
    # DROP_PREFIX = ('cond_stage_model.',)
    # pruned = {k: v for k, v in state.items() if not k.startswith(DROP_PREFIX)}

    # missing, unexpected = model.load_state_dict(pruned, strict=False)
    # print(f"[load] kept={len(pruned)} dropped={len(state)-len(pruned)} | "
    #       f"missing={len(missing)} unexpected={len(unexpected)}")

    # # 可选：看看丢了哪些 key（只打印前 10 个）
    # dropped = [k for k in state.keys() if k.startswith(DROP_PREFIX)]
    # print("[load] dropped keys (sample):", dropped[:10])



    model.learning_rate = args.learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Datasets
    DConf = OmegaConf.load('./configs/datasets.yaml')
    dataset1 = LVISDataset(**DConf.Train.LVIS, area_ratio=DConf.Defaults.area_ratio, obj_thr=DConf.Defaults.obj_thr)
    dataset2 = VITONHDDataset(**DConf.Train.VITONHD, area_ratio=DConf.Defaults.area_ratio, obj_thr=DConf.Defaults.obj_thr)
    dataset3 = Objects365Dataset(**DConf.Train.Objects365, area_ratio=DConf.Defaults.area_ratio, obj_thr=DConf.Defaults.obj_thr)
    dataset4 = CityscapesDataset(**DConf.Train.Cityscapes, area_ratio=DConf.Defaults.area_ratio, obj_thr=DConf.Defaults.obj_thr)
    dataset5 = MapillaryVistasDataset(**DConf.Train.MapillaryVistas, area_ratio=DConf.Defaults.area_ratio, obj_thr=DConf.Defaults.obj_thr)
    dataset6 = BDD100KDataset(**DConf.Train.BDD100K, area_ratio=DConf.Defaults.area_ratio, obj_thr=DConf.Defaults.obj_thr)

    if args.is_joint:
        dataset = ConcatDataset( [dataset1, dataset2, dataset3, dataset4, dataset5, dataset6] )
    else:
        if args.dataset_name == 'lvis':
            dataset = dataset1
        elif args.dataset_name == 'vitonhd':
            dataset = dataset2
        elif args.dataset_name == 'object365':
            dataset = dataset3
        elif args.dataset_name == 'cityscapes':
            dataset = dataset4
        elif args.dataset_name == 'mvd':
            dataset = dataset5
        elif args.dataset_name == 'bdd100k':
            dataset = dataset6
        elif args.dataset_name == 'small':
            dataset = ConcatDataset( [dataset1, dataset2, dataset4, dataset5, dataset6] )
        else:
            raise ValueError(f"Unsupported dataset name: {args.dataset_name}")

    dataloader = DataLoader(dataset, num_workers=8, batch_size=args.batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=args.logger_freq, log_images_kwargs=obj_thr)
    # trainer = pl.Trainer(default_root_dir=args.root_dir, limit_train_batches=args.limit_train_batches, strategy="ddp", precision=16, accelerator="auto", callbacks=[logger], accumulate_grad_batches=accumulate_grad_batches, max_epochs=50)
    trainer = pl.Trainer(default_root_dir=args.root_dir, limit_train_batches=args.limit_train_batches, gpus=1, precision=16, accelerator="auto", callbacks=[logger], accumulate_grad_batches=accumulate_grad_batches, max_epochs=50)
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('COIN Training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

