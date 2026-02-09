import os
import argparse
import pytorch_lightning as pl
from braceexpand import braceexpand
from torch.utils.data import DataLoader
from datasets.webdataset import MultiWebDataset

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from torch.utils.data import ConcatDataset
from cldm.hack import disable_verbosity, enable_sliced_attention
from omegaconf import OmegaConf
import torch

from datasets.base import BaseDataset

class BaseLogic(BaseDataset):
    def __init__(self, area_ratio, obj_thr):
        self.area_ratio = area_ratio
        self.obj_thr = obj_thr

print("Number of GPUs available: ", torch.cuda.device_count())
print("Current device: ", torch.cuda.current_device())
print("Device name: ", torch.cuda.get_device_name(0))

def get_args_parser():
    parser = argparse.ArgumentParser('PICS Training Script', add_help=False)

    parser.add_argument('--resume_path', required=None, type=str)
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

    model = create_model('./configs/pics.yaml').cpu()
    if args.resume_path and os.path.exists(args.resume_path):
        print(f"Loading checkpoint from: {args.resume_path}")
        checkpoint = load_state_dict(args.resume_path, location='cpu')
        model.load_state_dict(checkpoint, strict=False)
    else:
        print("No checkpoint found or provided. Training from scratch...")

    model.learning_rate = args.learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    DConf = OmegaConf.load('./configs/datasets.yaml')

    if args.is_joint:
        # weights = {'LVIS': 30, 'VITONHD': 60, 'Objects365': 1, 'Cityscapes': 180, 'MapillaryVistas': 180,'BDD100K': 180}
        weights = {'LVIS': 3, 'VITONHD': 6, 'Objects365': 1, 'Cityscapes': 18, 'MapillaryVistas': 18, 'BDD100K': 18}
    else:
        if args.dataset_name == 'lvis':
            weights = {'LVIS': 1, 'VITONHD': 0, 'Objects365': 0, 'Cityscapes': 0, 'MapillaryVistas': 0, 'BDD100K': 0}
        elif args.dataset_name == 'vitonhd':
            weights = {'LVIS': 0, 'VITONHD': 1, 'Objects365': 0, 'Cityscapes': 0, 'MapillaryVistas': 0, 'BDD100K': 0}
        elif args.dataset_name == 'object365':
            weights = {'LVIS': 0, 'VITONHD': 0, 'Objects365': 1, 'Cityscapes': 0, 'MapillaryVistas': 0, 'BDD100K': 0}
        elif args.dataset_name == 'cityscapes':
            weights = {'LVIS': 0, 'VITONHD': 0, 'Objects365': 0, 'Cityscapes': 1, 'MapillaryVistas': 0, 'BDD100K': 0}
        elif args.dataset_name == 'mapillaryvistas':
            weights = {'LVIS': 0, 'VITONHD': 0, 'Objects365': 0, 'Cityscapes': 0, 'MapillaryVistas': 1, 'BDD100K': 0}
        elif args.dataset_name == 'bdd100k':
            weights = {'LVIS': 0, 'VITONHD': 0, 'Objects365': 0, 'Cityscapes': 0, 'MapillaryVistas': 0, 'BDD100K': 1}
        else:
            raise ValueError(f"Unsupported dataset name: {args.dataset_name}")
        
    all_urls = []
    dataset_shards = [
        ('LVIS', DConf.Train.LVIS.shards),
        ('VITONHD', DConf.Train.VITONHD.shards),
        ('Objects365', DConf.Train.Objects365.shards),
        ('Cityscapes', DConf.Train.Cityscapes.shards),
        ('MapillaryVistas', DConf.Train.MapillaryVistas.shards),
        ('BDD100K', DConf.Train.BDD100K.shards)
    ]

    for name, path in dataset_shards:
        expanded = list(braceexpand(path))
        all_urls.extend(expanded * weights.get(name, 1))
    
    import random
    random.shuffle(all_urls)

    logic_helper = BaseLogic(
        area_ratio=DConf.Defaults.area_ratio, 
        obj_thr=DConf.Defaults.obj_thr
    )

    dataset = MultiWebDataset(
        urls=all_urls,
        construct_collage_fn=logic_helper._construct_collage, 
        shuffle_size=10000,
        seed=42,
        decode_mode="pil",
    )

    dataloader = DataLoader(
        dataset, 
        num_workers=8, 
        batch_size=args.batch_size, 
    )
    
    logger = ImageLogger(batch_frequency=args.logger_freq, log_images_kwargs=obj_thr)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.root_dir, 'checkpoints'),
        filename='pics-{step:06d}', 
        every_n_train_steps=2000, 
        save_top_k=-1, 
    )

    trainer = pl.Trainer(
        default_root_dir=args.root_dir,
        limit_train_batches=args.limit_train_batches,
        accelerator="gpu",
        devices=1,
        precision=16,
        callbacks=[logger, checkpoint_callback],
        accumulate_grad_batches=accumulate_grad_batches,
        max_epochs=50,
        val_check_interval=2000,
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PICS Training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

