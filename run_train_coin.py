import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.lvis import LVISDataset
from datasets.viton_hd import VITONHDDataset

from datasets.cityscapes import CityscapesDataset

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from torch.utils.data import ConcatDataset
from cldm.hack import disable_verbosity, enable_sliced_attention
from omegaconf import OmegaConf
import torch

print("Number of GPUs available: ", torch.cuda.device_count())
print("Current device: ", torch.cuda.current_device())
print("Device name: ", torch.cuda.get_device_name(0))

n_gpus = torch.cuda.device_count()

save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

# resume_path = 'checkpoints/control_sd21_ini.ckpt'
# resume_path = 'lightning_logs/version_7/checkpoints/epoch=6-step=14945.ckpt'
resume_path = 'lightning_logs/version_8/checkpoints/epoch=16-step=12359.ckpt'

batch_size = 16
logger_freq = 1000
learning_rate = 1e-5
sd_locked = False
only_mid_control = False
n_gpus = 1
accumulate_grad_batches = 1
obj_thr = {'obj_thr': 2}

model = create_model('./configs/grabcompose.yaml').cpu()
checkpoint = load_state_dict(resume_path, location='cpu')

model.load_state_dict(checkpoint, strict=False)

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Datasets
DConf = OmegaConf.load('./configs/datasets.yaml')
dataset1 = LVISDataset(**DConf.Train.LVIS, area_ratio=DConf.Defaults.area_ratio, obj_thr=DConf.Defaults.obj_thr)
dataset2 = VITONHDDataset(**DConf.Train.VITONHD, area_ratio=DConf.Defaults.area_ratio, obj_thr=DConf.Defaults.obj_thr)

# dataset5 = CityscapesDataset(**DConf.Train.Cityscapes, area_ratio=DConf.Defaults.area_ratio, obj_thr=DConf.Defaults.obj_thr)

# image_data = [dataset1, dataset5]
# video_data = [dataset3, dataset4]
# tryon_data = [dataset2]

# dataset = ConcatDataset( image_data + video_data + tryon_data + video_data + tryon_data  )
# dataset = ConcatDataset( image_data )
# dataset = ConcatDataset( dataset1 + dataset2 )

dataloader = DataLoader(dataset1, num_workers=8, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq, log_images_kwargs=obj_thr)
trainer = pl.Trainer(strategy="ddp", precision=16, accelerator="auto", callbacks=[logger], accumulate_grad_batches=accumulate_grad_batches, max_epochs=50)


# Train!
trainer.fit(model, dataloader)

