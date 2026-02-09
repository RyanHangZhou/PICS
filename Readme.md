<h1 align="center">PICS: Pairwise Image Compositing with Spatial Interactions</h1>
<p align="center"><img src="assets/figure.jpg" width="100%"></p>

***Check out our [Project Page](https://ryanhangzhou.github.io/pics/) for more visual demos!***

<!-- Updates -->
## ⏩ Updates

**02/08/2026**
- Release training and inference code.

<!-- **06/24/2025**
- Release inference code and data. -->

<!-- TODO List -->
## 🚧 TODO List
- [x] Release training and inference code
- [ ] Release pretrained models
- [ ] Release dataset
- [ ] Release any-object compositing code

<!-- Installation -->
## 📦 Installation

### Prerequisites
- **System**: The code is currently tested only on **Linux**. 
- **Hardware**: An NVIDIA GPU with at least 40GB of memory is necessary. The code has been verified on NVIDIA A6000 GPUs.  
- **Software**:   
  - [Conda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) is recommended for managing dependencies.  
  - Python version 3.6 or higher is required. 

  Create a new conda environment named `PICS` and install the dependencies: 
  ```
  conda env create --file=PICS.yml
  ```
  Download the ViT-g/14 version of DINOv2 for object feature extraction [here](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth) and put it in the directory ```checkpoints/dinov2_vitg14_pretrain.pth```. 

Download pretrained model
```
mkdir checkpoints

```



<!-- Pretrained Models -->
## 🤖 Pretrained Models

<!-- We provide the following pretrained models:

| Model | Description | #Params | Download |
| --- | --- | --- | --- |
| BOOTPLACE_Cityscapes | Multiple supervision | 523M | [Download](https://drive.google.com/file/d/1OeCourPQf1a6yM2BYNNuUKI3yvXRcD_N/view?usp=drive_link) | -->


<!-- Usage -->
<!-- ## 💡 Usage

### Minimal Example

Here is an [example](test.py) of how to use the pretrained models for object placement. -->


<!-- Dataset -->
## 📚 Dataset
<!-- We use the data from Cityscapes and [OPA](https://github.com/bcmi/Object-Placement-Assessment-Dataset-OPA). Note that we have provided instructions to prepare customized Cityscapes dataset for object composition in supplementary material. -->
<!-- We provide **TRELLIS-500K**, a large-scale dataset containing 500K 3D assets curated from [Objaverse(XL)](https://objaverse.allenai.org/), [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future), [HSSD](https://huggingface.co/datasets/hssd/hssd-models), and [Toys4k](https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k), filtered based on aesthetic scores. Please refer to the [dataset README](DATASET.md) for more details. -->
<!-- We provide a large-scale street-scene vehicle placement dataset [Download](https://drive.google.com/file/d/1wOzpMPy3Vy0tdBRD0xC1eW3SO2aVeCVX/view?usp=sharing) curated from [Cityscapes](https://www.cityscapes-dataset.com/). 
The file structures are: 
```
├── train
    ├── backgrounds:
        ├── imgID.png
        ├── ……
    ├── objects:
        ├── imgID:
            ├── object_name_ID.png
            ├── ……
        ├── ……
    ├── location:
        ├── imgID:
            ├── object_name_ID.txt
            ├── ……
        ├── ……
    ├── annotations.json
├── test
    ├── backgrounds:
        ├── imgID.png
        ├── ……
    |── backgrounds_single
        ├── imgID.png
        ├── ……
    ├── objects:
        ├── imgID:
            ├── object_name_ID.png
            ├── ……
        ├── ……
    ├── objects_single:
        ├── imgID:
            ├── object_name_ID.png
            ├── ……
        ├── ……
    ├── location:
        ├── imgID:
            ├── object_name_ID.txt
            ├── ……
        ├── ……
    ├── location_single:
        ├── imgID:
            ├── object_name_ID.txt
            ├── ……
        ├── ……
    ├── annotations.json
``` -->


## Training

<!-- To train a model on Cityscapes:
```
python -m main \
    --epochs 200 \
    --batch_size 2 \
    --save_freq 10 \
    --set_cost_class 1 \
    --ce_loss_coef 1 \
    --num_queries 120 \
    --eos_coef 0.1 \
    --lr 1e-4 \
    --data_path data/Cityscapes \
    --output_dir results/Cityscapes_ckpt \
    --resume weights/detr-r50-e632da11.pth
``` -->

## Inference
<!-- ```
python test.py \
    --num_queries 120 \
    --data_path data/Cityscapes \
    --pretrained_model 'results/Cityscapes_ckpt/checkpoint.pth' \
    --im_root 'data/Cityscapes/test' \
    --output_dir 'results/Cityscape_inference'
```
 -->



<!-- License -->
## ⚖️ License

This project is licensed under the terms of the MIT license.



<!-- Citation -->
## 📜 Citation

<!-- If you find this work helpful, please consider citing our paper: -->

<!-- ```bibtex
@inproceedings{zhou2025bootplace,
  title={BOOTPLACE: Bootstrapped Object Placement with Detection Transformers},
  author={Zhou, Hang and Zuo, Xinxin and Ma, Rui and Cheng, Li},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={19294--19303},
  year={2025}
}
``` -->