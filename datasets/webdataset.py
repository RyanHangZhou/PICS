import webdataset as wds
from torch.utils.data import IterableDataset
from PIL import Image
import numpy as np
import cv2

class MultiWebDataset(IterableDataset):
    def __init__(
        self,
        urls,
        construct_collage_fn,
        shuffle_size=0,
        seed=0,
        decode_mode="pil",
    ):
        super().__init__()
        self.urls = urls
        self.shuffle_size = shuffle_size
        self.seed = seed
        self.decode_mode = decode_mode
        self.construct_collage_fn = construct_collage_fn

    def _to_rgb_np(self, img):
        if isinstance(img, Image.Image):
            return np.array(img.convert("RGB"))
        elif isinstance(img, np.ndarray):
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if img.ndim == 3 and img.shape[2] == 4:
                return img[:, :, :3]
            return img
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

    def _to_mask_np(self, img):
        if isinstance(img, Image.Image):
            m = np.array(img.convert("L"))
        elif isinstance(img, np.ndarray):
            if img.ndim == 3:
                m = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                m = img
        else:
            raise TypeError(f"Unsupported mask type: {type(img)}")
        m = (m > 127).astype(np.uint8) * 255
        return m

    def __iter__(self):
        ds = wds.WebDataset(self.urls, shardshuffle=True, empty_check=False)

        if self.shuffle_size and self.shuffle_size > 0:
            ds = ds.shuffle(self.shuffle_size)

        ds = ds.decode("pil")

        ds = ds.rename(
            bg="bg.jpg",
            obj0="obj0.png",
            mask0="mask0.png",
            obj1="obj1.png",
            mask1="mask1.png",
        )

        for sample in ds:
            bg    = sample["bg"]
            obj0  = sample["obj0"]
            obj1  = sample["obj1"]
            mask0 = sample["mask0"]
            mask1 = sample["mask1"]

            bg_np    = self._to_rgb_np(bg)
            obj0_np  = self._to_rgb_np(obj0)
            obj1_np  = self._to_rgb_np(obj1)
            mask0_np = self._to_mask_np(mask0)
            mask1_np = self._to_mask_np(mask1)

            collage = self.construct_collage_fn(
                bg_np, obj0_np, obj1_np, mask0_np, mask1_np
            )
            yield collage
