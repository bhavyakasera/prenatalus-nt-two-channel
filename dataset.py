import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class PreNatalSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None, seg_part='baby'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.seg_part = seg_part


        # Lists all the file in the FOLDER
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_path = os.path.join(self.image_dir, self.images[index])
        file_extension = Path(img_path).suffix
        seg_m = f'_{self.seg_part}_mask.jpg'

        seg_mask_path = os.path.join(self.mask_dir, self.images[index].replace(file_extension, seg_m))

        image = np.array(Image.open(img_path).convert("RGB"))
        # print(image)

        seg_mask = np.array(Image.open(seg_mask_path).convert("L"), dtype=np.float32)  # 0.0, 255.0
        seg_mask = seg_mask.reshape((1, seg_mask.shape[0], seg_mask.shape[1]))

        seg_mask[seg_mask > 0] = 1.0

        if self.transform is not None:
            transformed = (self.transform(image=image, mask=seg_mask))
            image = transformed["image"]
            seg_mask = transformed["mask"]

        seg_mask = (seg_mask > 0.0).float()

        return image, seg_mask, self.images[index].rsplit(".", 1)[0]


class PreNatalTwoChannelDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform


        # Lists all the file in the FOLDER
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        seg_m1 = '_baby_mask.jpg'
        seg_m2 = '_NT_mask.jpg'

        # print(f"Index: {self.images[index]}")

        img_path = os.path.join(self.image_dir, self.images[index])
        file_extension = Path(img_path).suffix

        seg1_mask_path = os.path.join(self.mask_dir, self.images[index].replace(file_extension, seg_m1))
        seg2_mask_path = os.path.join(self.mask_dir, self.images[index].replace(file_extension, seg_m2))

        image = np.array(Image.open(img_path).convert("RGB"))
        # print(image)

        seg1_mask = np.array(Image.open(seg1_mask_path).convert("L"), dtype=np.float32)  # 0.0, 255.0
        seg2_mask = np.array(Image.open(seg2_mask_path).convert("L"), dtype=np.float32)  # 0.0, 255.0

        seg1_mask[seg1_mask > 0] = 1.0  # Convert all white pixels to a 1
        seg2_mask[seg2_mask > 0] = 1.0

        if self.transform is not None:
            transformed = (self.transform(image=image, masks=[seg1_mask, seg2_mask]))
            image = transformed["image"]
            seg1_mask = transformed["masks"][0]
            seg2_mask = transformed["masks"][1]

        seg1_mask = (seg1_mask > 0.0).float()
        seg2_mask = (seg2_mask > 0.0).float()

        mask = np.stack((seg1_mask, seg2_mask), axis=0)

        return image, mask, self.images[index].rsplit(".", 1)[0]


class PreNatalTestDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None, bodysegmentation=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform

        self.bodysegmentation = bodysegmentation

        # Lists all the file in the FOLDER
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        anomaly_m = '_anomaly_mask.jpg'

        # print(f"Index: {self.images[index]}")

        img_path = os.path.join(self.image_dir, self.images[index])
        file_extension = Path(img_path).suffix

        anomaly_mask_path = os.path.join(self.mask_dir, self.images[index].replace(file_extension, anomaly_m))

        image = np.array(Image.open(img_path).convert("RGB"))
        # print(image)

        anomaly_mask = np.array(Image.open(anomaly_mask_path).convert("L"), dtype=np.float32)  # 0.0, 255.0

        anomaly_mask[anomaly_mask > 0] = 1.0  # Convert all white pixels to a 1

        if self.transform is not None:
            # augmentations = self.transform(image=image, anomaly_mask=anomaly_mask, baby_mask=baby_mask)
            # augmentations = self.transform(anomaly_mask=anomaly_mask, baby_mask=baby_mask)
            image = (self.transform(image=image))["image"]
            anomaly_mask = (self.mask_transform(image=anomaly_mask))["image"]

        anomaly_mask = (anomaly_mask > 0.0).float()

        mask = np.stack((anomaly_mask, anomaly_mask), axis=0)

        return image, mask, self.images[index].rsplit(".", 1)[0]

