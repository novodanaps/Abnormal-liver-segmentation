from glob import glob

import numpy as np

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    AddChannelD,
    Compose,
    EnsureChannelFirstD,
    LoadImageD,
    ScaleIntensityD,
    EnsureTypeD,
    RandFlipD,
    RandRotateD,
)


class ALSDataset:
    def __init__(self, image_size):
        self.h, self.w, self.z = image_size

    @staticmethod
    def create_dataset(
        image_folder: str,
        workers: int,
        training: bool = False
    ):
        image_paths = glob(image_folder + '/*.nii.gz')
        data_dic = [{"im": name} for name in image_paths]

        if training:
            transforms = Compose(
                [
                    LoadImageD(keys=["im"]),
                    EnsureChannelFirstD(keys=["im"]),
                    ScaleIntensityD(keys=["im"]),
                    RandRotateD(keys=["im"], range_x=np.pi / 12, prob=0.5, keep_size=True),
                    RandFlipD(keys=["im"], spatial_axis=0, prob=0.5),
                    EnsureTypeD(keys=["im"]),
                ]
            )
        else:
            transforms = Compose(
                [
                    LoadImageD(keys=["im"]),
                    EnsureChannelFirstD(keys=["im"]),
                    ScaleIntensityD(keys=["im"]),
                    EnsureTypeD(keys=["im"]),
                ]
            )

        dataset = CacheDataset(data_dic, transforms, num_workers=workers)
        return dataset

    @staticmethod
    def create_dataloader(
        dataset: CacheDataset,
        workers: int,
        batch_size: int = 1,
        training: bool = False
    ):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            num_workers=workers,
        )
        return dataloader

    @staticmethod
    def get_image(dataset: CacheDataset, idx: int):
        return dataset[idx]['img']
