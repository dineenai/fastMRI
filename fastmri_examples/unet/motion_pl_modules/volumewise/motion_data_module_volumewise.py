"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, Union

import fastmri
import pytorch_lightning as pl
import torch
# from fastmri.data import CombinedSliceDataset, SliceDataset
# from fastmri.data import CombinedSliceDataset, SliceDataset
# from motion_data.motion_mri_data import CombinedSliceDataset, SliceDataset
from motion_data.volumewise.motion_mri_data_volumewise import CombinedSliceDataset, SliceDataset

def worker_init_fn(worker_id):
    """Handle random seeding for all mask_func."""
    worker_info = torch.utils.data.get_worker_info()
    data: Union[
        SliceDataset, CombinedSliceDataset
    ] = worker_info.dataset  # pylint: disable=no-member

    # Check if we are using DDP
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    # Can i remove all of the below as no mask?
    # if isinstance(data, CombinedSliceDataset):
    #     for i, dataset in enumerate(data.datasets):
    #         if dataset.transform.mask_func is not None:
    #             if (
    #                 is_ddp
    #             ):  # DDP training: unique seed is determined by worker, device, dataset
    #                 seed_i = (
    #                     base_seed
    #                     - worker_info.id
    #                     + torch.distributed.get_rank()
    #                     * (worker_info.num_workers * len(data.datasets))
    #                     + worker_info.id * len(data.datasets)
    #                     + i
    #                 )
    #             else:
    #                 seed_i = (
    #                     base_seed
    #                     - worker_info.id
    #                     + worker_info.id * len(data.datasets)
    #                     + i
    #                 )
    #             dataset.transform.mask_func.rng.seed(seed_i % (2 ** 32 - 1))
    # elif data.transform.mask_func is not None:
    #     if is_ddp:  # DDP training: unique seed is determined by worker and device
    #         seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
    #     else:
    #         seed = base_seed
    #     data.transform.mask_func.rng.seed(seed % (2 ** 32 - 1))


class FastMriDataModule(pl.LightningDataModule):
    """
    Data module class for fastMRI data sets.

    This class handles configurations for training on fastMRI data. It is set
    up to process configurations independently of training modules.

    Note that subsampling mask and transform configurations are expected to be
    done by the main client training scripts and passed into this data module.

    For training with ddp be sure to set distributed_sampler=True to make sure
    that volumes are dispatched to the same GPU for the validation loop.
    """

    def __init__(
        self,
        data_path: Path,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        combine_train_val: bool = False,
        test_path: Optional[Path] = None,
        volume_sample_rate: Optional[float] = None,
        use_dataset_cache_file: bool = True,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
    ):
        """
        Args:
            data_path: Path to root data directory. For example, if knee/path
                is the root directory with subdirectories multicoil_train and
                multicoil_val, you would input knee/path for data_path.
            train_transform: A transform object for the training split.
            val_transform: A transform object for the validation split.
            test_transform: A transform object for the test split.
            combine_train_val: Whether to combine train and val splits into one
                large train dataset. Use this for leaderboard submission.
            test_path: An optional test path. Passing this overwrites data_path
                and test_split.
            volume_sample_rate: Fraction of volumes of the training data split to use. Can be
                set to less than 1.0 for rapid prototyping. If not set, it defaults to 1.0.
                To subsample the dataset either set sample_rate (sample by slice) or
                volume_sample_rate (sample by volume), but not both.
            use_dataset_cache_file: Whether to cache dataset metadata. This is
                very useful for large datasets like the brain data.
            batch_size: Batch size.
            num_workers: Number of workers for PyTorch dataloader.
            distributed_sampler: Whether to use a distributed sampler. This
                should be set to True if training with ddp.
        """
        super().__init__()

        self.data_path = data_path
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.combine_train_val = combine_train_val
        self.test_path = test_path
        self.volume_sample_rate = volume_sample_rate
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def _create_data_loader(
        self,
        data_transform: Callable,
        data_partition: str,
        volume_sample_rate: Optional[float] = None,
    ) -> torch.utils.data.DataLoader:
        
        print(f"FILE: motion_data_module.py\n  A data loader is being created!\n  The data partition is: {data_partition}")
        
        # TRY ARTIFICALLY SETTING! 
        # Get as far as: FILE: motion_unet_module.py\n  validation_step is running!
        # data_partition = "train" #RuntimeError: Expected key target in dict returned by validation_step.
        
        if data_partition == "train":
            is_train = True
            volume_sample_rate = (
                self.volume_sample_rate
                if volume_sample_rate is None
                else volume_sample_rate
            )
        else:
            is_train = False
            volume_sample_rate = None  # default case, no subsampling
        

        # if desired, combine train and val together for the train split
        dataset: Union[SliceDataset, CombinedSliceDataset]
        if is_train and self.combine_train_val:
            data_paths = [
                self.data_path / "train",
                self.data_path / "val",
            ]
            data_transforms = [data_transform, data_transform]
            volume_sample_rates = None  # default: no subsampling
            if volume_sample_rate is not None:
                volume_sample_rates = [volume_sample_rate, volume_sample_rate]
            dataset = CombinedSliceDataset(
                roots=data_paths,
                transforms=data_transforms,
                volume_sample_rates=volume_sample_rates,
                use_dataset_cache=self.use_dataset_cache_file,
            )
        else:
            if data_partition in ("test") and self.test_path is not None:
                data_path = self.test_path
            else:
                data_path = self.data_path / f"{data_partition}"
                print(f"FILE: motion_data_module.py\n  THIS PARTITION {data_path}")

            # ADD MANUALLY!
            # sample_rate = None

            # MANUALLY SET TO FALSE: DO FILES LOAD NOW?
            # use_dataset_cache = False #DOES NOT WORK HERE - set in SliceDataset
            # THIS SEEMS TO WORK - INVESITGATE EMPTY CACHE AT LATER TIMEPOINT!!!
            # SET self.use_dataset_cache_file {self.use_dataset_cache_file} TO FALSE PROPERLY!!!!!!!
            # MANUAL OVERWRITE FOR NOW!

            dataset = SliceDataset(
                root=data_path,
                transform=data_transform,
                volume_sample_rate=volume_sample_rate,
                # use_dataset_cache=self.use_dataset_cache_file
                use_dataset_cache=False
            )

            # KEY QUESTION: WHY IS THIS DATA SET FROM THIS DATA PATH RETURNING EMPTY!!!!!
            print(f"dataset: {dataset}, root {data_path}, volume_sample_rate {volume_sample_rate} ")
            # for val:
            # dataset: <motion_data.motion_mri_data.SliceDataset object at 0x7f79fbc86590>, root /data2/motion_dnns/motion_data/val, sample_rate 1.0, volume_sample_rate None, self.use_dataset_cache_file True
            #  for train:
            # dataset: <motion_data.motion_mri_data.SliceDataset object at 0x7f8162fdff90>, root /data2/motion_dnns/motion_data/train, sample_rate None, volume_sample_rate None, self.use_dataset_cache_file True 

            # ataset: <motion_data.motion_mri_data.SliceDataset object at 0x7f1c7c6cca50>, root /data2/motion_dnns/motion_data/val, sample_rate None, volume_sample_rate None, self.use_dataset_cache_file True 

            # print(f"data_transform is: {data_transform}")
            #val:   <motion_data.transformsMotionAD.UnetDataTransform object at 0x7f6b230ccc10>
            #train: <motion_data.transformsMotionAD.UnetDataTransform object at 0x7fd04a3dbdd0>

            print("FILE: motion_data_module.py\n  PROBLEM WITH DATASET =  SliceDataset????DEFINED IN motion_mri_data.py LOOK HERE!")
            print(f"dataset {dataset}") #dataset <motion_data.motion_mri_data.SliceDataset object at 0x7f8fb0f9e6d0>
            print(f"length of dataset {len(dataset)}") 
            # Is the problem the dataset SliceDataset classs or how we are initialising the object here?

        # ensure that entire volumes go to the same GPU in the ddp setting
        sampler = None
        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset)
            else:
                sampler = fastmri.data.VolumeSampler(dataset, shuffle=False)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
        )

        print(f"FILE: motion_data_module.py\n  About to return dataloader - creation was sucessful?\n  Data path is: {data_path}")
        print(f"  length of dataloader: {len(dataloader)}") #length of dataloader: 0
        print(f"  dataset: {len(dataset)}")  #EMPTY! - problem is before the data loader is called!
        # TO DO: FIND WHERE THE DATALOADER IS CALLED!
        # TRAIN RETURNS: length of dataloader: 4752
        # TRAIN RETURNS: dataset: 9504

        # Where is the data loaded to the dataloader, where is called next? 
        # TEST: Remove val data loader and check what is returned for train here!!
        return dataloader

    def prepare_data(self):
        # call dataset for each split one time to make sure the cache is set up on the
        # rank 0 ddp process. if not using cache, don't do this
        
        print(f"FILE: motion_data_module.py\n  prepare_data: Data is being prepared")
        if self.use_dataset_cache_file:
            if self.test_path is not None:
                test_path = self.test_path
            else:
                test_path = self.data_path / "test"
            data_paths = [
                self.data_path / "train",
                self.data_path / "val",
                test_path,
            ]
            data_transforms = [
                self.train_transform,
                self.val_transform,
                # TRY THIS 7/7/22
                # self.val_transform = self.train_transform, #SyntaxError: invalid syntax
                self.test_transform,
            ]
            for i, (data_path, data_transform) in enumerate(
                zip(data_paths, data_transforms)
            ):
                volume_sample_rate = self.volume_sample_rate if i == 0 else None
                _ = SliceDataset(
                    root=data_path,
                    transform=data_transform,
                    volume_sample_rate=volume_sample_rate,
                    use_dataset_cache=self.use_dataset_cache_file,
                )
        print(f"FILE: motion_data_module.py\n  prepare_data: data_paths: {data_paths}")
        # All three seem fine!:
        # data_paths: [PosixPath('/data2/motion_dnns/motion_data/train'), PosixPath('/data2/motion_dnns/motion_data/val'), PosixPath('/data2/motion_dnns/motion_data/test')]

    def train_dataloader(self):
        print(f'FILE: motion_data_module.p\yn  train_dataloader!') #Does not print!
        return self._create_data_loader(self.train_transform, data_partition="train")

    # Problem with val data loader.....: TMP Removed!
    # def val_dataloader(self):
    #     return self._create_data_loader(
    #         self.val_transform, data_partition="val", sample_rate=1.0
    #     )
    # def val_dataloader(self):
    #     return self._create_data_loader(self.val_transform, data_partition="val")

    # Original val_dataloader
    def val_dataloader(self):
        print(f'FILE: motion_data_module.p\yn  val_dataloader!') #This is the last thing that prints!
        
        # Path is fine!!
        # data_partition="val"
        # data_path = self.data_path / f"{data_partition}"
        # print(f"val data path: {data_path}")
        
        # Check the val_transform: val_transform
        # Where is this defined?
        print(f"self.val_transform: {self.val_transform}") # <motion_data.transformsMotionAD.UnetDataTransform object at 0x7f3b20e93550>
        # Where is this object assigned?
        # 
        print(f'self.batch_size: { self.batch_size}') #self.batc_size: 1

        # Where is this called, must be problem with passing the data to it...

        # TRY REPLACING VAL_transforw wit train: - DID NOT WORK
        # return self._create_data_loader(
        #     self.val_transform, data_partition="val", sample_rate=1.0
        # )
        # return self._create_data_loader(
        #     self.train_transform, data_partition="val"
        # )
        return self._create_data_loader(
            self.train_transform, data_partition="val"
        )   
        
       
    def test_dataloader(self):
        return self._create_data_loader(
            self.test_transform
        )

    @staticmethod
    def add_data_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # dataset arguments
        parser.add_argument(
            "--data_path",
            default=None,
            type=Path,
            help="Path to fastMRI data root",
        )
        parser.add_argument(
            "--test_path",
            default=None,
            type=Path,
            help="Path to data for test mode. This overwrites data_path and test_split",
        )
        parser.add_argument(
            "--volume_sample_rate",
            default=None,
            type=float,
            help="Fraction of volumes of the dataset to use (train split only). If not given all will be used. Cannot set together with sample_rate.",
        )
        parser.add_argument(
            "--use_dataset_cache_file",
            default=True,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
        )
        parser.add_argument(
            "--combine_train_val",
            default=False,
            type=bool,
            help="Whether to combine train and val splits for training",
        )

        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="Number of workers to use in data loader",
        )

        return parser
