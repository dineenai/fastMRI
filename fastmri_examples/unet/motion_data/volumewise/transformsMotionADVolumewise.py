"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# Ammended from transforms.py by Aine Dineen, Cusack Lab, 3/7/22

# rename target and motion:
# Target: still 
# Image: Motion
# If not cropping (initially attempt not to) the below required only to normalise the data
# Does it do anything else that I should be aware of?

# Motion and still should be treated the same now...... as the same types of np arrays
# NB REVIEW THIS!


from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import fastmri
import numpy as np
import torch



def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)

# no mask required: remove all references


# We do not want to crop our volumes if it can be avoided...
# Question: Does this default training use 2D or 3D data???

# All of our images are real images at this stage BUT we would prefer not to crop if possible
# Check the cropping dimentions and memeory soze of images used!

def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, Union[torch.Tensor], Union[torch.Tensor]]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std


class UnetSample(NamedTuple):
    """
    A subsampled image for U-Net reconstruction.

    Args:
        image: Subsampled image after inverse FFT.
        target: The target image (if applicable).
        mean: Per-channel mean values used for normalization.
        std: Per-channel standard deviations used for normalization.
        fname: File name.
        # slice_num: The slice index.
        max_value: Maximum image value.
    """

    # image: torch.Tensor
    # target: torch.Tensor

    motion: torch.Tensor
    # still: torch.Tensor
    target: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    fname: str
    # slice_num: int
    max_value: float


class UnetDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self
    ):
        """
        Args:
           
            None remaining.....

        """

    def __call__(
        self,
        # kspace: np.ndarray,
        motion: np.ndarray,
        # mask: np.ndarray,
        # target: np.ndarray,
        still: np.ndarray,
        attrs: Dict,
        fname: str,
        # slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            # kspace: Input k-space of shape (num_coils, rows, cols) for
            #     multi-coil data or (rows, cols) for single coil data.
            motion: replaces image
            # mask: Mask from the test dataset.
            still: replaces Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            # slice_num: Serial number of the slice.

        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """

        # print(f"Are we getting as far as this transform?") #NO when add val_dataloader! - YES SINCE STOPPED USING CACHE and load val files instead!!!!!

        # Question Do we need a max value? If so add to attributes, is it just the value max in the array?
        # check for max value
        # try removing - does not appear to make a difference? where is this vlue used?
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0
        # print(f'if max_value needed reinstate here!')
        

        motion, mean, std = normalize_instance(motion, eps=1e-11)

        motion_torch = to_tensor(motion)
        # target_torch = center_crop(target_torch, crop_size)
        

        # motion_torch = normalize(motion_torch, mean, std, eps=1e-11)
        motion_torch = motion_torch.clamp(-6, 6)

        target_torch = to_tensor(still)
            # target_torch = center_crop(target_torch, crop_size)
        target_torch = normalize(target_torch, mean, std, eps=1e-11)
        target_torch = target_torch.clamp(-6, 6)

        # Remove print statments as training :) 06/07/22
        # print('test tensors!') #prints 8 times before crashing

        # print(f"FILE: transformsMotionAD.py\n  target_torch {target_torch.shape}, device {target_torch.device}, dtype {target_torch.dtype}")
        # print(f"FILE: transformsMotionAD.py\n  motion_torch {motion_torch.shape}, device {motion_torch.device}, dtype {motion_torch.dtype}")

        # temporary solution 6/7/22 to address the below error
        #  I think it would be preferable to change weights instead if possible as we might be losing precision here?
        # RuntimeError: Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same
        target_torch = target_torch.to(torch.float32)
        motion_torch = motion_torch.to(torch.float32)

        # print(f"FILE: transformsMotionAD.py\n  target_torch {target_torch.shape}, device {target_torch.device}, dtype {target_torch.dtype}")
        # print(f"FILE: transformsMotionAD.py\n  motion_torch {motion_torch.shape}, device {motion_torch.device}, dtype {motion_torch.dtype}")


        # target_torch torch.Size([64, 44]), device cpu, dtype torch.float64
        # motion_torch torch.Size([64, 44]), device cpu, dtype torch.float64

        # Motion and still should be treated the same now...... as the same types of np arrays

        
        # print(f"data must pass through this transform? {motion}, {still}  ")

        target = still

        return UnetSample(
            motion=motion_torch,
            # still=target_torch,
            target=target_torch, #AttributeError: 'UnetSample' object has no attribute 'target'
            mean=mean,
            std=std,
            fname=fname,
            # slice_num=slice_num,
            max_value=max_value
        )

