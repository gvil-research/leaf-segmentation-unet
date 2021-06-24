import numpy as np
from torchvision import transforms, datasets, models


def reverse_transform(inp):
    """
    Reverse transform a normal image(non-mask)
    after converting from torch tensor to numpy array
    """
    inp = inv_normalize(inp)
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp


def reverse_transform_mask(inp):
    """
    Reverse transform a image mask after
    converting from torch tensor to numpy array
    """
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
