from torchvision import transforms
from torch.utils.data import Dataset

from .common import load_images, get_filelist
from .transforms import gaussian_noise

import numpy as np


class NoisyDataset(Dataset):
    def __init__(self, img_dir, transform=None, return_std=False):
        self.img_dir = img_dir
        self.images = load_images(
            get_filelist(self.img_dir),
            transform=transform
        )
        self.return_std = return_std

        self.img_to_torch = transforms.PILToTensor()
        self.torch_to_img = transforms.ToPILImage()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        std = np.random.randint(10, 56) / 255.0
        noisy = gaussian_noise(self.torch_to_img(self.images[idx]), std=std)

        if not self.return_std:
            return self.img_to_torch(noisy) / 255.0, self.images[idx]
        else:
            # need to be done for projection layer
            raise NotImplementedError()
