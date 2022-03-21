import numpy as np
from scipy import signal

from PIL import Image


def img_to_np(image: Image):
    return np.array(image) / 255.0


def np_to_img(array: np.ndarray):
    return Image.fromarray((array * 255.0).astype(np.uint8))


def rescale(image: Image, scale=0.25):
    width, height = image.size
    width, height = round(width * scale), round(height * scale)
    return image.resize((width, height))


def gaussian_kernel(size=5, sigma=2):
    gkern1d = signal.gaussian(size, std=sigma)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d / gkern2d.sum()


def load_kernel(kernel_path):
    return img_to_np(Image.open(kernel_path))


def blur(image, kernel=gaussian_kernel()):
    return np_to_img(
        np.stack(
            [
                signal.convolve2d(img_to_np(channel), kernel)
                for channel in image.split()
            ], axis=-1
        )
    )


def gaussian_noise(image, loc=0.0, std=5e-2):
    (width, height), channels = image.size, len(image.split())
    noise = np.random.normal(loc, std, (height, width, channels))
    return np_to_img(np.clip(img_to_np(image) + noise, 0, 1))
