import os
import shutil
from PIL import Image


IMG_EXTS = ['.png', '.jpg', '.jpeg', '.gif', '.tif']


def get_filelist(input_dir: str, extensions=IMG_EXTS):
    '''
    '''
    return sorted([
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if os.path.splitext(filename)[-1].lower() in extensions
    ])


def load_images(filelist, transform=None):
    '''
    '''
    images = [Image.open(filename) for filename in filelist]
    return images if transform is None else [
        transform(image) for image in images
    ]


def save_images(images, filenames, target_ext=None):
    '''
    '''
    for image, filename in zip(images, filenames):
        if target_ext is not None:
            filename = os.path.splitext(filename)[0] + target_ext
        image.save(filename)


def enumerate_filenames(filelist, target_ext=None, start_idx=0):
    '''
    '''

    ext_list = [target_ext] * len(filelist) if target_ext is not None else [
        os.path.splitext(filename)[1]
        for filename in filelist
    ]

    return [
        f'{(start_idx+idx):08d}{ext}' for idx, ext in enumerate(ext_list)
    ]
