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


def process_dataset(input_dir, output_dir, transform=None, target_ext=None, extensions=IMG_EXTS):
    '''
    '''

    filelist = get_filelist(input_dir, extensions)

    ext_list = [target_ext] * len(filelist) if target_ext is not None else [
        os.path.splitext(filename)[1]
        for filename in filelist
    ]

    out_filelist = [
        os.path.join(output_dir, f'{(idx):08d}{ext}')
        for idx, ext in enumerate(ext_list)
    ]

    if target_ext is None and transform is None:
        # simply copy images with new names
        for src, dst in zip(filelist, out_filelist):
            shutil.copyfile(src, dst)
    else:
        # load images, tranform them and save with new names
        save_images(
            load_images(filelist, transform),
            out_filelist
        )


def merge_datasets(input_dirs: list, output_dir: str, transform=None, target_ext=None, extensions=IMG_EXTS):
    idx_shift = 0
    for input_dir in input_dirs:
        # TODO
        idx_shift += len(get_filelist(input_dir, extensions))
