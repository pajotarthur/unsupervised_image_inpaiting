import glob

import imageio
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.transform import resize, rotate
from torch.utils.data.dataset import Dataset


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def imread(path):
    return imageio.imread(path).astype(np.float)


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return resize(x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w], mode='constant', anti_aliasing=True)


def transform(image, input_height, input_width, resize_height=64, resize_width=64, angle=90, is_crop=True):
    if is_crop:
        cropped_image = center_crop(image, input_height, input_width,
                                    resize_height, resize_width)
    else:
        cropped_image = resize(
                image, [resize_height, resize_width], mode='constant', anti_aliasing=True)
    cropped_image = rotate(cropped_image, angle)
    return np.array(cropped_image) / 127.5 - 1.


def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, angle=90, is_crop=True):
    image = imread(image_path)
    return transform(image, input_height, input_width, resize_height, resize_width, angle, is_crop)


class FFHQ(Dataset):
    def __init__(self, root: str, train: bool = 'train', im_size=None):
        # Get the data file names

        if train == 'train':
            root = glob.glob(root + '/[0-7]*.png')
        elif train == 'test':
            root = glob.glob(root + '/[9]*.png')
        elif train == 'train+val':
            root = glob.glob(root + '/[0-8]*.png')
        elif train == 'val':
            root = glob.glob(root + '/[8]*.png')
        else:
            raise FileNotFoundError('not a good train argument')

        self.total = len(root)

        self.attr2idx = {}
        self.idx2attr = {}

        self.transforms = transforms.Compose([
                # transforms.Resize([im_size, im_size], 2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

        self.root = root

    def __getitem__(self, index):
        batch_file = self.root[index]

        # try:
        # x_real = pil_loader(batch_file)
        im = get_image(batch_file,
                       input_height=128,
                       input_width=128,
                       resize_height=128,
                       resize_width=128,
                       is_crop=False, angle=0)
        im = torch.tensor(im, dtype=torch.float).permute(2, 0, 1)

        # except:
        #     return None, None

        # im = self.transforms(x_real)

        return im, 0

    def __len__(self):
        return self.total
