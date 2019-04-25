import glob

import imageio
import torch
from skimage.transform import resize
from torch.utils.data.dataset import Dataset


def imread(path):
    img = imageio.imread(path)
    return resize(img, [128, 128], mode='constant', anti_aliasing=True)


class Galaxy(Dataset):
    def __init__(self, root, prop):
        # Get the data file names

        self.datafiles = glob.glob(
                root + '/*.jpg')

        size = int(len(self.datafiles) * prop)
        if prop > 0.5:
            self.datafiles = self.datafiles[:size]
        else:
            self.datafiles = self.datafiles[-size:]

        self.total = len(self.datafiles)

    def __getitem__(self, index):

        batch_file = self.datafiles[index]
        im = imread(batch_file)
        im = torch.tensor(im, dtype=torch.float).permute(2, 0, 1)
        return im

    def __len__(self):
        return self.total
