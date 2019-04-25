import glob

import SimpleITK as sitk
import numpy as np
import torch
from dltk.io.augmentation import flip
from dltk.io.preprocessing import normalise_one_one
from dltk.io.preprocessing import whitening
from skimage.transform import resize
from torch.utils.data.dataset import Dataset


class IXI_Dataset(Dataset):
    def __init__(self, filename, prop):
        # Get the data file names
        self.datafiles = glob.glob(
                filename + '/*.nii.gz')
        size = int(len(self.datafiles) * prop)
        self.size = size
        if prop > 0.5:
            self.datafiles = self.datafiles[:size]
        else:
            self.datafiles = self.datafiles[-size:]
        self.total = len(self.datafiles) * 70

    def __getitem__(self, index):

        num_file = index // 70
        num_index = index % 70
        batch_file = self.datafiles[index // 70]

        t1 = sitk.GetArrayFromImage(sitk.ReadImage(batch_file))
        t1 = t1[30:-30, ::-1, :]

        t1 = t1[num_index]
        t1 = whitening(t1)
        t1 = normalise_one_one(t1)
        t1 = flip(t1)

        im = np.expand_dims(t1, axis=-1).astype(np.float32)
        im = resize(im, [128, 128], mode='constant', anti_aliasing=True)

        im = torch.tensor(im, dtype=torch.float).permute(2, 0, 1)
        return im

    def __len__(self):
        return self.total
