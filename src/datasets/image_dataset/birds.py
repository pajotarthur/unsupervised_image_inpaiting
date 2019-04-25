import glob

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Birds(Dataset):
    def __init__(self, root: str, train: bool = True, im_size=None):
        # Get the data file names

        if train:
            self.datafiles = glob.glob(root + "/images/*/*.jpg")
        else:
            self.datafiles = glob.glob(root + "/images/*/*.jpg")

        self.total = len(self.datafiles)
        self.output_height = im_size or 64
        self.output_width = im_size or 64
        self.transforms = transforms.Compose([
                transforms.Resize([self.output_height, self.output_width], 2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

    def __getitem__(self, index):

        batch_file = self.datafiles[index]
        im = pil_loader(batch_file)
        im = self.transforms(im)
        return im

    def __len__(self):
        return self.total
