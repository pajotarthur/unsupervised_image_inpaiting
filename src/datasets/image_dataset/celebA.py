import glob

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset


def get_image(image_path):
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class CelebA(Dataset):
    def __init__(self, root: str, train: bool = 'train', im_size=128, center_crop=128):
        """

        :type center_crop: int
        """
        # Get the data file namesp

        self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']

        self.attr_path = root + '/../list_attr_celeba.txt'
        if train == 'train':
            root = glob.glob(root + '/?[0-7]*.jpg')
        elif train == 'test':
            root = glob.glob(root + '/?[9]*.jpg')
        elif train == 'train+val':
            root = glob.glob(root + '/?[0-8]*.jpg')  # + glob.glob("/local/pajot/data/thumbnails128x128/*.png")
        elif train == 'val':
            root = glob.glob(root + '/?[8]*.jpg')
        else:
            raise FileNotFoundError('not a good train argument')

        self.total = len(root)

        self.attr2idx = {}
        self.idx2attr = {}

        # self.file2label = self.preprocess()

        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]

        train_transform = [transforms.CenterCrop(center_crop),
                           transforms.Resize(im_size)]
        self.train_transform = transforms.Compose(train_transform + [
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)])

        self.root = root

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        dico = {}

        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')
            dico[filename] = label

        return dico

    def __getitem__(self, index):
        batch_file = self.root[index]

        im = get_image(batch_file)
        im = self.train_transform(im)
        # print(im.shape)
        return {'x': im, 'label': 0}

    def __len__(self):
        return self.total
