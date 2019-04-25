import os
import string
import sys

import six
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import lmdb


class LSUN(data.Dataset):
    def __init__(self, root, train=True, im_size=64):

        if train:
            root = os.path.join(root, 'bedroom_train_lmdb')
        else:
            root = os.path.join(root, 'bedroom_val_lmdb')
        self.root = os.path.expanduser(root)

        self.transform = transforms.Compose([
                transforms.Resize(im_size),
                transforms.CenterCrop(im_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        cache_file = os.path.join(root, '_cache_' + ''.join(c for c in root if c in string.ascii_letters))
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return {'x': img, 'label': 0}

    def __len__(self):
        return self.length
