import glob, os, zipfile, re
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import imageio as io


def kth_sequences_parser(filename, group='train'):
    f = open(filename)
    r = re.compile(r'''
        (?P<filename>\w+)\s+frames\s+
            (?P<s1>\d+)-(?P<e1>\d+),\ 
            (?P<s2>\d+)-(?P<e2>\d+),\ 
            (?P<s3>\d+)-(?P<e3>\d+),\ 
            (?P<s4>\d+)-(?P<e4>\d+)\s*
        ''', re.X)
    r_fn = re.compile(r'person(?P<num>\d+).*')
    infos = list()

    if group == 'train':
        psns = [11, 12, 13, 14, 15, 16, 17, 18]
    elif group == 'val':
        psns = [19, 20, 21, 23, 24, 25, 1, 4]
    elif group == 'test':
        psns = [22, 2, 3, 5, 6, 7, 8, 9, 10]
    for line in f:
        if not line.startswith('person'):
            continue

        res = re.match(r, line)
        if res is None:
            continue

        res_fn = re.match(r_fn, res.group('filename'))
        psn_no = int(res_fn.group('num'))

        if psn_no in psns:
            infos.append(
                    {
                            'filename': res.group('filename') + '_uncomp.avi',
                            'fgmts':    [
                                    (int(res.group('s1')) - 1, int(res.group('e1')) - 1),
                                    (int(res.group('s2')) - 1, int(res.group('e2')) - 1),
                                    (int(res.group('s3')) - 1, int(res.group('e3')) - 1),
                                    (int(res.group('s4')) - 1, int(res.group('e4')) - 1)],
                            }
                    )

    f.close()

    lens = sorted([idx_e - idx_s + 1 for info in infos for idx_s, idx_e in info['fgmts']])
    avg_len = int(sum(lens) / len(lens))

    return infos, avg_len


class KTH(Dataset):
    def __init__(self, root: str, group: str = 'train', output_size=(64, 64), **kwargs):
        self.output_width = output_size[0]
        self.output_height = output_size[1]
        self.root = root
        self.infos, avg_len = kth_sequences_parser(os.path.join(root, '00sequences.txt'), group=group)
        self.max_len = 40
        self.total = len(self.infos) * 4
        self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.Resize((self.output_height, self.output_width)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                ])

    def __getitem__(self, index):

        idx_video = index // 4
        idx_fgmt = index % 4
        info = self.infos[idx_video]
        filename = os.path.join(self.root, info['filename'])
        idx_start_frame, idx_end_frame = info['fgmts'][idx_fgmt]

        x_real_frames = []
        reader = io.get_reader(filename)
        for i, frame in enumerate(reader):
            if i >= idx_start_frame and i <= idx_end_frame:
                x_real = self.transforms(np.array(frame))
                x_real_frames.append(x_real)

        num_frames = len(x_real_frames)
        start = 0
        end = num_frames
        if num_frames > self.max_len:
            end = self.max_len
            # start = torch.randint(num_frames - self.max_len, (1,))
            # end = start + self.max_len

        return torch.stack(x_real_frames[start:end], dim=1), None

    def __len__(self):
        return self.total


if __name__ == '__main__':
    ds = KTH('/data/yiny/KTH')