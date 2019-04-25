from torch.utils.data import Dataset


class ProgressiveDataset(Dataset):
    def __init__(self, dataset, corruption, output_transform=lambda x: x):
        self.dataset = dataset
        self.corruption = corruption
        self.output_transform = output_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        batch = self.dataset[item]
        x, label = self.output_transform(batch)
        y, theta = self.corruption(x.unsqueeze(0), seed=item)
        if label is not None:
            return {
                    'x':     x,
                    'label': label,
                    'y':     y.squeeze(0),
                    'theta': theta.squeeze(0)
                    }
        else:
            return {
                    'x':     x,
                    'y':     y.squeeze(0),
                    'theta': theta.squeeze(0)
                    }
