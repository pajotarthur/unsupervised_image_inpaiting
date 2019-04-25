import torch
import torch.utils.data.dataset as dataset
import xarray as xr
from sklearn.preprocessing import MinMaxScaler


class AssimDataset(dataset.Dataset):
    def __init__(self, fic, name, time=None):

        if name == 'cloud':
            variables = ['hphy', 'hphy_b', 'hphy_cl', 'uphy', 'vphy', 'uphy_b', 'vphy_b']
        else:
            variables = ['hphy', 'hphy_b', 'hphy_o', 'uphy', 'vphy', 'uphy_b', 'vphy_b']

        super().__init__()
        ds = xr.open_dataset(fic)
        scaler = MinMaxScaler((0, 1))
        if time is None:
            time_slice = slice(None, 172788)
        else:
            time_slice = time

        ds = ds[variables]

        ds = ds.sel(time=time_slice)

        for i in ds.data_vars:
            ds[i] = (ds[i] - ds[i].min()) / (ds[i].max() - ds[i].min())
            ds[i] = ds[i] * 2 - 1

        self.ds = torch.Tensor(ds.to_array().data)

    def __len__(self):
        return self.ds.shape[1]

    def __getitem__(self, item):
        return torch.split(self.ds[:, item], 1)
