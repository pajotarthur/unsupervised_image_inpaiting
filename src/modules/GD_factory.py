import torch
from torch import nn


# Parallelized G_D to minimize cross-gpu communication
# Without this, Generator outputs would get all-gathered and then rebroadcast.
class G_D(nn.Module):
    def __init__(self, G, D):
        super(G_D, self).__init__()
        self.G = G
        self.D = D

    def forward(self, z, x=None, label=None, train_G=False, return_G_z=False, corruption=None):
        # If training G, enable grad tape
        with torch.set_grad_enabled(train_G):
            # Get Generator output given noise
            G_z = self.G(z, label)

            #only if ambientgan styple
            if corruption is not None:
                G_z, _ = corruption(G_z)

        if train_G:
            pred_fake = self.D(G_z, label)
            return pred_fake
        else:
            pred_real = self.D(x, label)
            pred_fake = self.D(G_z, label)

            return pred_real, pred_fake
