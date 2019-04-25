import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class SelfAttention(nn.Module):

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim

        self.conv_theta = spectral_norm(
                nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1))
        self.conv_phi = spectral_norm(
                nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1))
        self.conv_g = spectral_norm(
                nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1))
        self.conv_attn = spectral_norm(
                nn.Conv2d(in_channels=in_dim // 2, out_channels=in_dim, kernel_size=1))

        self.sigma = nn.Parameter(torch.zeros(1))

        self.pool = nn.MaxPool2d((2, 2), stride=2)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        batch_size, num_channels, w, h = x.size()
        location_num = h * w
        downsampled_num = location_num // 4
        theta = self.conv_theta(x)

        theta = theta.view(batch_size, num_channels // 8, location_num).permute(0, 2, 1)

        phi = self.conv_phi(x)

        phi = self.pool(phi)
        phi = phi.view(batch_size, num_channels // 8, downsampled_num)

        attn = torch.bmm(theta, phi)
        attn = self.softmax(attn)

        g = self.conv_g(x)

        g = self.pool(g)
        g = g.view(batch_size, num_channels // 2, downsampled_num).permute(0, 2, 1)

        attn_g = torch.bmm(attn, g).permute(0, 2, 1)
        attn_g = attn_g.view(batch_size, num_channels // 2, w, h)

        attn_g = self.conv_attn(attn_g)
        return x + self.sigma * attn_g
