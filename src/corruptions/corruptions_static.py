import random

import numpy as np
import torch


class Corruption(object):
    def sample_theta(self, im_shape, seed=None):
        raise NotImplementedError

    def __call__(self, x, device, theta=None, seed=None):
        raise NotImplementedError


class KeepPatch(Corruption):
    def __init__(self, size_percent=0.5, im_size=64):
        self.size_percent = size_percent

        if isinstance(size_percent, int):
            self.size_percent = (size_percent, size_percent)

        self.im_size = im_size
        self.size_patch = (int(self.size_percent[0] * im_size),
                           int(self.size_percent[1] * im_size))

    def sample_theta(self, im_shape, seed=None):
        s = np.random.RandomState(seed)

        mask = np.zeros(im_shape)
        for i in range(im_shape[0]):
            x_patch = s.randint(0, self.im_size - self.size_patch[0] + 1)
            y_patch = s.randint(0, self.im_size - self.size_patch[1] + 1)
            mask[i, :, x_patch:x_patch + self.size_patch[0], y_patch:y_patch + self.size_patch[1]] = 1

        return mask

    def __call__(self, x, t, device, theta=None, seed=None):
        assert (self.im_size is not None)
        fx = x.clone()

        if theta is None:
            mask = self.sample_theta(im_shape=x.shape, seed=seed)
            mask = torch.tensor(mask, dtype=torch.uint8, device=device, requires_grad=False)
        else:
            mask = theta

        fx[mask] = 0

        return fx, mask


class SmallPatch(Corruption):
    def __init__(self, size_percent=0.5, num_patch=5):
        self.size_percent = size_percent
        self.num_patch = num_patch

        if isinstance(size_percent, int):
            self.size_percent = (size_percent, size_percent)

    def sample_theta(self, im_shape, seed=None):
        s = np.random.RandomState(seed)

        im_size = im_shape[2]
        size_patch = (int(self.size_percent[0] * im_size),
                      int(self.size_percent[1] * im_size))

        mask = np.zeros(im_shape)
        for i in range(im_shape[0]):
            for i in range(self.num_patch):
                x_patch = s.randint(0, im_size - size_patch[0] + 1)
                y_patch = s.randint(0, im_size - size_patch[1] + 1)
                mask[i, :, x_patch:x_patch + size_patch[0], y_patch:y_patch + size_patch[1]] = 1

        return mask

    def __call__(self, x, t, device, theta=None, seed=None):
        assert (self.im_size is not None)
        fx = x.clone()

        if theta is None:
            mask = self.sample_theta(im_shape=x.shape, seed=seed)
            mask = torch.tensor(mask, dtype=torch.uint8, device=device, requires_grad=False)
        else:
            mask = theta

        fx[mask] = 0

        return fx, mask, None




class RemovePixelDark(Corruption):
    def __init__(self, percent=0.9, percent_top=0.9, im_size=64):
        self.percent = percent
        self.percent_top = percent

    def sample_theta(self, im_shape, seed=None):
        s = np.random.RandomState(seed)

        mask = np.zeros((im_shape[0], im_shape[1], im_shape[2], im_shape[3]))
        for i in range(im_shape[0]):
            percent = random.uniform(self.percent, self.percent_top)
            # print(percent,self.percent, self.percent_top)
            ones = np.zeros([im_shape[2] * im_shape[3]])
            ones[:int(percent * im_shape[2] * im_shape[3])] = 1
            s.shuffle(ones)
            ones = ones.reshape(im_shape[2], im_shape[3])

            mask[i, :, ] = ones

        return mask

    def __call__(self, x, theta=None, seed=None):

        if theta is None:
            mask = self.sample_theta(im_shape=x.shape, seed=seed)
            mask = torch.tensor(mask, dtype=torch.uint8, device=x.device, requires_grad=False)
        else:
            mask = theta
        # print('corruption', x.shape, mask.shape)
        mask = mask.float()
        mask = mask.to(x.device)

        return x*mask, mask


class ConvNoise(Corruption):
    def __init__(self, conv_size, noise_variance, im_size=64):
        super().__init__()
        self.conv_size = conv_size
        self.noise_variance = noise_variance

    def sample_theta(self, im_shape, seed=None):
        s = np.random.RandomState(seed)
        x = np.zeros(im_shape, dtype='f')
        x[:] = s.randn(*x.shape)

        return x * self.noise_variance

    def __call__(self, x, t, device, theta=None, seed=None):
        fx = x.clone()

        if theta is None:
            noise = self.sample_theta(im_shape=x.shape, seed=seed)
            noise = torch.tensor(noise, device=device, dtype=torch.float32, requires_grad=False)
        else:
            noise = theta
        eps = torch.ones(1, 1, self.conv_size, self.conv_size, device=device) / (self.conv_size * self.conv_size)
        p = int((1 - 1 + 1 * self.conv_size) / 2)
        for i in range(x.shape[1]):
            fx[:, i:i + 1] = torch.nn.functional.conv2d(x[:, i:i + 1], eps, padding=p)

        return fx + noise, noise, None


class MovingVerticalBar(Corruption):
    def __init__(self, im_size=(64, 64), width_ratio=[0.1, 0.5], max_speed=0.05):
        super().__init__()
        assert len(im_size) == 2
        assert len(width_ratio) == 2

        self.im_height, self.im_width = im_size
        self.max_width = im_size[1] * width_ratio[0]
        self.min_width = im_size[1] * width_ratio[1]
        self.max_speed = im_size[1] * max_speed

    def sample_theta(self, batch_size, seed=None):
        s = np.random.RandomState(seed)
        width = s.rand(batch_size, 1) * (self.max_width - self.min_width) + self.min_width
        speed = s.rand(batch_size, 1) * self.max_speed
        start_pos = s.rand(batch_size, 1) * (self.im_width + width)
        for i in range(batch_size):
            if start_pos[i][0] - width[i][0] / 2 > self.im_width / 2:
                speed[i][0] = -speed[i][0]

        return {
                'width':     width,
                'speed':     speed,
                'start_pos': start_pos,
                }

    def __call__(self, x, t, device, theta=None, real_seq_len=None, seed=None, mask=None):

        fx = x.clone()
        if mask is None:
            mask = torch.zeros_like(x, dtype=torch.uint8)
            batch_size, _, _, width = x.shape

            if t < 0:
                return fx, mask, theta

            if theta is None:
                if t == 0:
                    theta = self.sample_theta(batch_size=batch_size, seed=seed)
                elif t > 0:
                    raise ValueError('theta is required for all steps other than the initial one.')

            right_edge_pos = (theta['start_pos'] + t * theta['speed']) % (self.im_width + theta['width'])
            left_edge_pos = right_edge_pos - theta['width']

            right_edge_pos[right_edge_pos > width] = width
            right_edge_pos[right_edge_pos < 0] = 0

            left_edge_pos[left_edge_pos > width] = width
            left_edge_pos[left_edge_pos < 0] = 0

            for i in range(batch_size):
                if real_seq_len is not None:
                    if t >= real_seq_len[i]:
                        continue
                mask[i, :, :, int(left_edge_pos[i][0]):int(right_edge_pos[i][0])] = 1

        fx[mask] = 1

        return fx, mask, theta


class BouncingSquare(Corruption):
    def __init__(self, im_size=(64, 64), width_ratio=[0.1, 0.5], max_speed=0.05):
        super().__init__()
        assert len(im_size) == 2
        assert len(width_ratio) == 2

        self.im_height, self.im_width = im_size
        self.max_width = im_size[1] * width_ratio[0]
        self.min_width = im_size[1] * width_ratio[1]
        self.max_speed_h = im_size[1] * max_speed
        self.max_speed_w = im_size[1] * max_speed

    def sample_theta(self, batch_size, seed=None):
        s = np.random.RandomState(seed)
        width = s.rand(batch_size, 1) * (self.max_width - self.min_width) + self.min_width
        speed = s.rand(batch_size, 1) * self.max_speed
        start_pos = s.rand(batch_size, 1) * (self.im_width - width)
        for i in range(batch_size):
            if start_pos[i][0] - width[i][0] / 2 > self.im_width / 2:
                speed[i][0] = -speed[i][0]

        return {
                'width':     width,
                'speed':     speed,
                'start_pos': start_pos,
                }

    def __call__(self, x, t, device, theta=None, real_seq_len=None, seed=None, mask=None):

        fx = x.clone()
        if mask is None:
            mask = torch.zeros_like(x, dtype=torch.uint8)
            batch_size, _, _, width = x.shape

            if t < 0:
                return fx, mask, theta

            if theta is None:
                if t == 0:
                    theta = self.sample_theta(batch_size=batch_size, seed=seed)
                elif t > 0:
                    raise ValueError('theta is required for all steps other than the initial one.')

            right_edge_pos = (theta['start_pos'] + t * theta['speed']) % (self.im_width + theta['width'])
            left_edge_pos = right_edge_pos - theta['width']

            right_edge_pos[right_edge_pos > width] = width
            right_edge_pos[right_edge_pos < 0] = 0

            left_edge_pos[left_edge_pos > width] = width
            left_edge_pos[left_edge_pos < 0] = 0

            for i in range(batch_size):
                if real_seq_len is not None:
                    if t >= real_seq_len[i]:
                        continue
                mask[i, :, :, int(left_edge_pos[i][0]):int(right_edge_pos[i][0])] = 1

        fx[mask] = 1

        return fx, mask, theta


class StochasticMosaic(Corruption):
    try:
        import face_recognition as R
    except Exception:
        pass

    def __init__(self, im_size=(64, 64)):
        super().__init__()

    def sample_theta(self, batch_size, seed=None):
        pass

    def __call__(self, x, t, device, theta=None, real_seq_len=None, seed=None):
        pass