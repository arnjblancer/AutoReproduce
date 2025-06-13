import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import math
from timeit import default_timer
import scipy

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = True
        self._load_file()

    def _load_file(self):

        self.data = scipy.io.loadmat(self.file_path)

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)

        return torch.sum(all_norms)

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        return torch.sum(diff_norms / y_norms)


    def __call__(self, x, y):
        return self.rel(x, y)

# Multiscale modules 2D
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Patchify and Neural Spectral Block
class NeuralSpectralBlock2d(nn.Module):
    def __init__(self, width, num_basis, patch_size=[3, 3], num_token=4):
        super(NeuralSpectralBlock2d, self).__init__()
        self.patch_size = patch_size
        self.width = width
        self.num_basis = num_basis

        # basis
        self.modes_list = (1.0 / float(num_basis)) * torch.tensor([i for i in range(num_basis)],
                                                                  dtype=torch.float).cuda()
        self.weights = nn.Parameter(
            (1 / (width)) * torch.rand(width, self.num_basis * 2, dtype=torch.float))
        # latent
        self.head = 8
        self.num_token = num_token
        self.latent = nn.Parameter(
            (1 / (width)) * torch.rand(self.head, self.num_token, width // self.head, dtype=torch.float))
        self.encoder_attn = nn.Conv2d(self.width, self.width * 2, kernel_size=1, stride=1)
        self.decoder_attn = nn.Conv2d(self.width, self.width, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=-1)

    def self_attn(self, q, k, v):
        # q,k,v: B H L C/H
        attn = self.softmax(torch.einsum("bhlc,bhsc->bhls", q, k))
        return torch.einsum("bhls,bhsc->bhlc", attn, v)

    def latent_encoder_attn(self, x):
        # x: B C H W
        B, C, H, W = x.shape
        L = H * W
        latent_token = self.latent[None, :, :, :].repeat(B, 1, 1, 1)
        x_tmp = self.encoder_attn(x).view(B, C * 2, -1).permute(0, 2, 1).contiguous() \
            .view(B, L, self.head, C // self.head, 2).permute(4, 0, 2, 1, 3).contiguous()
        latent_token = self.self_attn(latent_token, x_tmp[0], x_tmp[1]) + latent_token
        latent_token = latent_token.permute(0, 1, 3, 2).contiguous().view(B, C, self.num_token)
        return latent_token

    def latent_decoder_attn(self, x, latent_token):
        # x: B C L
        x_init = x
        B, C, H, W = x.shape
        L = H * W
        latent_token = latent_token.view(B, self.head, C // self.head, self.num_token).permute(0, 1, 3, 2).contiguous()
        x_tmp = self.decoder_attn(x).view(B, C, -1).permute(0, 2, 1).contiguous() \
            .view(B, L, self.head, C // self.head).permute(0, 2, 1, 3).contiguous()
        x = self.self_attn(x_tmp, latent_token, latent_token)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C, H, W) + x_init  # B H L C/H
        return x

    def get_basis(self, x):
        # x: B C N
        x_sin = torch.sin(self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi)
        x_cos = torch.cos(self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi)
        return torch.cat([x_sin, x_cos], dim=-1)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bilm,im->bil", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        # patchify
        x = x.view(x.shape[0], x.shape[1],
                   x.shape[2] // self.patch_size[0], self.patch_size[0], x.shape[3] // self.patch_size[1],
                   self.patch_size[1]).contiguous() \
            .permute(0, 2, 4, 1, 3, 5).contiguous() \
            .view(x.shape[0] * (x.shape[2] // self.patch_size[0]) * (x.shape[3] // self.patch_size[1]), x.shape[1],
                  self.patch_size[0],
                  self.patch_size[1])
        # Neural Spectral
        # (1) encoder
        latent_token = self.latent_encoder_attn(x)
        # (2) transition
        latent_token_modes = self.get_basis(latent_token)
        latent_token = self.compl_mul2d(latent_token_modes, self.weights) + latent_token
        # (3) decoder
        x = self.latent_decoder_attn(x, latent_token)
        # de-patchify
        x = x.view(B, (H // self.patch_size[0]), (W // self.patch_size[1]), C, self.patch_size[0], self.patch_size[1]).permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W).contiguous()
        return x

class LSM_2D(nn.Module):
    def __init__(self, args):
        super(LSM_2D, self).__init__()
        in_channels = args.in_dim
        out_channels = args.out_dim
        width = args.d_model
        num_token = args.num_token
        num_basis = args.num_basis
        patch_size = [int(x) for x in args.patch_size.split(',')]
        padding = [int(x) for x in args.padding.split(',')]
        # multiscale modules
        self.inc = DoubleConv(width, width)
        self.down1 = Down(width, width * 2)
        self.down2 = Down(width * 2, width * 4)
        self.down3 = Down(width * 4, width * 8)
        factor = 2
        self.down4 = Down(width * 8, width * 16 // factor)
        self.up1 = Up(width * 16, width * 8 // factor)
        self.up2 = Up(width * 8, width * 4 // factor)
        self.up3 = Up(width * 4, width * 2 // factor)
        self.up4 = Up(width * 2, width)
        self.outc = OutConv(width, width)
        # Patchified Neural Spectral Blocks
        self.process1 = NeuralSpectralBlock2d(width, num_basis, patch_size, num_token)
        self.process2 = NeuralSpectralBlock2d(width * 2, num_basis, patch_size, num_token)
        self.process3 = NeuralSpectralBlock2d(width * 4, num_basis, patch_size, num_token)
        self.process4 = NeuralSpectralBlock2d(width * 8, num_basis, patch_size, num_token)
        self.process5 = NeuralSpectralBlock2d(width * 16 // factor, num_basis, patch_size, num_token)
        # projectors
        self.padding = padding
        self.fc0 = nn.Linear(in_channels + 2, width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        if not all(item == 0 for item in self.padding):
            x = F.pad(x, [0, self.padding[0], 0, self.padding[1]])

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(self.process5(x5), self.process4(x4))
        x = self.up2(x, self.process3(x3))
        x = self.up3(x, self.process2(x2))
        x = self.up4(x, self.process1(x1))
        x = self.outc(x)

        if not all(item == 0 for item in self.padding):
            x = x[..., :-self.padding[1], :-self.padding[0]]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

def main(args):
    TRAIN_PATH = os.path.join(args.data_path, 'piececonst_r421_N1024_smooth1.mat')
    TEST_PATH = os.path.join(args.data_path, 'piececonst_r421_N1024_smooth2.mat')

    ntrain = args.ntrain
    ntest = args.ntest
    r1 = args.h_down
    r2 = args.w_down
    s1 = int(((args.h - 1) / r1) + 1)
    s2 = int(((args.w - 1) / r2) + 1)

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    step_size = args.step_size
    gamma = args.gamma

    save_path = args.save_path
    save_name = args.save_name
    model = LSM_2D(args).cuda()

    # load data and data normalization
    
    reader = MatReader(TRAIN_PATH)
    x_train = reader.read_field('coeff')[:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = reader.read_field('sol')[:ntrain, ::r1, ::r2][:, :s1, :s2]

    reader.load_file(TEST_PATH)
    x_test = reader.read_field('coeff')[:ntest, ::r1, ::r2][:, :s1, :s2]
    y_test = reader.read_field('sol')[:ntest, ::r1, ::r2][:, :s1, :s2]

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)
    y_normalizer.cuda()

    x_train = x_train.reshape(ntrain, s1, s2, 1)
    x_test = x_test.reshape(ntest, s1, s2, 1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    # training and evaluation
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = LpLoss()

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x).reshape(batch_size, s1, s2)
            out = y_normalizer.decode(out)

            loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()

        scheduler.step()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

                out = model(x).reshape(batch_size, s1, s2)
                out = y_normalizer.decode(out)

                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        print(ep, t2 - t1, train_l2, test_l2)
        if ep % step_size == 0:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print('save model')
            torch.save(model.state_dict(), os.path.join(save_path, save_name))

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser('Training Latent Spectral Models')
    # dataset
    args = parser.parse_args()
    # dataset
    args.data_path = 'source'
    args.ntotal = 1200
    args.ntrain = 1000
    args.ntest = 200
    args.in_dim = 1
    args.out_dim = 1
    args.h = 421
    args.w = 421
    args.h_down = 5
    args.w_down = 5
    # optimization
    args.batch_size = 20
    args.learning_rate = 0.001
    args.epochs = 500
    args.step_size = 100
    args.gamma = 0.5

    # Model parameters
    args.d_model = 64
    args.num_basis = 12
    args.num_token = 4
    args.patch_size = '6,6'
    args.padding = '11,11'

    # save
    args.save_path = './checkpoints/'
    args.save_name = 'lsm.pt'

    main(args)