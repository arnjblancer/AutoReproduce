import argparse
import torch
import scipy
import numpy as np
import os

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

def get_dataloader(args):
    TRAIN_PATH = os.path.join(args.data_path, 'piececonst_r421_N1024_smooth1.mat')
    TEST_PATH = os.path.join(args.data_path, 'piececonst_r421_N1024_smooth2.mat')

    r1 = args.h_down
    r2 = args.w_down
    s1 = int(((args.h - 1) / r1) + 1)
    s2 = int(((args.w - 1) / r2) + 1)

    reader = MatReader(TRAIN_PATH)
    x_train = reader.read_field('coeff')[:args.ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = reader.read_field('sol')[:args.ntrain, ::r1, ::r2][:, :s1, :s2]

    reader.load_file(TEST_PATH)
    x_test = reader.read_field('coeff')[:args.ntest, ::r1, ::r2][:, :s1, :s2]
    y_test = reader.read_field('sol')[:args.ntest, ::r1, ::r2][:, :s1, :s2]

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)
    y_test = y_normalizer.encode(y_test)
    y_normalizer.cuda()

    x_train = x_train.reshape(args.ntrain, s1, s2, 1)
    x_test = x_test.reshape(args.ntest, s1, s2, 1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader, x_normalizer, y_normalizer

parser = argparse.ArgumentParser('Training Latent Spectral Models')
# dataset
args = parser.parse_args()
# dataset
args.data_path = 'zxl/lsm/source'
args.ntotal = 1200
args.ntrain = 1000
args.ntest = 200
args.h = 421
args.w = 421
args.h_down = 5
args.w_down = 5
# optimization
args.batch_size = 40