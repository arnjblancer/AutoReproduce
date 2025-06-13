from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from yacs.config import CfgNode as CN

def get_cifar100_train_transform():
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    return train_transform

def get_cifar100_test_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

class CIFAR100Instance(datasets.CIFAR100):
    """CIFAR100Instance Dataset."""

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index
    
def get_cifar100_dataloaders(data_folder, batch_size, val_batch_size, num_workers):
    train_transform = get_cifar100_train_transform()
    test_transform = get_cifar100_test_transform()
    train_set = CIFAR100Instance(
        root=data_folder, download=True, train=True, transform=train_transform
    )
    num_data = len(train_set)
    test_set = datasets.CIFAR100(
        root=data_folder, download=True, train=False, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=1,
    )
    return train_loader, test_loader, num_data

def get_dataset(cfg):
    train_loader, val_loader, num_data = get_cifar100_dataloaders(
        data_folder=cfg.DATASET.PATH,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
        num_workers=cfg.DATASET.NUM_WORKERS,
    )
    num_classes = 100
    return train_loader, val_loader, num_data, num_classes


if __name__ == "__main__":
    cfg = CN()

    # Model
    cfg.TEACHER = CN()
    cfg.TEACHER.PATH = "zxl/dkd/source"

    # Dataset
    cfg.DATASET = CN()
    cfg.DATASET.TYPE = "cifar100"
    cfg.DATASET.PATH = 'zxl/dkd/source'
    cfg.DATASET.NUM_WORKERS = 10
    cfg.DATASET.TEST = CN()
    cfg.DATASET.TEST.BATCH_SIZE = 64

        # Solver
    cfg.SOLVER = CN()
    cfg.SOLVER.BATCH_SIZE = 64