from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import MNIST_PATH, MNIST_MEAN, MNIST_STD, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MNIST_MEAN, MNIST_STD)
])


def load_data(train=True, transforms=transforms) -> DataLoader:
    data_set = datasets.MNIST(MNIST_PATH, download=True, train=train, transforms=transforms)
    batch_size = BATCH_SIZE_TRAIN if train else BATCH_SIZE_TEST
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

    return data_loader
