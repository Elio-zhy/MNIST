from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import MNIST_PATH, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST, MNIST_INFO


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_INFO['MEAN'],), (MNIST_INFO['STD'],))
])


def load_data(train=True, transforms=transforms) -> DataLoader:
    data_set = datasets.MNIST(MNIST_PATH, download=True, train=train, transform=transform)
    batch_size = BATCH_SIZE_TRAIN if train else BATCH_SIZE_TEST
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

    return data_loader
