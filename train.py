import time
import torch
import torch.nn.functional as F
from torch import optim
from vit_pytorch import ViT
from vit_pytorch.dino import MLP
import matplotlib.pyplot as plt

from config import N_EPOCHS, PATCH_SIZE, DEPTH, DIM, HEADS, MLP_DIM, LR, MNIST_INFO, SAVE, MODEL_PATH, TEST_LOSS_FIG_PATH, TRAIN_LOSS_FIG_PATH
from load_data import load_data


def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
        
            loss_history.append(loss.item())


def evaluate(model, data_loader, loss_history):
    model.eval()
    
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')


def train(train_loader, test_loader, patch_size, dim, depth, heads, mlp_dim, lr):
    start_time = time.time()
    model = ViT(
        image_size=MNIST_INFO['IMAGE_SIZE'], 
        patch_size=patch_size, 
        num_classes=MNIST_INFO['NUM_CLASSES'], 
        channels=MNIST_INFO['CHANNELS'],
        dim=dim, 
        depth=depth, 
        heads=heads, 
        mlp_dim=mlp_dim
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_history, test_loss_history = [], []
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        train_epoch(model, optimizer, train_loader, train_loss_history)
        evaluate(model, test_loader, test_loss_history)

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

    if SAVE:
        torch.save(model.state_dict(), MODEL_PATH)

    save_fig(train_loss_history, test_loss_history, TRAIN_LOSS_FIG_PATH, TEST_LOSS_FIG_PATH)


def save_fig(train_loss_history, test_loss_history, train_path, test_path):
    x1 = list(range(1, len(train_loss_history)+1))
    x2 = list(range(1, len(test_loss_history)+1))
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(x1, train_loss_history, 'r', label='train_loss')
    # plt.show()
    plt.savefig(train_path)

    plt.clf()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(x2, test_loss_history, 'b', label='test_loss')
    # plt.show()
    plt.savefig(test_path)


if __name__ == '__main__':
    train_loader = load_data(train=True)
    test_loader = load_data(train=False)
    
    train(train_loader, test_loader, PATCH_SIZE, DIM, DEPTH, HEADS, MLP_DIM, LR)
