import os


MNIST_PATH = os.path.join(os.getcwd(), 'MNIST')
MNIST_INFO = {
    'MEAN': 0.1307,
    'STD': 0.3081,
    'IMAGE_SIZE': 28,
    'NUM_CLASSES': 10,
    'CHANNELS': 1
}

BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 1000

N_EPOCHS = 25

# training related parameters
PATCH_SIZE = 7
DIM = 64
DEPTH = 6
HEADS = 8
MLP_DIM = 128
LR = 0.003

# save model
SAVE = True
MODEL_PATH = os.path.join(os.getcwd(), 'parameters.pkl')

# result image save path
TRAIN_LOSS_FIG_PATH = os.path.join(os.getcwd(), 'train-loss.png')
TEST_LOSS_FIG_PATH = os.path.join(os.getcwd(), 'test-loss.png')
