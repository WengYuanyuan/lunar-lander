import matplotlib.pyplot as plt
import numpy as np

from run import get_names, BEST

DATA_DIR = 'data/'
PLOT_DIR = 'images/'


def chunk_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def training():
    """ Plots reward for each training episode while training the best agent """
    data = np.loadtxt(open(DATA_DIR + BEST + '.csv', 'rb'), delimiter=',')

    plt.figure(1)

    size = 5
    chunks = list(chunk_list(data[:, 1], size))
    averages = [sum(chunk) / len(chunk) for chunk in chunks]
    success = next(i[0] for i in data if i[2] > 200)

    plt.plot(range(0, len(data), size), averages)
    plt.axhline(y=200, color='g', linestyle='--')
    plt.axvline(x=success, color='r')

    plt.title('Learning Performance')
    plt.xlabel('Episode')
    plt.ylabel('Episode Total Reward (Averaged Per {})'.format(size))

    plt.savefig(PLOT_DIR + 'training.png')


def evaluate_100():
    data = np.loadtxt(open(DATA_DIR + BEST + '.loaded.csv', 'rb'), delimiter=',')
    data = data[:100]

    plt.figure(2)
    plt.title('100-trial Performance')
    plt.xlabel('Episode')
    plt.ylabel('Episode Total Reward')
    plt.plot(data[:, 0], data[:, 1])

    plt.savefig(PLOT_DIR + 'evaluate_100.png')


def histogram():
    """ Plots a histogram of 1000 trials using the best agent """
    data = np.loadtxt(open(DATA_DIR + 'alpha.loaded.csv', 'rb'), delimiter=',')

    plt.figure(3)
    plt.title('Rewards Histogram')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.hist(data[:, 1], 20)
    plt.grid(True)

    plt.savefig(PLOT_DIR + 'histogram.png')


def grid():
    """ Plots grid search on hyperparameters """
    pass

    data = np.loadtxt(open(DATA_DIR + 'grid.csv', 'rb'), delimiter=',')

    # plt.figure(4)
    # plt.title('Hyperparameter Comparison')
    # plt.xlabel('Episode')
    # plt.ylabel('Episode Total Reward')
    #
    # plt.savefig(PLOT_DIR + 'grid0.png')


if __name__ == '__main__':
    training()
    evaluate_100()
    histogram()
    grid()
    plt.show()
