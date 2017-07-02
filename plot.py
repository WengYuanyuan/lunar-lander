import matplotlib.pyplot as plt
import numpy as np
import sys

from itertools import groupby

from run import get_name, BEST, PARAMS

DATA_DIR = 'data/'
PLOT_DIR = 'images/'


def chunk_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def read(name):
    return np.loadtxt(open(DATA_DIR + name + '.csv', 'rb'), delimiter=',')


def training(override=None):
    """ Plots reward for each training episode while training the best agent """
    name = BEST if override is None else override
    data = read(name)

    plt.figure(1)

    size = 5
    chunks = list(chunk_list(data[:, 1], size))
    averages = [sum(chunk) / len(chunk) for chunk in chunks]

    plt.plot(range(0, len(data), size), averages)
    plt.axhline(y=200, color='g', linestyle='--')

    try:
        success = [i[0] for i in data if i[2] >= 200][0] - 200
        plt.axvline(x=success, color='r')
    except IndexError:
        print('[WARN] Learner did not succeed')

    plt.title('Learning Performance')
    plt.xlabel('Episode')
    plt.ylabel('Episode Total Reward (Averaged Per {})'.format(size))

    plt.savefig(PLOT_DIR + 'training.{}.png'.format(name))


def evaluate_100(override=None):
    name = BEST if override is None else override
    data = read(name)[:100]

    plt.figure(2)
    plt.title('100-trial Performance')
    plt.xlabel('Episode')
    plt.ylabel('Episode Total Reward')
    plt.axhline(y=200, color='g', linestyle='--')
    plt.plot(data[:, 0], data[:, 1])

    plt.savefig(PLOT_DIR + 'evaluate_100.{}.png'.format(name))


def histogram(override=None):
    """ Plots a histogram of 1000 trials using the best agent """
    name = BEST if override is None else override
    data = read(name)

    plt.figure(3)
    plt.title('Rewards Histogram')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.hist(data[:, 1], 50, alpha=0.5, edgecolor='black', linewidth=1)
    plt.axvline(x=200, color='r')
    plt.grid(True)

    plt.savefig(PLOT_DIR + 'histogram.{}.png'.format(name))


def grid():
    """ Plots grid search on hyperparameters """
    datasets = [{
        'name': get_name(params),
        'alpha': params['alpha'],
        'gamma': params['gamma']
    } for params in PARAMS]

    plt.figure(4)
    selected = list(filter(lambda x: x['alpha'] == 0.0001, datasets))
    gammas = []
    for dataset in selected:
        try:
            name  = dataset['name']
            data  = read(name)
            gammas.append(dataset['gamma'])
            plt.plot(data[100:, 0] - 100, data[100:, 2])
        except StandardError:
            print('{} failed!'.format(name))

    plt.axhline(y=200, color='g', linestyle='--')
    plt.title('Learning Performance')
    plt.xlabel('Episodes')
    plt.ylabel('Running Average Reward')
    plt.legend([r'$\gamma$ = {}'.format(gamma) for gamma in gammas])
    plt.savefig(PLOT_DIR + 'grid.learning.png')

    plt.figure(5)
    alphas = []
    datasets.sort(key=lambda x: x['alpha'])
    for alpha, group in groupby(datasets, lambda x: x['alpha']):
        alphas.append(alpha)
        gammas  = []
        rewards = []
        for dataset in group:
            try:
                name  = dataset['name']
                data  = read('{}.loaded'.format(name))
                gammas.append(dataset['gamma'])
                rewards.append(np.mean(data[:, 1]))
            except StandardError:
                print('{} failed!'.format(name))
        plt.plot(gammas, rewards)

    plt.title(r'Discount ($\gamma$) Across Different Learning Rates ($\alpha$)')
    plt.xlabel('$\gamma$')
    plt.ylabel('Average Reward')
    plt.legend([r'$\alpha$ = {}'.format(alpha) for alpha in alphas])
    plt.savefig(PLOT_DIR + 'grid.gamma.png')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        training(sys.argv[1])
        plt.show()
    else:
        training()
        evaluate_100()
        histogram()
        grid()
        plt.show()
