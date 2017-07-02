from agent import Trial
import csv

# BEST = 'alpha'
BEST = 'bravo'
DIR = 'data/'
PARAMS = [
    {'alpha': 0.01,   'gamma': 0.9,   'epsilon_decay': 0.9},
    {'alpha': 0.001,  'gamma': 0.9,   'epsilon_decay': 0.9},
    {'alpha': 0.0001, 'gamma': 0.9,   'epsilon_decay': 0.9},
    {'alpha': 0.01,   'gamma': 0.99,  'epsilon_decay': 0.9},
    {'alpha': 0.001,  'gamma': 0.99,  'epsilon_decay': 0.9},
    {'alpha': 0.0001, 'gamma': 0.99,  'epsilon_decay': 0.9},
    {'alpha': 0.01,   'gamma': 0.999, 'epsilon_decay': 0.9},
    {'alpha': 0.001,  'gamma': 0.999, 'epsilon_decay': 0.9},
    {'alpha': 0.0001, 'gamma': 0.999, 'epsilon_decay': 0.9},
    {'alpha': 0.01,   'gamma': 0.9,   'epsilon_decay': 0.99},
    {'alpha': 0.001,  'gamma': 0.9,   'epsilon_decay': 0.99},
    {'alpha': 0.0001, 'gamma': 0.9,   'epsilon_decay': 0.99},
    {'alpha': 0.01,   'gamma': 0.99,  'epsilon_decay': 0.99},
    {'alpha': 0.001,  'gamma': 0.99,  'epsilon_decay': 0.99},
    {'alpha': 0.0001, 'gamma': 0.99,  'epsilon_decay': 0.99},
    {'alpha': 0.01,   'gamma': 0.999, 'epsilon_decay': 0.99},
    {'alpha': 0.001,  'gamma': 0.999, 'epsilon_decay': 0.99},
    {'alpha': 0.0001, 'gamma': 0.999, 'epsilon_decay': 0.99},
    {'alpha': 0.01,   'gamma': 0.9,   'epsilon_decay': 0.999},
    {'alpha': 0.001,  'gamma': 0.9,   'epsilon_decay': 0.999},
    {'alpha': 0.0001, 'gamma': 0.9,   'epsilon_decay': 0.999},
    {'alpha': 0.01,   'gamma': 0.99,  'epsilon_decay': 0.999},
    {'alpha': 0.001,  'gamma': 0.99,  'epsilon_decay': 0.999},
    {'alpha': 0.0001, 'gamma': 0.99,  'epsilon_decay': 0.999},
    {'alpha': 0.01,   'gamma': 0.999, 'epsilon_decay': 0.999},
    {'alpha': 0.001,  'gamma': 0.999, 'epsilon_decay': 0.999},
    {'alpha': 0.0001, 'gamma': 0.999, 'epsilon_decay': 0.999},
]


def get_name(params):
    return 'grid-a{}-g-{}-e{}'.format(params['alpha'], params['gamma'], params['epsilon_decay'])


def get_names():
    return [get_name(params) for params in PARAMS]


def train_best():
    name = 'bravo'
    trial = Trial(name, episodes=2000, stop=False, verbosity=2)
    max_episode, mean_reward, params, rewards = trial.run()


def evaluate_best():
    name = 'bravo'
    trial = Trial(name, episodes=1000, load=True, verbosity=1, params={
        'epsilon_max': 0
    })
    max_episode, mean_reward, params, rewards = trial.run()


def train_grid():
    with open(DIR + 'grid.csv', 'w+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for params in PARAMS:
            name = get_name(params)
            trial = Trial(name, episodes=2000, record=True, stop=True, verbosity=1, params=params)
            max_episode, mean_reward, params, _rewards = trial.run()
            writer.writerow([
                name,
                params['alpha'],
                params['gamma'],
                params['epsilon_decay'],
                max_episode,
                mean_reward
            ])


def evaluate_grid():
    with open(DIR + 'grid.loaded.csv', 'w+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for params in PARAMS:
            name = get_name(params)
            params['epsilon_max'] = 0
            trial = Trial(name, episodes=100, load=True, stop=False, verbosity=1, params=params)
            max_episode, mean_reward, params, _rewards = trial.run()
            writer.writerow([
                name,
                params['alpha'],
                params['gamma'],
                params['epsilon_decay'],
                max_episode,
                mean_reward
            ])


if __name__ == '__main__':
    train_best()
    evaluate_best()
    train_grid()
    evaluate_grid()
