from ann import ANN
from collections import deque
from keras.optimizers import Adam
import csv
import gym
import numpy as np
import random

DATA_DIR = 'data/'


class OpenStruct:
    pass


class Agent:
    """ A reinforcement learning agent implementing Deep Q Network learning """

    def __init__(
        self, name, env,

        # Configuration
        verbosity=2,
        load=False,

        # Q-learning hyperparameters
        epsilon_decay=0.998,
        epsilon_max=1.0,
        epsilon_min=0.0,
        gamma=0.99,

        # ANN hyperparameters
        alpha=0.0001,
        layers=[128, 64],
        activation='relu',
        loss='mean_squared_error',

        # Experience Replay hyperparameters
        memory_max=2**16,
        memory_min=2**6,
        batch_size=2**5,
    ):
        self.name               = name
        self.env                = env
        self.verbosity          = verbosity

        self.memory             = self.create_memory(memory_max)
        self.memory_min         = memory_min
        self.memory_max         = memory_max

        self.ns                 = env.observation_space.shape[0]
        self.na                 = env.action_space.n
        self.batch_size         = batch_size

        self.load               = load

        self.epsilon            = OpenStruct()
        self.epsilon.max        = epsilon_max
        self.epsilon.min        = epsilon_min
        self.epsilon.decay      = epsilon_decay
        self.epsilon.value      = epsilon_max
        self.gamma              = gamma

        self.is_learning        = False
        self.full_memory        = False

        self.ann                = self.create_ann('{}.ann'.format(name), alpha, layers, activation, loss)
        self.target_ann         = self.create_ann('{}.target_ann'.format(name), alpha, layers, activation, loss)

        self.frames             = 0

    # ANN
    def create_ann(self, name, alpha, layers, activation, loss):
        optimizer = Adam(lr=alpha)

        ann = ANN(
            name,
            input_dim=self.ns,
            output_dim=self.na,
            activation=activation,
            layers=layers,
            loss=loss,
            optimizer=optimizer
        )

        if self.load:
            ann.load()

        return ann

    # Experience Replay
    def create_memory(self, memory_max):
        return deque([], maxlen=memory_max)

    def store_experience(self, experience):
        if not self.full_memory and len(self.memory) == self.memory_max:
            if self.verbosity >= 2:
                print('[INFO] Memory banks filled!')
            self.full_memory = True
        self.memory.append(experience)

    def get_batch(self):
        # Guards
        if self.batch_size > len(self.memory):
            raise ValueError('Memory only has {} experiences!'.format(len(self.memory)))
        if len(self.memory) < self.memory_min:
            raise ValueError('Memory does not have enough experiences!')

        # Sample
        E = random.sample(self.memory, self.batch_size)
        S = np.array([e['s'] for e in E])
        A = np.array([e['a'] for e in E])
        R = np.array([e['r'] for e in E])
        T = np.array([e["s'"] for e in E])
        D = np.array([e['done'] for e in E])

        # Compute
        q = self.ann.predict(S)
        q_t = self.target_ann.predict(T)
        y = np.zeros((len(E), self.na))
        for i in xrange(len(E)):
            r, a, t = R[i], A[i], q[i]

            if D[i]:  # if done
                t[a] = r
            else:
                t[a] = r + self.gamma * np.max(q_t[i])

            y[i] = t

        return (S, y)

    # Agent interface
    def can_learn(self):
        if self.is_learning:
            return True
        elif len(self.memory) >= self.memory_min:
            if self.verbosity >= 2:
                print '[INFO] Started learning!'
            self.is_learning = True
            return True
        else:
            return False

    def get_action(self, state):
        # Epsilon greedy action selection
        if (random.random() < self.epsilon.value):
            return self.env.action_space.sample()
        else:
            reshaped = np.asarray(state).reshape((1, self.ns))
            return np.argmax(self.ann.predict([reshaped]))

    def experience(self, s, a, r, s_, done):
        experience = {
            's': s, 'a': a, 'r': r, "s'": s_, 'done': done
        }
        self.store_experience(experience)
        self.frames += 1
        return experience

    def learn(self):
        if self.can_learn() and not self.load:
            X, y = self.get_batch()
            self.ann.train(X, y, batch_size=self.batch_size)
            return True
        else:
            return False

    def update(self, ann=True, epsilon=True):
        if ann and not self.load:
            self.target_ann.set(self.ann.get())
            self.target_ann.dump()
        if epsilon:
            self.epsilon.value *= self.epsilon.decay

    def clear_frames(self):
        self.frames = 0


class Trial:
    def __init__(
        self,
        name='default',
        episodes=2000,
        training_frequency=1,
        load=False,
        monitor=True,
        record=True,
        stop=True,
        truncate=False,
        verbosity=2,
        params={}
    ):
        env = gym.make('LunarLander-v2')
        if monitor:
            monitor_dir = '{}.loaded'.format(name) if load else name
            env = gym.wrappers.Monitor(env, 'tmp/{}'.format(monitor_dir), force=True)

        self.episodes           = episodes
        self.truncate           = truncate
        self.name               = name
        self.params             = params
        self.training_frequency = training_frequency

        self.load               = load
        self.record             = record
        self.stop               = stop
        self.verbosity          = verbosity

        self.rewards            = []
        self.running_rewards    = deque([], maxlen=100)

        self.agent              = Agent(name, env, load=load, verbosity=verbosity, **params)
        self.env                = env

    def run(self):
        if self.verbosity >= 1:
            print '=== Running {} ==='.format(self.name)
            print self.params
            print

        agent = self.agent
        env   = self.env
        celebrate = False
        max_episode = 0

        if self.record:
            suffix  = '.loaded.csv' if self.load else '.csv'
            csvfile = open(DATA_DIR + self.name + suffix, 'w+')
            writer  = csv.writer(csvfile, delimiter=',')

        for episode in range(self.episodes):
            agent.clear_frames()
            total_reward = 0
            current = env.reset()
            done = False
            while not done:
                previous = current
                action = agent.get_action(current)
                current, reward, done, _unused = env.step(action)
                agent.experience(previous, action, reward, current, done)
                agent.learn()
                total_reward += reward

            if episode % self.training_frequency == 0:
                agent.update(ann=True, epsilon=True)
            else:
                agent.update(ann=False, epsilon=True)

            self.rewards.append(total_reward)
            self.running_rewards.append(total_reward)
            mean_reward = self.mean_reward()

            if self.record:
                writer.writerow([episode, total_reward, mean_reward])

            if self.verbosity >= 1.5:
                if self.verbosity >= 2 or episode % 10 == 0:
                    status = '*' if total_reward > 200 else ' '
                    print '{} #{}, Reward: {:.2f}, Epsilon: {:.2f}, Frames: {}; Running: {:.2f}'.format(status, episode, total_reward, agent.epsilon.value, agent.frames, mean_reward)

            if mean_reward > 200:
                max_episode = episode
                if self.verbosity >= 2 and not celebrate:
                    print
                    print '*** WINNER, WINNER, CHICKEN DINNER ***'
                    print
                    celebrate = True
                if self.stop and len(self.rewards) > 100:
                    break
            elif mean_reward < -100 and episode >= 1000 and self.truncate:
                if self.verbosity >= 1:
                    print
                    print '*** ABORT, ABORT! ***'
                    print
                break

        if self.verbosity >= 1:
            print '* Completed "{}" in {} episodes with {:.2f} mean reward'.format(self.name, episode, mean_reward)
            print

        if self.record:
            csvfile.close()

        return max_episode, mean_reward, self.params, self.rewards

    def mean_reward(self):
        return np.mean(self.running_rewards)


if __name__ == '__main__':
    trial = Trial('default')
    trial.run()
