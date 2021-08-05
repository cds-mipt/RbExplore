from config import *
import numpy as np


use_gae = default_config.getboolean('UseGAE')
lam = float(default_config['Lambda'])


def make_train_data(reward, done, value, gamma):
    assert len(reward) == len(done)
    assert len(reward) == len(value)

    discounted_return = []
    adv = []
    if use_gae:
        for i, rewards in enumerate(reward):
            if len(rewards) == 0:
                continue

            dones = done[i]
            values = value[i]
            assert len(rewards) == len(dones)
            assert len(rewards) == len(values) - 1
            gae = 0
            worker_discounted_return = []
            worker_adv = []
            for t in range(len(rewards) - 1, -1, -1):
                delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
                gae = delta + gamma * lam * (1 - dones[t]) * gae
                worker_adv.append(gae)
                worker_discounted_return.append(gae + values[t])

            adv += reversed(worker_adv)
            discounted_return += reversed(worker_discounted_return)
    else:
        for i, rewards in enumerate(reward):
            if len(rewards) == 0:
                continue

            dones = done[i]
            values = value[i]
            assert len(rewards) == len(dones)
            assert len(rewards) == len(values) - 1

            running_add = values[-1]
            worker_discounted_return = []
            worker_adv = []
            for t in range(len(rewards) - 1, -1, -1):
                running_add = rewards[t] + gamma * running_add * (1 - dones[t])
                worker_discounted_return.append(running_add)
                worker_adv.append(running_add - values[t])

            adv += reversed(worker_adv)
            discounted_return += reversed(worker_discounted_return)

    return np.asarray(discounted_return), np.asarray(adv)


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RewardForwardFilter(object):
    def __init__(self, n_workers, gamma):
        self.rewems = [0] * n_workers
        self.gamma = gamma

    def update(self, rews):
        assert len(self.rewems) == len(rews)
        cumulative_rewards = []
        for i, rewards in enumerate(rews):
            for reward in rewards:
                self.rewems[i] = self.rewems[i] * self.gamma + reward
                cumulative_rewards.append(self.rewems[i])

        return np.asarray(cumulative_rewards)
