import math
import random

import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim

from torch.distributions.categorical import Categorical

from rnd import CnnActorCriticNetwork, RNDModel


class RandomAgent:
    def __init__(self, env):
        super().__init__()
        self.n_actions = env.action_space.n

    def get_action(self, state=None):
        return random.randrange(self.n_actions)


class NoopAgent:
    def __init__(self):
        super().__init__()

    def get_action(self, state=None):
        return 0


class RNDAgent(object):
    def __init__(
            self,
            output_size,
            learning_rate=1e-4,
            ent_coef=0.01,
            epoch=3,
            batch_size=128,
            ppo_eps=0.1,
            update_proportion=0.25,
            use_gpu=False):
        self.model = CnnActorCriticNetwork(output_size)
        self.output_size = output_size
        self.epoch = epoch
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.ppo_eps = ppo_eps
        self.update_proportion = update_proportion
        self.device = torch.device('cuda' if use_gpu else 'cpu')

        self.rnd = RNDModel()
        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.rnd.predictor.parameters()),
                                    lr=learning_rate)
        self.rnd = self.rnd.to(self.device)

        self.model = self.model.to(self.device)

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        policy, value_ext, value_int = self.model(state)
        action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()

        action = self.random_choice_prob_index(action_prob)

        return action, value_ext.data.cpu().numpy().squeeze(axis=1), value_int.data.cpu().numpy().squeeze(axis=1), policy.detach()

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def compute_intrinsic_reward(self, next_obs):
        next_obs = torch.FloatTensor(next_obs).to(self.device)

        target_next_feature = self.rnd.target(next_obs)
        predict_next_feature = self.rnd.predictor(next_obs)
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()

    def train_model(self, s_batch, target_ext_batch, target_int_batch, y_batch, adv_batch, next_obs_batch, old_policy):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        target_ext_batch = torch.FloatTensor(target_ext_batch).to(self.device)
        target_int_batch = torch.FloatTensor(target_int_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)

        sample_range = np.arange(len(s_batch))
        forward_mse = nn.MSELoss(reduction='none')

        with torch.no_grad():
            policy_old_list = torch.stack(old_policy).to(self.device)

            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)

        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                predict_next_state_feature, target_next_state_feature = self.rnd(next_obs_batch[sample_idx])

                forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
                mask = torch.rand(len(forward_loss)).to(self.device)
                mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))

                policy, value_ext, value_int = self.model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_ext_loss = F.mse_loss(value_ext.sum(1), target_ext_batch[sample_idx])
                critic_int_loss = F.mse_loss(value_int.sum(1), target_int_batch[sample_idx])

                critic_loss = critic_ext_loss + critic_int_loss
                # critic_loss = critic_int_loss

                entropy = m.entropy().mean()

                self.optimizer.zero_grad()
                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy + forward_loss
                loss.backward()
                global_grad_norm_(list(self.model.parameters())+list(self.rnd.predictor.parameters()))
                self.optimizer.step()


def global_grad_norm_(parameters, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == math.inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm
