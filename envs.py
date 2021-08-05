import copy

import gym
import retro
import numpy as np
from abc import abstractmethod
from collections import namedtuple


State = namedtuple('State', ['emulator_state', 'ram', 'observation'])


class RollbackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._emulator_state = None
        self._ram = None
        self._observation = None

    def get_emulator_state(self):
        return self._emulator_state

    def get_ram(self):
        return self._ram

    def get_observation(self):
        return self._observation

    def get_state(self):
        return State(self.get_emulator_state(), self.get_ram(), self.get_observation())

    @abstractmethod
    def set_state(self, state: State):
        raise NotImplementedError

    @abstractmethod
    def _update_state(self, observation):
        raise NotImplementedError

    def reset(self, **kwargs):
        observation = self.env.reset()
        self._update_state(observation)

        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._update_state(observation)

        return observation, reward, done, info


class RetroRollbackWrapperImpl(RollbackWrapper):
    def _update_state(self, observation):
        self._observation = observation
        self._ram = self.unwrapped.get_ram()
        self._emulator_state = self.unwrapped.em.get_state()

    def set_state(self, state: State):
        self.unwrapped.em.set_state(state.emulator_state)
        self._emulator_state, self._ram, self._observation = state


class AtariRollbackWrapperImpl(RollbackWrapper):
    def _update_state(self, observation):
        self._observation = observation
        self._ram = self.unwrapped.ale.getRAM()
        self._emulator_state = self.unwrapped.clone_full_state()

    def set_state(self, state: State):
        self.unwrapped.restore_full_state(state.emulator_state)
        self._emulator_state, self._ram, self._observation = state


def _get_mask(mask, length):
    if mask is not None:
        return mask

    return int.from_bytes(b'\xFF' * length, byteorder='little')


class RamFeatureExtractor:
    def __init__(self, index, length=1, mask=None):
        super().__init__()
        self._index = index
        self._length = length
        self._mask = _get_mask(mask, length)

    def extract(self, ram):
        return int.from_bytes(ram[self._index:self._index + self._length].tobytes(), byteorder='little') & self._mask


class RamFeaturesWrapper(gym.Wrapper):
    def __init__(self, env: RollbackWrapper):
        super().__init__(env)
        self._feature_extractors = {}

    def add_feature_extractor(self, feature_name, feature_extractor):
        self._feature_extractors[feature_name] = feature_extractor

    def add_feature_extractors(self, feature_extractors):
        self._feature_extractors.update(feature_extractors)

    def extract(self, feature_name):
        return self._feature_extractors[feature_name].extract(self.env.get_ram())


class EnsureMinutesLeftWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._minutes_left = tuple('99'.encode('ascii'))
        self._minutes_left_index = 1314

    def _ensure_minutes_left(self):
        emulator_state = self.unwrapped.em.get_state()
        if self._minutes_left != tuple(emulator_state[self._minutes_left_index: self._minutes_left_index + len(self._minutes_left)]):
            emulator_state = bytearray(emulator_state)
            emulator_state[self._minutes_left_index: self._minutes_left_index + len(self._minutes_left)] = self._minutes_left
            emulator_state = bytes(emulator_state)
            self.unwrapped.em.set_state(emulator_state)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._ensure_minutes_left()
        return observation, reward, done, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def make_pop(game, state):
    env = retro.make(
        game,
        inttype=retro.data.Integrations.ALL,
        obs_type=retro.Observations.IMAGE,
        use_restricted_actions=retro.Actions.DISCRETE,
        state=f'{state}.state'
    )
    env = RetroRollbackWrapperImpl(env)
    feature_extractors = {
        'lives': RamFeatureExtractor(index=1743),
        'room': RamFeatureExtractor(index=1246),
        'x': RamFeatureExtractor(index=1243, length=2),
        'y': RamFeatureExtractor(index=1777, mask=224)
    }
    env = RamFeaturesWrapper(env)
    env.add_feature_extractors(feature_extractors)
    env = EnsureMinutesLeftWrapper(env)
    env = MaxAndSkipEnv(env)

    return env, copy.deepcopy(feature_extractors)


def make_montezuma(env_id):
    env = gym.make(env_id)
    env = AtariRollbackWrapperImpl(env)
    feature_extractors = {
        'lives': RamFeatureExtractor(index=58),
        'room': RamFeatureExtractor(index=3),
        'x': RamFeatureExtractor(index=42),
        'y': RamFeatureExtractor(index=43)
    }
    env = RamFeaturesWrapper(env)
    env.add_feature_extractors(feature_extractors)
    env = MaxAndSkipEnv(env)

    return env, copy.deepcopy(feature_extractors)


def make(env_id):
    if env_id.startswith('POP'):
        game, state = env_id.split(':')
        return make_pop(game, state)
    elif env_id.startswith('MontezumaRevenge'):
        return make_montezuma(env_id)
    else:
        raise ValueError(f'Unsupported environment: {env_id}')
