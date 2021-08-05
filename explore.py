import collections
import random

import cv2
import numpy as np

import sptm
import envs

from torch.multiprocessing import Pipe, Process

import utils
from config import *

gamma = float(default_config['Gamma'])
int_gamma = float(default_config['IntGamma'])
ext_coef = float(default_config['ExtCoef'])
int_coef = float(default_config['IntCoef'])
trim_steps = int(default_config['TrimSteps'])
sticky_action = default_config.getboolean('StickyAction')
p = float(default_config['ActionProb'])
terminate_on_life_loss = default_config.getboolean('TerminateOnLifeLoss')
rollout_steps = int(default_config['RolloutSteps'])


class Cluster:
    def __init__(self, state: envs.State, trajectory_length: int):
        self.state = state
        self.trajectory_length = trajectory_length
        self.score = 0


class TrajectoryElement:
    def __init__(self, action, observation, reward, done, state: envs.State):
        self.action = action
        self.observation = observation
        self.reward = reward
        self.done = done
        self.state = state


RollbackRequest = collections.namedtuple('RollbackRequest', ['cluster_id', 'cluster'])
ActionRequest = collections.namedtuple('ActionRequest', ['action'])
ActionResponse = collections.namedtuple('ActionResponse', ['frame_stack', 'reward', 'force_done', 'done', 'log_reward'])
DoneResponse = collections.namedtuple(
    'DoneResponse',
    ['cluster_id', 'clusters', 'observations', 'images1', 'images2', 'labels', 'visited_rooms', 'n_frames_compute',
     'n_frames_true', 'trim', 'ext_return']
)


def discretize(value, interval=32):
    return (value // interval) * interval


class ExplorationWorker(Process):
    def __init__(self, env_maker, worker_id, child_conn, max_steps, frame_stack_size=4, policy_shape=(84, 84),
                 cluster_shape=(96, 96)):
        super().__init__()
        self.daemon = True
        self.env = None
        self.env_factory = env_maker
        self.worker_id = worker_id
        self.max_steps = max_steps
        self.child_conn = child_conn

        self.sticky_action = sticky_action
        self.p = p
        self.policy_shape = policy_shape
        self.cluster_shape = cluster_shape

        self.frame_stack_size = frame_stack_size
        self.terminate_on_life_loss = terminate_on_life_loss
        self.ignore_black = True
        self.trim_steps = trim_steps
        self.ext_gamma = gamma
        self.steps_after_done = 3

    def run(self):
        self.env = self.env_factory()
        self.env.reset()

        super().run()
        episode = 0
        while True:
            rollback_request = self.child_conn.recv()
            start_cluster_id = rollback_request.cluster_id
            start_cluster = rollback_request.cluster

            self.rollback(start_cluster.state)
            frame_stack = self.init_frame_stack(start_cluster.state.observation)
            self.child_conn.send(ActionResponse(
                frame_stack=frame_stack[:, :, :], reward=None, force_done=None, done=None, log_reward=None
            ))

            clusters = [start_cluster]
            observations = [self.preprocess_cluster(start_cluster.state.observation)]
            visited_rooms = collections.defaultdict(set)
            visited_rooms[self.env.extract('room')].add((self.env.extract('x'), self.env.extract('y')))

            episode += 1
            done = False
            force_done = False
            steps = 0
            last_action = 0
            ext_return = 0
            gamma_pow = 1
            while not done:
                action_request = self.child_conn.recv()
                action = action_request.action

                if self.sticky_action:
                    if np.random.rand() <= self.p:
                        action = last_action
                    last_action = action

                lives = self.env.extract('lives')
                s, reward, done, info = self.env.step(action)
                steps += 1
                visited_rooms[self.env.extract('room')].add((self.env.extract('x'), self.env.extract('y')))
                cluster = Cluster(state=self.env.get_state(), trajectory_length=start_cluster.trajectory_length + steps)
                observation = cluster.state.observation
                if not self.ignore_black or observation.max() != 0:
                    clusters.append(cluster)
                    observations.append(self.preprocess_cluster(observation))

                if self.terminate_on_life_loss:
                    done = done or self.env.extract('lives') < lives

                if self.max_steps <= steps:
                    done = True
                    force_done = True

                log_reward = reward
                ext_return += gamma_pow * reward
                gamma_pow *= self.ext_gamma

                frame_stack[:3, :, :] = frame_stack[1:, :, :]
                frame_stack[3, :, :] = self.preprocess_policy(s)

                self.child_conn.send(ActionResponse(
                    frame_stack=frame_stack[:, :, :],
                    reward=reward,
                    force_done=force_done,
                    done=done,
                    log_reward=log_reward
                ))

            for _ in range(self.steps_after_done):
                self.env.step(0)
                visited_rooms[self.env.extract('room')].add((self.env.extract('x'), self.env.extract('y')))

            images1, images2, labels = sptm.create_training_data(observations)
            n_frames_compute = steps
            n_frames_true = start_cluster.trajectory_length + steps
            trim = self.trim_steps - int(force_done)
            observations = observations
            clusters = clusters

            self.child_conn.send(DoneResponse(
                cluster_id=start_cluster_id,
                clusters=clusters,
                observations=np.asarray(observations, dtype=np.float32),
                images1=images1,
                images2=images2,
                labels=labels,
                visited_rooms=visited_rooms,
                n_frames_compute=n_frames_compute,
                n_frames_true=n_frames_true,
                trim=trim,
                ext_return=ext_return
            ))

    def rollback(self, state: envs.State):
        self.env.set_state(state)

    def preprocess(self, observation, shape):
        return np.float32(cv2.resize(cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY), shape))

    def preprocess_policy(self, observation):
        return self.preprocess(observation, self.policy_shape)

    def preprocess_cluster(self, observation):
        return self.preprocess(observation, self.cluster_shape)[np.newaxis] / 255.

    def init_frame_stack(self, s):
        return np.asarray([self.preprocess_policy(s)] * self.frame_stack_size)


class Explorer:
    def __init__(self, env_maker, explore_steps, n_workers, r_network_trainer, sptm_instance, agent):
        self.env_maker = env_maker
        self.explore_steps = explore_steps
        self.n_workers = n_workers
        self.n_iterations = 0
        self.train_interval = -1
        self.last_train = 0
        self.n_frames_true = 0
        self.n_frames_compute = 0
        self.clip_int_reward = 5
        self.visited_rooms = collections.defaultdict(set)
        self.r_network_trainer = r_network_trainer
        self.sptm_instance = sptm_instance
        self.agent = agent
        self.n_step_rollout = rollout_steps
        self.discounted_reward = utils.RewardForwardFilter(self.n_workers, int_gamma)

        self.workers = []
        self.parent_conns = []
        self.child_conns = []
        for idx in range(self.n_workers):
            parent_conn, child_conn = Pipe()
            worker = ExplorationWorker(env_maker, idx, child_conn, explore_steps)
            worker.start()
            self.workers.append(worker)
            self.parent_conns.append(parent_conn)
            self.child_conns.append(child_conn)

        self.reward_rms = utils.RunningMeanStd()
        self.obs_rms = utils.RunningMeanStd(shape=(1, 1, 84, 84))

    def explore(self, pretrain=False):
        self.n_iterations += 1
        total_state, total_reward, total_done, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy, total_policy_np = \
            [[[] for _ in range(self.n_workers)] for _ in range(10)]
        nodes, probabilities = self.sptm_instance.least_frequent_nodes(self.n_workers)
        node2probability = {node: probability for node, probability in zip(nodes, probabilities)}
        states = []
        for parent_conn, node in zip(self.parent_conns, nodes):
            parent_conn.send(RollbackRequest(node, self.sptm_instance.node2cluster[node]))
            action_response = parent_conn.recv()
            states.append(action_response.frame_stack)

        states = np.stack(states)
        worker_ids = list(range(self.n_workers))
        while len(worker_ids) > 0:
            actions, value_ext, value_int, policy = self.agent.get_action(np.float32(states) / 255.)

            for index, worker_id in enumerate(worker_ids):
                self.parent_conns[worker_id].send(ActionRequest(action=actions[index]))

            next_states, rewards, dones, force_dones, log_rewards, next_obs = [], [], [], [], [], []
            done_worker_ids = set()
            for worker_id in worker_ids:
                action_response = self.parent_conns[worker_id].recv()
                frame_stack = action_response.frame_stack
                reward = action_response.reward
                force_done = action_response.force_done
                done = action_response.done
                log_reward = action_response.log_reward

                next_states.append(frame_stack)
                rewards.append(reward)
                dones.append(done)
                force_dones.append(force_done)
                log_rewards.append(log_reward)
                next_obs.append(frame_stack[3, :, :].reshape([1, 84, 84]))

                if done:
                    done_worker_ids.add(worker_id)

            next_states = np.stack(next_states)
            rewards = np.hstack(rewards)
            dones = np.hstack(dones)
            force_dones = np.hstack(force_dones)
            next_obs = np.stack(next_obs)

            intrinsic_reward = self.agent.compute_intrinsic_reward(
                ((next_obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var)).clip(-self.clip_int_reward, self.clip_int_reward))
            intrinsic_reward = np.hstack(intrinsic_reward)

            for index, worker_id in enumerate(worker_ids):
                total_next_obs[worker_id].append(next_obs[index])
                total_int_reward[worker_id].append(intrinsic_reward[index])
                total_state[worker_id].append(states[index])
                total_reward[worker_id].append(rewards[index])
                total_done[worker_id].append(dones[index])
                total_action[worker_id].append(actions[index])
                total_ext_values[worker_id].append(value_ext[index])
                total_int_values[worker_id].append(value_int[index])
                total_policy[worker_id].append(policy[index])
                total_policy_np[worker_id].append(policy[index].cpu().numpy())

            indices = [i for i, worker_id in enumerate(worker_ids) if worker_id not in done_worker_ids]
            worker_ids = [worker_ids[i] for i in indices]
            states = next_states[indices]

            if pretrain or (len(worker_ids) > 0 and sum(len(worker_states) for worker_states in total_state) < self.n_workers * self.n_step_rollout):
                continue

            actual_worker_ids = [worker_id for worker_id, worker_states in enumerate(total_state) if len(worker_states) > 0]
            last_states = np.stack([total_state[worker_id][-1] for worker_id in actual_worker_ids])
            _, value_ext, value_int, _ = self.agent.get_action(np.float32(last_states) / 255.)
            for index in range(last_states.shape[0]):
                total_ext_values[actual_worker_ids[index]].append(value_ext[index])
                total_int_values[actual_worker_ids[index]].append(value_int[index])

            cumulative_rewards = self.discounted_reward.update(total_int_reward)
            mean, std, count = np.mean(cumulative_rewards), np.std(cumulative_rewards), cumulative_rewards.shape[0]
            self.reward_rms.update_from_moments(mean, std ** 2, count)

            norm_reward = np.sqrt(self.reward_rms.var)
            for i in range(len(total_int_reward)):
                for j in range(len(total_int_reward[i])):
                    total_int_reward[i][j] /= norm_reward

            ext_target, ext_adv = utils.make_train_data(total_reward, total_done, total_ext_values, gamma)

            non_episodic_done = [[0] * len(int_reward) for int_reward in total_int_reward]
            int_target, int_adv = utils.make_train_data(total_int_reward, non_episodic_done, total_int_values, int_gamma)

            total_adv = int_adv * int_coef + ext_adv * ext_coef

            total_next_obs = [obs for worker_obs in total_next_obs for obs in worker_obs]
            total_next_obs = np.stack(total_next_obs)
            self.obs_rms.update(total_next_obs)

            total_state = [state for worker_states in total_state for state in worker_states]
            total_state = np.stack(total_state)

            total_action = [action for worker_actions in total_action for action in worker_actions]
            total_action = np.stack(total_action)

            total_policy = [state_policy for worker_state_policies in total_policy for state_policy in worker_state_policies]

            self.agent.train_model(np.float32(total_state) / 255., ext_target, int_target, total_action,
                              total_adv, ((total_next_obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var)).clip(-5, 5),
                              total_policy)

            total_state, total_reward, total_done, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy, total_policy_np = \
                [[[] for _ in range(self.n_workers)] for _ in range(10)]

        results = []
        visited_rooms_discretized = collections.defaultdict(set)
        for parent_conn in self.parent_conns:
            done_response = parent_conn.recv()
            results.append(done_response)
            cluster_id = done_response.cluster_id
            clusters = done_response.clusters
            observations = done_response.observations
            images1 = done_response.images1
            images2 = done_response.images2
            labels = done_response.labels
            visited_rooms = done_response.visited_rooms
            n_frames_compute = done_response.n_frames_compute
            n_frames_true = done_response.n_frames_true
            self.r_network_trainer.add_to_history(images1, images2, labels)

            _, prefixes = self.sptm_instance.build_prefix_trajectory(cluster_id)
            prefix_image_locations = [(prefix_id, prefix_image_id) for i, prefix in enumerate(prefixes) for prefix_id, prefix_image_id in zip([i] * prefix.shape[0], range(prefix.shape[0]))]
            full_prefix_len = len(prefix_image_locations)
            if full_prefix_len > 0:
                trajectory_negative_examples = []
                prefix_negative_examples = []
                for i, label in enumerate(labels):
                    if label == 0:
                        image = images1[i] if random.random() < 0.5 else images2[i]
                        trajectory_negative_examples.append(image)
                        index = random.randrange(full_prefix_len)
                        prefix_id, prefix_image_id = prefix_image_locations[index]
                        prefix_negative_examples.append(prefixes[prefix_id][prefix_image_id])

                negative_labels = np.zeros(len(trajectory_negative_examples), dtype=np.int64)
                trajectory_negative_examples = np.asarray(trajectory_negative_examples, dtype=np.float32)
                prefix_negative_examples = np.asarray(prefix_negative_examples, dtype=np.float32) / 255.
                self.r_network_trainer.add_to_history(trajectory_negative_examples, prefix_negative_examples,
                                                      negative_labels)

            for room, coordinates in visited_rooms.items():
                self.visited_rooms[room] |= coordinates
                visited_rooms_discretized[room] |= set([(discretize(x), discretize(y)) for x, y in coordinates])

            self.n_frames_true += n_frames_true
            self.n_frames_compute += n_frames_compute

        validation_stats = None
        if self.r_network_trainer.can_train():
            validation_stats = self.r_network_trainer.train()
            self.train_interval = self.n_iterations - self.last_train
            self.last_train = self.n_iterations
            self.sptm_instance.update_embeddings()
        else:
            self.train_interval = -1

        exploration_results = []
        n_empty_trajectories = 0
        for result in results:
            cluster_id, clusters, observations, images1, images2, labels, visited_rooms, n_frames_compute, n_frames_true, trim, ext_return = result
            assert len(clusters) == observations.shape[0]
            exploration_results.append({
                'cluster_id': cluster_id,
                'probability': node2probability[cluster_id],
                'observations': np.around(observations * 255).astype(np.uint8)
            })
            observations = observations[:-trim]
            clusters = clusters[:-trim]
            if len(clusters) == 0:
                n_empty_trajectories += 1

            self.sptm_instance.add_trajectory(cluster_id, clusters, observations, ext_return=ext_return)

        return exploration_results, visited_rooms_discretized, n_empty_trajectories, validation_stats
