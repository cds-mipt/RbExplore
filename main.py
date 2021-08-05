import math
import multiprocessing
import os
import shutil
import sys
import time
import pickle
import json

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from config import *
import agents
import envs
import explore
import retro

import sptm


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32) or isinstance(obj, np.float64) or isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return obj.item()
        return json.JSONEncoder.default(self, obj)


def sampling_freq(sptm_instance, horizon):
    history = np.hstack(list(sptm_instance.sampling_history)[-horizon:])
    unique, unique_counts = np.unique(history, return_counts=True)
    freq = [0] * len(sptm_instance.node2cluster)
    for node, count in zip(unique, unique_counts):
        freq[node] = count

    return freq


def log_exploration_data(exploration_results, base_path):
    exploration_data_directory = os.path.join(base_path, 'exploration_data')
    shutil.rmtree(exploration_data_directory, ignore_errors=True)
    for result in exploration_results:
        cluster_directory = os.path.join(
            exploration_data_directory, f'cluster_id-{result["cluster_id"]:06d}_prob-{result["probability"]:.3f}'
        )
        os.makedirs(cluster_directory, exist_ok=True)
        for i, observation in enumerate(np.squeeze(result['observations'], axis=1)):
            observation_filename = os.path.join(cluster_directory, f'{i:06d}.png')
            cv2.imwrite(observation_filename, observation)


def log_memory_usage_stats(memory_usage_stats):
    print(
        f'n_clusters={memory_usage_stats["n_clusters"]},'
        f' emulator_states={memory_usage_stats["emulator_states_size"]:.0f} MB,'
        f' cluster_observations={memory_usage_stats["cluster_observations_size"]:.0f} MB,'
        f' state_observations={memory_usage_stats["state_observations_size"]:.0f} MB'
    )
    print(
        f'n_prefixes={memory_usage_stats["n_prefixes"]},'
        f' n_prefix_images={memory_usage_stats["n_prefix_images"]},'
        f' prefix_images={memory_usage_stats["prefix_images_size"]:.0f} MB'
    )


def log_iteration_results(explorer, visited_rooms_discretized, n_empty_trajectories, validation_stats):
    if validation_stats is not None:
        print('Class balance: 0:{}, 1:{}'.format(validation_stats['class_balance'][0], validation_stats['class_balance'][1]))
        print('Validation loss:', validation_stats['val_loss'])
        print('Validation accuracy:', validation_stats['accuracy'])
        print('Validation recall 0:', validation_stats['recall_0'])
        print('Validation precision 0:', validation_stats['precision_0'])
        print('Validation F1 0:', validation_stats['f1_0'])
        print('Validation recall 1:', validation_stats['recall_1'])
        print('Validation precision 1:', validation_stats['precision_1'])
        print('Validation F1 1:', validation_stats['f1_1'])

    print(f'Iterations: {explorer.n_iterations}')
    print(f'Game step: {explorer.n_frames_true}')
    print(f'Compute step: {explorer.n_frames_compute}')
    print(f'Time: {int(time.time() - start_time)}')
    print(f'Compute clusters: {len(explorer.sptm_instance.node2cluster)}')
    print(f'Rooms: {set(explorer.visited_rooms.keys())}')
    print(f'Train interval: {explorer.train_interval}')
    print(f'Min similarity: {explorer.sptm_instance.min_similarity}')
    print(f'Max similarity: {explorer.sptm_instance.max_similarity}')
    print(f'Empty trajectories: {n_empty_trajectories}')
    print(f'Visited rooms coordinates:\n{visited_rooms_discretized}')


def log_data(explorer, feature_extractors, iteration, base_path):
    sptm_instance = explorer.sptm_instance
    observations_numpy = sptm_instance.node2observation
    cells = sptm_instance.node2cluster
    log_graph = sptm_instance.graph.copy()

    frequency10 = sampling_freq(sptm_instance, 10)
    frequency50 = sampling_freq(sptm_instance, 50)
    frequency100 = sampling_freq(sptm_instance, 100)

    for node, data in log_graph.nodes(data=True):
        data['prefix_length_mean'] = data['prefix_length'] / (data['visits'] + 1e-8)
        data['trajectory_length_mean'] = data['trajectory_length'] / (data['starts'] + 1e-8)
        data['ext_return_mean'] = data['ext_return'] / (data['starts'] + 1e-8)
        data['frequency10'] = frequency10[node]
        data['frequency50'] = frequency50[node]
        data['frequency100'] = frequency100[node]
        data['room'] = feature_extractors['room'].extract(cells[node].state.ram)
        data['x'] = feature_extractors['x'].extract(cells[node].state.ram)
        data['y'] = feature_extractors['y'].extract(cells[node].state.ram)

    iteration_directory = base_path + f'/iteration_{iteration:06d}/'
    observation_directory = iteration_directory + 'observation/'
    emulator_state_directory = iteration_directory + 'emulator-state/'
    os.makedirs(observation_directory, exist_ok=True)
    os.makedirs(emulator_state_directory, exist_ok=True)
    for i, observation_numpy in enumerate(observations_numpy):
        plt.imsave(observation_directory + f'{i:06d}.png', observation_numpy.squeeze(), cmap='gray')

    for i, cell in enumerate(cells):
        with open(emulator_state_directory + f'{i:06d}.pickle', 'wb') as file_obj:
            pickle.dump({'emulator_state': cell.state.emulator_state}, file_obj)

    with open(iteration_directory + 'rooms_data.json', mode='w', encoding='utf-8') as file_obj:
        rooms_data = {room: list(coordinates) for room, coordinates in explorer.visited_rooms.items()}
        json.dump({"iteration": iteration, "steps": explorer.n_frames_compute, "rooms": rooms_data}, file_obj, sort_keys=True)

    with open(iteration_directory + 'graph.json', mode='w', encoding='utf-8') as file_obj:
        json.dump(nx.node_link_data(log_graph), file_obj, indent=4, cls=NumpyEncoder)


def log_rnetwork_weights(iteration, base_path, r_network):
    r_network_directory = os.path.join(base_path, 'r_network')
    os.makedirs(r_network_directory, exist_ok=True)
    torch.save(r_network.state_dict(), os.path.join(r_network_directory, f'r_network_{iteration:06d}.pt'))


if __name__ == '__main__':
    env_id = default_config['EnvID']
    use_gpu = default_config.getboolean('UseGPU')
    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    rnd_batch_size = int(default_config['RNDBatchSize'])
    rnd_learning_rate = float(default_config['RNDLearningRate'])
    entropy_coef = float(default_config['Entropy'])
    update_proportion = float(default_config['UpdateProportion'])
    custom_integrations_path = default_config['CustomIntegrationsPath']
    base_path = default_config['BasePath']
    warm_up_steps = int(default_config['WarmUpSteps'])
    pretrain_steps = int(default_config['PretrainSteps'])
    log_data_interval = int(default_config['LogDataInterval'])
    merge_interval = int(default_config['MergeInterval'])
    log_rnetwork_interval = int(default_config['LogRNetworkInterval'])
    explore_steps = int(default_config['ExploreSteps'])
    n_workers = int(default_config['NWorkers'])
    similarity_threshold = float(default_config['SimilarityThreshold'])
    r_network_batch_size = int(default_config['RNetworkBatchSize'])
    r_network_learning_rate = float(default_config['RNetworkLearningRate'])
    sptm_batch_size = int(default_config['SPTMBatchSize'])
    n_prefixes = int(default_config['NPrefixes'])
    depth_limit = int(default_config['DepthLimit'])

    os.makedirs(base_path)
    sys.stdout = open(os.path.join(base_path, 'log.out'), 'w')
    sys.stderr = open(os.path.join(base_path, 'log.err'), 'w')
    retro.data.Integrations.add_custom_path(custom_integrations_path)

    env, feature_extractors = envs.make(env_id)
    init_observation = env.reset()
    init_observation = np.float32(cv2.resize(cv2.cvtColor(init_observation, cv2.COLOR_RGB2GRAY), (96, 96)))[np.newaxis] / 255.
    init_cluster = explore.Cluster(state=env.get_state(), trajectory_length=0)
    env.close()

    device = torch.device('cuda' if use_gpu else 'cpu')
    similarity_network = sptm.SimilarityNetwork()
    embedding_network = sptm.build_embedding_network()
    r_network = sptm.RNetwork(embedding_network, similarity_network).to(device)
    r_network_trainer = sptm.RNetworkTrainer(
        r_network, device, batch_size=r_network_batch_size, lr=r_network_learning_rate
    )
    sptm_instance = sptm.SPTM(
        r_network, init_cluster=init_cluster, init_observation_numpy=init_observation, device=device,
        similarity_threshold=similarity_threshold, batch_size=sptm_batch_size, n_prefixes=n_prefixes,
        depth_limit=depth_limit
    )

    agent = agents.RNDAgent(
        output_size=env.action_space.n,
        learning_rate=rnd_learning_rate,
        ent_coef=entropy_coef,
        epoch=epoch,
        batch_size=rnd_batch_size,
        ppo_eps=ppo_eps,
        update_proportion=update_proportion,
        use_gpu=use_gpu
    )

    explorer = explore.Explorer(
        env_maker=lambda: envs.make(env_id)[0],
        explore_steps=explore_steps,
        n_workers=n_workers,
        r_network_trainer=r_network_trainer,
        sptm_instance=sptm_instance,
        agent=agent
    )

    n_iteration = 0
    init_sptm = False
    log_process = None
    start_time = time.time()

    while True:
        if not init_sptm and explorer.n_frames_compute >= warm_up_steps:
            sptm_instance = sptm.SPTM(
                r_network, init_cluster=init_cluster, init_observation_numpy=init_observation, device=device,
                similarity_threshold=similarity_threshold, batch_size=sptm_batch_size, n_prefixes=n_prefixes,
                depth_limit=depth_limit
            )
            explorer.sptm_instance = sptm_instance
            explorer.visited_rooms.clear()
            init_sptm = True

        if explorer.n_frames_compute < warm_up_steps:
            explorer.sptm_instance.similarity_threshold = similarity_threshold * (explorer.n_frames_compute / warm_up_steps) ** 0.5
        else:
            explorer.sptm_instance.similarity_threshold = similarity_threshold

        explorer.sptm_instance.min_similarity = math.inf
        explorer.sptm_instance.max_similarity = -math.inf
        exploration_results, visited_rooms_discretized, n_empty_trajectories, validation_stats =\
            explorer.explore(pretrain=explorer.n_frames_compute < pretrain_steps)

        log_iteration_results(explorer, visited_rooms_discretized, n_empty_trajectories, validation_stats)
        log_memory_usage_stats(explorer.sptm_instance.get_memory_usage())

        if n_iteration % merge_interval == 0:
            explorer.sptm_instance.min_similarity = math.inf
            explorer.sptm_instance.max_similarity = -math.inf
            explorer.sptm_instance.multi_merge()
            print(f'Merge min similarity: {explorer.sptm_instance.min_similarity}')
            print(f'Merge max similarity: {explorer.sptm_instance.max_similarity}')
            print(f'Merge compute clusters: {len(explorer.sptm_instance.node2cluster)}')
            log_memory_usage_stats(explorer.sptm_instance.get_memory_usage())

        print('-' * 80, flush=True)

        if explorer.n_frames_compute >= warm_up_steps and n_iteration % log_data_interval == 0:
            log_data(explorer, feature_extractors, n_iteration, base_path)

        if explorer.n_frames_compute >= warm_up_steps and n_iteration % log_rnetwork_interval == 0:
            log_rnetwork_weights(n_iteration, base_path, r_network)

        n_iteration += 1

        if log_process is not None:
            log_process.join()

        log_process = multiprocessing.Process(target=log_exploration_data, args=(exploration_results, base_path))
        log_process.start()
