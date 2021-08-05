import gc
import itertools
import math

import networkx as nx
import numpy as np
import torch
from torch import nn, optim
import torchvision.models
import random
import collections

from typing import Dict


NEGATIVE_SAMPLE_MULTIPLIER = 5


def build_embedding_network(in_channels=1, embedding_dim=512):
    resnet18 = torchvision.models.resnet18(pretrained=False, progress=False, num_classes=embedding_dim)
    resnet18.conv1 = nn.Conv2d(
        in_channels,
        resnet18.conv1.out_channels,
        kernel_size=resnet18.conv1.kernel_size,
        stride=resnet18.conv1.stride,
        padding=resnet18.conv1.padding,
        bias=resnet18.conv1.bias is not None
    )
    nn.init.kaiming_normal_(resnet18.conv1.weight, mode='fan_out', nonlinearity='relu')

    return resnet18


class SimilarityNetwork(nn.Module):
    def __init__(self, embedding_dim=512, top_hidden=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.bottom = nn.Sequential(
            self._batch_norm_relu_linear(in_features=2 * self.embedding_dim, out_features=self.embedding_dim),
            *[self._batch_norm_relu_linear(in_features=self.embedding_dim) for _ in range(top_hidden - 1)]
        )
        self.head = self._batch_norm_relu_linear(in_features=self.embedding_dim, out_features=2)
        self.softmax = nn.Softmax()

    @staticmethod
    def _batch_norm_relu_linear(in_features, out_features=None):
        if out_features is None:
            out_features = in_features

        return nn.Sequential(
            nn.BatchNorm1d(num_features=in_features),
            nn.ReLU(),
            nn.Linear(in_features=in_features, out_features=out_features)
        )

    def forward(self, x1, x2, logits=True):
        x = torch.cat((x1, x2), dim=1)
        x = self.bottom(x)
        x = self.head(x)
        if logits:
            return x
        else:
            return self.softmax(x)


class RNetwork(nn.Module):
    def __init__(self, embedding_network, similarity_network):
        super().__init__()
        self.embedding_network = embedding_network
        self.similarity_network = similarity_network

    def forward(self, x1, x2, logits=True):
        x1, x2 = self.embedding_network(x1), self.embedding_network(x2)
        return self.similarity_network(x1, x2, logits=logits)


class RNetworkTrainer:
    def __init__(
            self, r_network, device, shape=(1, 96, 96), lr=0.0001,
            batch_size=256, grad_norm=1.0, can_train_history_size=1000):
        super().__init__()
        self.r_network = r_network
        self.observation_shape = shape
        self.device = device
        self.lr = lr
        self.optimizer = optim.Adam(params=self.r_network.parameters(), lr=self.lr)
        self.grad_norm = grad_norm
        self.batch_size = batch_size
        self.n_epoch = 1
        self.criterion = nn.CrossEntropyLoss()
        self.history_size = 0
        self.can_train_history_size = can_train_history_size
        self.history_storage_size = self.can_train_history_size // 2 * 3
        self.history_images = np.zeros((2, self.history_storage_size, *self.observation_shape), dtype=np.float32)
        self.history_labels = np.zeros(self.history_storage_size, dtype=np.int64)
        self.train_iteration = 0

    def add_to_history(self, images1, images2, labels):
        length = labels.shape[0]
        if length == 0:
            return

        if self.history_size + length > self.history_storage_size:
            self.history_storage_size = (self.history_size + length) // 2 * 3
            history_images = np.zeros((2, self.history_storage_size, *self.observation_shape), dtype=np.float32)
            history_labels = np.zeros(self.history_storage_size, dtype=np.int64)
            history_images[:, :self.history_images.shape[1], ...] = self.history_images
            history_labels[:self.history_labels.shape[0]] = self.history_labels
            self.history_images = history_images
            self.history_labels = history_labels

        self.history_images[0, self.history_size:self.history_size + length, ...] = images1
        self.history_images[1, self.history_size:self.history_size + length, ...] = images2
        self.history_labels[self.history_size:self.history_size + length] = labels

        self.history_size += length

    def can_train(self):
        return self.history_size >= self.can_train_history_size

    def validate(self, labels, images1, images2):
        self.r_network.eval()
        n_0 = np.sum(labels == 0) / labels.shape[0]
        n_1 = np.sum(labels == 1) / labels.shape[0]

        val_loss = 0
        prediction_labels = []
        with torch.no_grad():
            for i in range(0, labels.shape[0], self.batch_size):
                val_labels_batch = torch.tensor(labels[i:i + self.batch_size], device=self.device)
                val_images1_batch = torch.tensor(images1[i:i + self.batch_size], device=self.device)
                val_images2_batch = torch.tensor(images2[i:i + self.batch_size], device=self.device)
                val_prediction_batch = self.r_network(val_images1_batch, val_images2_batch, logits=True)
                val_loss += self.criterion(val_prediction_batch, val_labels_batch).item() * val_labels_batch.size(0)
                prediction_labels.append(torch.argmax(val_prediction_batch, dim=1).cpu().numpy())

        predictions = np.concatenate(prediction_labels)
        accuracy = accuracy_score(predictions, labels)
        recall_0, precision_0, f1_0, recall_1, precision_1, f1_1 = metrics(predictions, labels)
        return {
            'class_balance': (n_0, n_1),
            'val_loss': val_loss / labels.shape[0], 'accuracy': accuracy,
            'recall_0': recall_0, 'precision_0': precision_0, 'f1_0': f1_0,
            'recall_1': recall_1, 'precision_1': precision_1, 'f1_1': f1_1,
        }

    def train(self):
        if self.history_size < 1:
            return

        images1 = self.history_images[0, :self.history_size, ...]
        images2 = self.history_images[1, :self.history_size, ...]
        labels = self.history_labels[:self.history_size]

        validation_stats = self.validate(labels, images1, images2)

        self.r_network.train()
        for i in range(self.n_epoch):
            indices = np.random.choice(labels.shape[0], size=labels.shape[0], replace=False)
            start = 0
            prediction_labels = []
            train_loss = 0
            while start < indices.shape[0]:
                end = start + self.batch_size
                if indices.shape[0] - end <= 2:
                    end = indices.shape[0]
                batch_images1 = torch.tensor(images1[indices[start:end], ...], device=self.device)
                batch_images2 = torch.tensor(images2[indices[start:end], ...], device=self.device)
                batch_labels = torch.tensor(labels[indices[start:end]], device=self.device)
                prediction = self.r_network(batch_images1, batch_images2, logits=True)

                self.optimizer.zero_grad()
                loss = self.criterion(prediction, batch_labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.r_network.parameters(), max_norm=self.grad_norm)
                self.optimizer.step()
                start = end
                if i == self.n_epoch - 1:
                    train_loss += loss.item() * batch_labels.size(0)
                    prediction_labels.append(torch.argmax(prediction, dim=1).cpu().numpy())

        self.history_size = 0
        self.train_iteration += 1

        return validation_stats


def accuracy_score(predicted: np.ndarray, target: np.ndarray):
    return np.sum(predicted == target) / predicted.shape[0]


def f1_score(recall, precision):
    epsilon = 1e-8
    return 2 * precision * recall / (precision + recall + epsilon)


def metrics(predicted: np.ndarray, target: np.ndarray):
    epsilon = 1e-8
    target_0 = np.sum(target == 0)
    target_1 = target.shape[0] - target_0
    predicted_0 = np.sum(predicted == 0)
    predicted_1 = predicted.shape[0] - predicted_0
    true_0 = np.sum(predicted + target == 0)
    true_1 = np.sum(predicted + target == 2)
    recall_0 = true_0 / (target_0 + epsilon)
    precision_0 = true_0 / (predicted_0 + epsilon)
    recall_1 = true_1 / (target_1 + epsilon)
    precision_1 = true_1 / (predicted_1 + epsilon)
    f1_0 = f1_score(recall_0, precision_0)
    f1_1 = f1_score(recall_1, precision_1)

    return recall_0, precision_0, f1_0, recall_1, precision_1, f1_1


class SPTM:
    def __init__(
            self, r_network, init_cluster, init_observation_numpy,
            device, similarity_threshold, batch_size, n_prefixes, depth_limit):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.r_network = r_network
        self.embedding_network = r_network.embedding_network
        self.similarity_network = r_network.similarity_network

        self.node2cluster = [init_cluster]
        self.node2observation = [init_observation_numpy]
        self.node2embedding = self.to_embeddings(init_observation_numpy[np.newaxis])

        self.graph = nx.DiGraph()
        self.graph.add_node(0, visits=1, starts=0, ext_return=0, prefix_length=0, trajectory_length=0)

        self.prefix_graph = nx.MultiDiGraph()
        self.n_prefixes = n_prefixes
        self.prefix_graph.add_node(0)
        self.similarity_threshold = similarity_threshold
        self.max_similarity = -math.inf
        self.min_similarity = math.inf

        self.depth_limit = depth_limit
        self.sampling_history = collections.deque(maxlen=100)

    def to_embeddings(self, observations_numpy):
        embeddings = []
        self.embedding_network.eval()
        for i in range(0, len(observations_numpy), self.batch_size):
            observations_tensor = torch.tensor(observations_numpy[i:i + self.batch_size], device=self.device)
            with torch.no_grad():
                embeddings.append(self.embedding_network(observations_tensor))

        return torch.cat(embeddings, dim=0)

    def update_embeddings(self):
        self.node2embedding = self.to_embeddings(np.asarray(self.node2observation))

    def get_neighbours(self, node):
        neighbours = {node}
        for _, v in itertools.chain(nx.bfs_edges(self.graph, source=node, depth_limit=self.depth_limit),
                                    nx.bfs_edges(self.graph, source=node, depth_limit=self.depth_limit, reverse=True)):
            neighbours.add(v)

        return list(neighbours)

    def find_similar(self, embedding_tensor, cluster_embeddings_tensor):
        max_similarity = -1
        max_index = -1
        with torch.no_grad():
            embedding = embedding_tensor.expand(self.batch_size, *embedding_tensor.shape)
            for i in range(0, cluster_embeddings_tensor.shape[0], self.batch_size):
                cluster_embeddings = cluster_embeddings_tensor[i:i + self.batch_size]
                similarity, index = torch.max(self.similarity_network(embedding[:cluster_embeddings.size(0)], cluster_embeddings, logits=False)[:, 1], dim=0)
                similarity = similarity.item()
                if similarity >= self.similarity_threshold and similarity > max_similarity:
                    max_similarity = similarity
                    max_index = i + index.item()

                self.max_similarity = max(similarity, self.max_similarity)
                self.min_similarity = min(similarity, self.min_similarity)

        return max_index

    def find_node(self, previous_node, embedding_tensor, cluster_embeddings_tensor):
        node = -1
        if previous_node is not None and self.depth_limit > 0:
            neighbours = self.get_neighbours(previous_node)
            node = self.find_similar(embedding_tensor, cluster_embeddings_tensor[neighbours])
            if node >= 0:
                node = neighbours[node]

        if node < 0:
            node = self.find_similar(embedding_tensor, cluster_embeddings_tensor)

        if node >= 0:
            if previous_node is not None and node != previous_node:
                if self.graph.has_edge(previous_node, node):
                    self.graph.get_edge_data(previous_node, node)['visits'] += 1
                else:
                    self.graph.add_edge(previous_node, node, visits=1)

            return node

        node = self.node2embedding.size(0)
        self.node2embedding = torch.cat((self.node2embedding, embedding_tensor.unsqueeze(0)), dim=0)
        self.graph.add_node(node, visits=1, starts=0, ext_return=0, prefix_length=0, trajectory_length=0)
        if previous_node is not None:
            assert not self.graph.has_edge(previous_node, node)
            self.graph.add_edge(previous_node, node, visits=1)

        return node

    def add_trajectory(self, start_node, trajectory_cells, trajectory_observations_numpy, ext_return):
        self.r_network.eval()
        assert start_node is not None, 'start_node is None'

        self.graph.nodes[start_node]['trajectory_length'] += len(trajectory_cells)
        self.graph.nodes[start_node]['visits'] += 1
        self.graph.nodes[start_node]['starts'] += 1
        self.graph.nodes[start_node]['ext_return'] += ext_return

        if not trajectory_cells:
            return

        prefix_states = []
        parent = start_node

        node = start_node
        trajectory_embeddings_numpy = self.to_embeddings(trajectory_observations_numpy)
        cluster_embeddings_tensor = self.node2embedding
        for index, embedding_numpy in enumerate(trajectory_embeddings_numpy):
            node = self.find_node(node, embedding_numpy, cluster_embeddings_tensor)
            cell = trajectory_cells[index]
            is_new_node = node >= len(self.node2cluster)
            if is_new_node:
                self.node2cluster.append(cell)
                self.node2observation.append(trajectory_observations_numpy[index])
                cluster_embeddings_tensor = self.node2embedding
                self.prefix_graph.add_edge(node, parent, prefix=np.around(np.asarray(prefix_states) * 255).astype(np.uint8), primary=True)
                parent = node

                prefix_states = [trajectory_observations_numpy[index]]
            else:
                prefix_states.append(trajectory_observations_numpy[index])

            self.graph.nodes[node]['visits'] += 1
            self.graph.nodes[node]['prefix_length'] += index

        assert len(self.node2observation) == len(self.node2cluster)
        assert len(self.node2observation) == self.node2embedding.size(0)
        assert len(self.node2cluster) == self.graph.number_of_nodes()
        assert len(self.node2cluster) == self.prefix_graph.number_of_nodes()

    def least_frequent_nodes(self, size):
        weights = 1 / np.asarray([self.graph.nodes[node]['visits'] for node in range(len(self.node2cluster))])
        probability = weights / np.sum(weights)
        nodes = np.random.choice(a=len(self.node2cluster), size=size, p=probability)
        self.sampling_history.append(nodes)
        return nodes, probability[nodes]

    def random_nodes(self, size):
        nodes = np.random.choice(a=len(self.node2cluster), size=size)
        self.sampling_history.append(nodes)
        return nodes

    def get_memory_usage(self):
        MB = 1024 * 1024
        n_prefix_images = 0
        n_prefixes = 0
        prefix_images_MB = 0
        for i, cell in enumerate(self.node2cluster):
            for neighbour, out_edges in self.prefix_graph[i].items():
                for edge_id, data in out_edges.items():
                    n_prefixes += 1
                    prefix = data['prefix']
                    n_prefix_images += prefix.shape[0]
                    prefix_images_MB += prefix.nbytes

        prefix_images_MB /= MB

        n_clusters = len(self.node2cluster)
        cluster_observation_MB = self.node2observation[0].nbytes * n_clusters / MB
        emulator_states_MB = len(self.node2cluster[0].state.emulator_state) * n_clusters / MB
        state_observations_MB = self.node2cluster[0].state.observation.nbytes * n_clusters / MB

        return {
            'n_clusters': n_clusters,
            'n_prefixes': n_prefixes,
            'n_prefix_images': n_prefix_images,
            'emulator_states_size': emulator_states_MB,
            'cluster_observations_size': cluster_observation_MB,
            'state_observations_size': state_observations_MB,
            'prefix_images_size': prefix_images_MB
        }

    @staticmethod
    def merge_nodes(digraph, v, w):
        for attr, value in digraph.nodes[w].items():
            digraph.nodes[v][attr] += value

        for _, u, data in digraph.out_edges(w, data=True):
            if u == v:
                continue

            if u in digraph[v]:
                for attr, value in data.items():
                    digraph.edges[(v, u)][attr] += value
            else:
                digraph.add_edge(v, u, **data)

        for u, _, data in digraph.in_edges(w, data=True):
            if u == v:
                continue

            if v in digraph[u]:
                for attr, value in data.items():
                    digraph.edges[(u, v)][attr] += value
            else:
                digraph.add_edge(u, v, **data)

        digraph.remove_node(w)

    def merge_prefix_graph(self, v, w):
        for neighbour, out_edges in self.prefix_graph[w].items():
            for edge_id, data in out_edges.items():
                data['primary'] = False

        nx.contracted_nodes(self.prefix_graph, v, w, copy=False)
        if 'contraction' in self.prefix_graph.nodes[v]:
            del self.prefix_graph.nodes[v]['contraction']

        not_primary_edges = []
        for neighbour, out_edges in self.prefix_graph[v].items():
            for edge_id, data in out_edges.items():
                if not data['primary']:
                    not_primary_edges.append((neighbour, edge_id))

        if len(not_primary_edges) > self.n_prefixes - 1:
            for neighbour, edge_id in random.sample(not_primary_edges, len(not_primary_edges) - self.n_prefixes + 1):
                self.prefix_graph.remove_edge(v, neighbour, edge_id)

    def multi_merge(self):
        self.r_network.eval()

        node2observation = self.node2observation
        node2cell = self.node2cluster
        graph = self.graph
        embeddings = self.node2embedding

        self.node2observation = [node2observation[0]]
        self.node2cluster = [node2cell[0]]
        self.node2embedding = embeddings[:1]
        self.graph = nx.DiGraph()
        self.graph.add_node(0)

        new2old = {0: 0}
        old2new = {0: 0}
        actual_cluster_embeddings_tensor = self.node2embedding
        for old_node_id in range(1, len(node2cell)):
            embedding = embeddings[old_node_id]
            new_node_id = self.find_node(None, embedding, actual_cluster_embeddings_tensor)
            old2new[old_node_id] = new_node_id
            if new_node_id not in new2old:
                new2old[new_node_id] = old_node_id
            is_new_node = new_node_id >= len(self.node2cluster)
            if is_new_node:
                self.node2cluster.append(node2cell[old_node_id])
                self.node2observation.append(node2observation[old_node_id])

                actual_cluster_embeddings_tensor = self.node2embedding
            else:
                self.merge_prefix_graph(new2old[new_node_id], old_node_id)
                self.merge_nodes(graph, new2old[new_node_id], old_node_id)

        sampling_history = self.sampling_history
        self.sampling_history = collections.deque(maxlen=100)
        for nodes in sampling_history:
            nodes = [old2new[node] for node in nodes]
            self.sampling_history.append(nodes)

        n_nodes = len(self.node2cluster)
        assert self.prefix_graph.number_of_nodes() == n_nodes
        self.prefix_graph = nx.relabel.relabel_nodes(self.prefix_graph, mapping=old2new)

        assert graph.number_of_nodes() == n_nodes
        self.graph = nx.relabel.relabel_nodes(graph, mapping=old2new)

        assert n_nodes == len(self.node2observation)
        assert n_nodes == self.graph.number_of_nodes()
        assert n_nodes == self.prefix_graph.number_of_nodes()
        assert n_nodes == self.node2embedding.size(0)

        gc.collect()

    def build_prefix_trajectory(self, node, target=0):
        visited_nodes = set()
        shortest_paths = nx.single_target_shortest_path(self.prefix_graph, target)
        u = node
        path = [u]
        prefixes = []
        while u != target:
            u, data = self._next_node(u, visited_nodes, shortest_paths)
            path.append(u)
            prefixes.append(data['prefix'])

        return path, prefixes

    def _next_node(self, node, visited_nodes: set, shortest_paths: Dict[int, list]):
        if node in visited_nodes:
            assert len(shortest_paths[node]) > 1, f'Couldn\'t find shortest path for node {node}'
            shortest_path_neighbour = shortest_paths[node][1]

            return shortest_path_neighbour, random.choice(list(self.prefix_graph[node][shortest_path_neighbour].values()))

        visited_nodes.add(node)
        available_edges = []
        for neighbour, out_edges in self.prefix_graph[node].items():
            for edge_id, data in out_edges.items():
                available_edges.append((node, neighbour, edge_id, data))

        assert len(available_edges) > 0, f'Couldn\'t find output edges for node {node}'

        edge = random.choice(available_edges)
        return edge[1], edge[3]


def compute_next_buffer_position(buffer_position,
                                 positive_example_candidate,
                                 max_action_distance,
                                 mode):
    """Computes the buffer position for the next training example."""
    if mode == 'v3_affect_num_training_examples_overlap':
        # This version was initially not intended (changing max_action_distance
        # affects the number of training examples, and we can also get overlap
        # across generated examples), but we have it because it produces good
        # results (reward at ~40 according to raveman@ on 2018-10-03).
        # R-nets /cns/vz-d/home/dune/episodic_curiosity/raphaelm_train_r_mad2_4 were
        # generated with this version (the flag was set
        # v1_affect_num_training_examples, but it referred to a "buggy" version of
        # v1 that is reproduced here with that v3).
        return buffer_position + random.randint(1, max_action_distance) + 1
    if mode == 'v1_affect_num_training_examples':
        return positive_example_candidate + 1
    if mode == 'v2_fixed_num_training_examples':
        # Produces the ablation study in the paper submitted to ICLR'19
        # (https://openreview.net/forum?id=SkeK3s0qKQ), section S4.1.
        return buffer_position + random.randint(1, 5) + 1


def generate_positive_example(buffer_position,
                              next_buffer_position):
    """Generates a close enough pair of states."""
    first = buffer_position
    second = next_buffer_position

    # Make R-network symmetric.
    # Works for DMLab (navigation task), the symmetry assumption might not be
    # valid for all the environments.
    if random.random() < 0.5:
        first, second = second, first
    return first, second


def generate_negative_example(buffer_position,
                              len_episode_buffer,
                              max_action_distance):
    """Generates a far enough pair of states."""
    assert buffer_position < len_episode_buffer
    # Defines the interval that must be excluded from the sampling.
    time_interval = (NEGATIVE_SAMPLE_MULTIPLIER * max_action_distance)
    min_index = max(buffer_position - time_interval, 0)
    max_index = min(buffer_position + time_interval + 1, len_episode_buffer)

    # Randomly select an index outside the interval.
    effective_length = len_episode_buffer - (max_index - min_index)
    range_max = effective_length - 1
    if range_max <= 0:
        return buffer_position, None
    index = random.randint(0, range_max)
    if index >= min_index:
        index = max_index + (index - min_index)
    return buffer_position, index


def create_training_data(trajectory_by_cells):
    """Samples intervals and forms pairs."""
    max_action_distance = 5
    mode = 'v2_fixed_num_training_examples'
    x1 = []
    x2 = []
    labels = []
    buffer_position = -1
    positive_labels = 0
    while True:
        positive_example_candidate = (
                buffer_position + random.randint(1, max_action_distance))
        next_buffer_position = compute_next_buffer_position(
            buffer_position, positive_example_candidate,
            max_action_distance, mode)

        if (next_buffer_position >= len(trajectory_by_cells) or
                positive_example_candidate >= len(trajectory_by_cells)):
            break

        label = random.randint(0, 1)

        if label:
            first, second = generate_positive_example(buffer_position,
                                                      positive_example_candidate)
        else:
            first, second = generate_negative_example(buffer_position,
                                                      len(trajectory_by_cells),
                                                      max_action_distance)

        if first is None or second is None:
            break

        positive_labels += label

        x1.append(trajectory_by_cells[first])
        x2.append(trajectory_by_cells[second])
        labels.append(label)
        buffer_position = next_buffer_position

    return np.asarray(x1, dtype=np.float32), np.asarray(x2, dtype=np.float32), np.asarray(labels, dtype=np.int64)
