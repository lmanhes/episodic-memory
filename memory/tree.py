import networkx as nx
import pickle
import random
import math


class TreeMemory(object):
    """
    Multi Directed Graph
    Store agent's interactions with the world
    Retrieve sequences of actions and 'goals'
    """

    def __init__(self, path, max_size=None, stability_start=50):
        """

        :param path: location of the tree memory
        :param max_size:
        """

        self.path = path
        self.max_size = max_size
        self.stability_start = stability_start
        self.graph = None

    def __len__(self):
        return self.graph.number_of_edges()

    def sample_trajectories(self, n=None, horizon=6):
        trajectories = []
        n_rand_nodes = min(50, len(self.graph))

        # TODO : add p based on importance of nodes (Q value)
        rand_source_nodes = random.sample(list(self.graph.nodes), n_rand_nodes)
        rand_target_nodes = random.sample(list(self.graph.nodes), n_rand_nodes)
        for node_source in rand_source_nodes:
            for node_target in rand_target_nodes:
                trajectories.extend(nx.all_simple_edge_paths(self.graph,
                                                             node_source,
                                                             node_target,
                                                             cutoff=horizon))

        if trajectories and n:
            n = min(len(trajectories), n)
            return random.sample(trajectories, n)
        return trajectories

    def _compute_centrality(self):
        centrality = nx.degree_centrality(self.graph)
        nx.set_node_attributes(self.graph, centrality, 'centrality')

    def get_nodes(self):
        return list(self.graph.nodes(data=True))

    def update(self, state_m1_id, action_m1, state_id):
        """
        Update memory tree with a new sequence
        if the sequence existed, update the weight to be the max value (1)

        :param state_m1_id: state id at t-1 (int)
        :param action_m1: action at t-1 (string)
        :param state_id: state id at t (int)
        :return: list of forgeted nodes
        """

        if self.graph is None:
            self.graph = nx.MultiDiGraph()

        if not self.graph.has_node(state_m1_id):
            self.graph.add_node(state_m1_id)

        # add new node state if it doesn't exists
        # and link it with previous state
        if not self.graph.has_node(state_id):
            self.graph.add_node(state_id)
            self.graph.add_edge(state_m1_id, state_id,
                                key=action_m1,
                                action=action_m1,
                                weight=1,
                                oldness=0,
                                stability=self.stability_start)
        else:
            # Update weight to 1 if there is a known action between the nodes
            # else create a new link between the two
            edge_data = self.graph.get_edge_data(state_m1_id, state_id)
            if edge_data and action_m1 in edge_data:
                self.graph[state_m1_id][state_id][action_m1]['weight'] = 1
                self.graph[state_m1_id][state_id][action_m1]['stability'] += 1
                self.graph[state_m1_id][state_id][action_m1]['oldness'] = 0
            else:
                self.graph.add_edge(state_m1_id, state_id,
                                    key=action_m1,
                                    action=action_m1,
                                    weight=1,
                                    oldness=0,
                                    stability=self.stability_start)

        forgeted = self.forget()

        # compute centrality for all nodes
        try:
            self._compute_centrality()
        except TypeError:
            pass

        return forgeted

    def _decay(self, oldness, stability):
        """
        Calculate the natural memory decay as :
            exp(-(oldness / stability))

        :param oldness: life time of an edge
        :param stability: measure of how well an edge is encoded
        :return: new weight value between 0 and 1
        """

        return math.exp(- (oldness / stability))

    def forget(self):
        """
        - Increment oldness of edges
        - Update weight based on exp(-(oldness / stability))
        - Remove every edges whose weight ~ 0
        - Remove every isolated nodes

        :return: list of isolated nodes
        """

        edges_to_remove = []
        for sm1, s, data in self.graph.edges.data():
            # increase oldness
            self.graph[sm1][s][data['action']]['oldness'] += 1

            # calculate and update weight due to memory decay
            oldness = self.graph[sm1][s][data['action']]['oldness']
            stability = self.graph[sm1][s][data['action']]['stability']
            weight_after_decay = self._decay(oldness, stability)
            self.graph[sm1][s][data['action']]['weight'] = weight_after_decay

            # remove edges when their weight is ~ 0
            if weight_after_decay < 0.01:
                edges_to_remove.append((sm1, s, data['action']))

        # remove edges
        self.graph.remove_edges_from(edges_to_remove)

        # remove nodes if they are isolated
        isolated = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolated)

        return isolated

    @classmethod
    def load(cls, path):
        memory_params = pickle.load(open(f'{path}.params', "rb"))
        tree_memory = cls(path=path, **memory_params)
        tree_memory.graph = nx.read_gpickle(f'{path}.graph')
        return tree_memory

    def save(self):
        memory_params = {'max_size': self.max_size,
                         'stability_start': self.stability_start}
        pickle.dump(memory_params, open(f'{self.path}.params', "wb"))
        nx.write_gpickle(self.graph, f'{self.path}.graph')