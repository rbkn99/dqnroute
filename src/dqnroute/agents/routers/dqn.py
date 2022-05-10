import random
import math
import logging
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import pandas as pd
import pprint
import os

from typing import List, Tuple, Dict, Union
from ..base import *
from .link_state import *
from ...constants import DQNROUTE_LOGGER
from ...messages import *
from ...memory import *
from ...utils import *
from ...networks import *

logger = logging.getLogger(DQNROUTE_LOGGER)

class SharedBrainStorage:
    INSTANCE = None
    PROCESSED_NODES = 0
    
    @staticmethod
    def load(brain_loader: Callable[[], QNetwork], no_nodes: int) -> QNetwork:
        if SharedBrainStorage.INSTANCE is None:
            SharedBrainStorage.INSTANCE = brain_loader()
        SharedBrainStorage.PROCESSED_NODES += 1
        #print(f"Brain initialization: {SharedBrainStorage.PROCESSED_NODES} / {no_nodes} agents")
        result = SharedBrainStorage.INSTANCE
        if SharedBrainStorage.PROCESSED_NODES == no_nodes:
            # all nodes have been processes
            # prepare this class for possible reuse
            SharedBrainStorage.INSTANCE = None
            SharedBrainStorage.PROCESSED_NODES = 0
        return result
        

class DQNRouter(LinkStateRouter, RewardAgent):
    """
    A router which implements the DQN-routing algorithm.
    """
    def __init__(self, batch_size: int, mem_capacity: int, nodes: List[AgentId],
                 optimizer='rmsprop', brain=None, random_init=False, max_act_time=None,
                 additional_inputs=[], softmax_temperature: float = 1.5,
                 probability_smoothing: float = 0.0, load_filename: str = None,
                 use_single_neural_network: bool = False,
                 use_reinforce: bool = True,
                 use_combined_model: bool = False,
                 **kwargs):
        """
        Parameters added by Igor:
        :param softmax_temperature: larger temperature means larger entropy of routing decisions.
        :param probability_smoothing (from 0.0 to 1.0): if greater than 0, then routing probabilities will
            be separated from zero.
        :param load_filename: filename to load the neural network. If None, a new network will be created.
        :param use_single_neural_network: all routers will reference the same instance of the neural network.
            In particular, this very network will be influeced by training steps in all nodes.
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.memory = Memory(mem_capacity)
        self.additional_inputs = additional_inputs
        self.nodes = nodes
        self.max_act_time = max_act_time
        
        # changed by Igor: custom temperatures for softmax:
        self.min_temp = softmax_temperature
        # added by Igor: probability smoothing (0 means no smoothing):
        self.probability_smoothing = probability_smoothing

        self.use_reinforce = use_reinforce
        self.use_combined_model = use_combined_model

        # changed by Igor: brain loading process
        def load_brain():
            b = brain
            if b is None:
                b = self._makeBrain(additional_inputs=additional_inputs, **kwargs)
                if random_init:
                    b.init_xavier()
                else:
                    if load_filename is not None:
                        b.change_label(load_filename)
                    b.restore()
            return b
        if use_single_neural_network:
            self.brain = SharedBrainStorage.load(load_brain, len(nodes))
        else:
            self.brain = load_brain()
        self.use_single_neural_network = use_single_neural_network

        self.optimizer = get_optimizer(optimizer)(self.brain.parameters())
        self.loss_func = nn.MSELoss()

    def route(self, sender: AgentId, pkg: Package, allowed_nbrs: List[AgentId]) -> Tuple[AgentId, List[Message]]:
        if self.max_act_time is not None and self.env.time() > self.max_act_time:
            return super().route(sender, pkg, allowed_nbrs)
        else:
            to, estimate, saved_state, agents = self._act(pkg, allowed_nbrs)
            reward = self.registerResentPkg(pkg, estimate, to, saved_state, agents)
            return to, [OutMessage(self.id, sender, reward)] if sender[0] != 'world' else []

    def handleMsgFrom(self, sender: AgentId, msg: Message) -> List[Message]:
        if isinstance(msg, RewardMsg):
            action, Q_new, prev_state, agents = self.receiveReward(msg)
            self.memory.add((prev_state, action[1], -Q_new, agents))

            if self.use_reinforce:
                self._replay()
            return []
        else:
            return super().handleMsgFrom(sender, msg)

    def _makeBrain(self, additional_inputs=[], **kwargs):
        return QNetwork(len(self.nodes), additional_inputs=additional_inputs, one_out=False, **kwargs)

    def _act(self, pkg: Package, allowed_nbrs: List[AgentId]):
        state = self._getNNState(pkg, allowed_nbrs)
        prediction = self._predict(state)[0]
        distr = softmax(prediction, self.min_temp)
        estimate = -np.dot(prediction, distr)

        to = -1
        while ('router', to) not in allowed_nbrs:
            to = sample_distr(distr)

        return ('router', to), estimate, state, (self.id[1], pkg.dst[1], to[1])

    def _predict(self, x):
        self.brain.eval()
        return self.brain(*map(torch.from_numpy, x)).clone().detach().numpy()

    def _train(self, x, y):
        self.brain.train()
        self.optimizer.zero_grad()
        output = self.brain(*map(torch.from_numpy, x))
        loss = self.loss_func(output, torch.from_numpy(y))
        loss.backward()
        self.optimizer.step()
        return float(loss)

    def _getAddInput(self, tag, *args, **kwargs):
        if tag == 'amatrix':
            amatrix = nx.convert_matrix.to_numpy_array(
                self.network, nodelist=self.nodes, weight=self.edge_weight,
                dtype=np.float32)
            gstate = np.ravel(amatrix)
            return gstate
        else:
            raise Exception('Unknown additional input: ' + tag)

    def _getNNState(self, pkg: Package, nbrs: List[AgentId]):
        n = len(self.nodes)
        addr = np.array(self.id[1])
        dst = np.array(pkg.dst[1])

        neighbours = np.array(
            list(map(lambda v: v in nbrs, self.nodes)),
            dtype=np.float32)
        input = [addr, dst, neighbours]

        for inp in self.additional_inputs:
            tag = inp['tag']
            add_inp = self._getAddInput(tag)
            if tag == 'amatrix':
                add_inp[add_inp > 0] = 1
            input.append(add_inp)

        return tuple(input)

    def _sampleMemStacked(self):
        """
        Samples a batch of episodes from memory and stacks
        states, actions and values from a batch together.
        """
        i_batch = self.memory.sample(self.batch_size)
        batch = [b[1] for b in i_batch]

        states = stack_batch([l[0] for l in batch])
        actions = [l[1] for l in batch]
        values = [l[2] for l in batch]
        agents = [l[3] for l in batch]

        return states, actions, values, agents

    def _replay(self):
        """
        Fetches a batch of samples from the memory and fits against them.
        """
        states, actions, values, _ = self._sampleMemStacked()
        preds = self._predict(states)

        for i in range(self.batch_size):
            a = actions[i]
            preds[i][a] = values[i]

        self._train(states, preds)


class DQNRouterOO(DQNRouter):
    """
    Variant of DQN router which uses Q-network with scalar output.
    """
    def _makeBrain(self, additional_inputs=[], **kwargs):
        return QNetwork(len(self.nodes), additional_inputs=additional_inputs,
                        one_out=True, **kwargs)

    def _act(self, pkg: Package, allowed_nbrs: List[AgentId]):
        state = self._getNNState(pkg, allowed_nbrs)
        prediction = self._predict(state).flatten()
        distr = softmax(prediction, self.min_temp)
        
        # Igor: probability smoothing
        distr = (1 - self.probability_smoothing) * distr + self.probability_smoothing / len(distr)
        
        to_idx = sample_distr(distr)
        estimate = -np.dot(prediction, distr)

        saved_state = [s[to_idx] for s in state]
        to = allowed_nbrs[to_idx]
        return to, estimate, saved_state, (self.id[1], pkg.dst[1], to[1])

    def _nodeRepr(self, node):
        return np.array(node)

    def _getAddInput(self, tag, nbr):
        return super()._getAddInput(tag)

    def _getNNState(self, pkg: Package, nbrs: List[AgentId]):
        n = len(self.nodes)
        addr = self._nodeRepr(self.id[1])
        dst = self._nodeRepr(pkg.dst[1])

        get_add_inputs = lambda nbr: [self._getAddInput(inp['tag'], nbr)
                                      for inp in self.additional_inputs]

        input = [[addr, dst, self._nodeRepr(v[1])] + get_add_inputs(v) for v in nbrs]
        return stack_batch(input)

    def _replay(self):
        states, _, values, _ = self._sampleMemStacked()
        self._train(states, np.expand_dims(np.array(values, dtype=np.float32), axis=0))


class DQNRouterEmb(DQNRouterOO):
    """
    Variant of DQNRouter which uses graph embeddings instead of
    one-hot label encodings.
    """
    def __init__(self, embedding: Union[dict, Embedding], edges_num: int, **kwargs):
        # Those are used to only re-learn the embedding when the topology is changed
        self.prev_num_nodes = 0
        self.prev_num_edges = 0
        self.init_edges_num = edges_num
        self.network_initialized = False
        if 'embedding_load_filename' in kwargs and 'embedding_scope' in kwargs:
            self.embedding = load_object(kwargs['embedding_scope'], kwargs['embedding_load_filename'])
        elif type(embedding) == dict:
            self.embedding = get_embedding(**embedding)
        else:
            self.embedding = embedding
        super().__init__(**kwargs)

    def _makeBrain(self, additional_inputs=[], **kwargs):
        if not self.use_combined_model:
            return QNetwork(
                len(self.nodes), additional_inputs=additional_inputs,
                embedding_dim=self.embedding.dim, one_out=True, **kwargs
            )
        else:
            return CombinedNetwork(
                len(self.nodes), additional_inputs=additional_inputs,
                embedding_dim=self.embedding.dim, one_out=True, **kwargs
            )

    def _nodeRepr(self, node):
        return self.embedding.transform(node).astype(np.float32)

    def networkStateChanged(self):
        num_nodes = len(self.network.nodes)
        num_edges = len(self.network.edges)

        if not self.network_initialized and num_nodes == len(self.nodes) and num_edges == self.init_edges_num:
            self.network_initialized = True

        if self.network_initialized and (num_edges != self.prev_num_edges or num_nodes != self.prev_num_nodes):
            self.prev_num_nodes = num_nodes
            self.prev_num_edges = num_edges
            self.embedding.fit(self.network, weight=self.edge_weight)
            # self.log(pprint.pformat(self.embedding._X), force=self.id[1] == 0)


class EmbGlobalInstance(object):
    def __init__(self, embedding, emb_batch_size, use_single=True):
        self.embedding = embedding
        self.emb_batch_size = emb_batch_size
        self.adj_batch = np.zeros((embedding.nodes_num, self.embedding.dim))
        self.replay_counter = 0
        self.use_single = use_single
        self.nodes_len = 0
        self.network = None
        self.network_initialized = False

    def fit_emb(self, network, state_changed=False):
        if self.network_initialized and self.use_single and not state_changed:
            return
        # if self.network_initialized:
        #     return
        # self.network_initialized = True
        # self.adj_batch = np.zeros((len(network.nodes), self.embedding.dim))
        # self.replay_counter = 0
        print('fit qemb')
        self.network_initialized = True
        self.network = network
        self.nodes_len = len(network.nodes)
        self.embedding.fit(network, epoch=50, encoder_lr=0.0001, decoder_lr=0.001)

    def update(self, agents, batch_size, addr_grad, dst_grad, nbr_grad):
        self.adj_batch[agents[0][0], :] += addr_grad.detach().numpy()[0]
        self.adj_batch[agents[0][1], :] += dst_grad.detach().numpy()[0]
        self.adj_batch[agents[0][2], :] += nbr_grad.detach().numpy()[0]
        self.replay_counter += 1
        if self.replay_counter >= self.emb_batch_size:
            # print('propagate gradient')
            assert isinstance(self.embedding, QSDNE), 'QEmb currently supports SDNE only!'
            self.replay_counter = 0
            self.embedding.propagate(self.network, torch.from_numpy(self.adj_batch))
            self.adj_batch = np.zeros((self.nodes_len, self.embedding.dim))


# embInstance: EmbGlobalInstance = None


class DQNRouterQEmb(DQNRouterEmb):
    def __init__(self, embedding: Union[dict, Embedding], edges_num: int, emb_batch_size=1, **kwargs):
        super().__init__(embedding, edges_num, **kwargs)
        # global embInstance
        if isinstance(self.embedding, QSDNE):
            self.embInstance = EmbGlobalInstance(self.embedding, emb_batch_size)

    def networkStateChanged(self):
        if not isinstance(self.embedding, QSDNE):
            return super().networkStateChanged()
        num_nodes = len(self.network.nodes)
        num_edges = len(self.network.edges)
        if not self.network_initialized and num_nodes == len(self.nodes) and num_edges == self.init_edges_num:
            self.network_initialized = True
            # embInstance.fit_emb(self.network)
        if self.network_initialized and (num_edges != self.prev_num_edges or num_nodes != self.prev_num_nodes):
            print('network changed, fit emb')
            self.prev_num_nodes = num_nodes
            self.prev_num_edges = num_edges
            self.embInstance.fit_emb(self.network, state_changed=True)

    def _train(self, x, y):
        if not isinstance(self.embedding, QSDNE):
            return super()._train(x, y)
        self.brain.train()
        self.optimizer.zero_grad()
        addr, dst = torch.from_numpy(x[0]), torch.from_numpy(x[1])
        nbrs = list(map(torch.from_numpy, x[2:]))
        addr.requires_grad = True
        dst.requires_grad = True
        nbrs[0].requires_grad = True
        output = self.brain(addr, dst, *nbrs)
        loss = self.loss_func(output, torch.from_numpy(y))
        loss.backward()
        self.optimizer.step()
        return float(loss), addr.grad, dst.grad, nbrs[0].grad

    def _replay(self):
        if not isinstance(self.embedding, QSDNE):
            return super()._replay()
        states, _, values, agents = self._sampleMemStacked()
        _, addr_grad, dst_grad, nbr_grad = self._train(states, np.expand_dims(np.array(values, dtype=np.float32), axis=0))
        self.embInstance.update(agents, self.batch_size, addr_grad, dst_grad, nbr_grad)

    def _nodeRepr(self, node):
        if not isinstance(self.embedding, QSDNE):
            return super()._nodeRepr(node)
        return self.embInstance.embedding.transform(node).astype(np.float32)


class DQNRouterNetwork(NetworkRewardAgent, DQNRouter):
    pass


class DQNRouterOONetwork(NetworkRewardAgent, DQNRouterOO):
    pass


class DQNRouterEmbNetwork(NetworkRewardAgent, DQNRouterEmb):
    pass


class DQNRouterQEmbNetwork(NetworkRewardAgent, DQNRouterQEmb):
    pass

class ConveyorAddInputMixin:
    """
    Mixin which adds conveyor-specific additional NN inputs support
    """
    def _getAddInput(self, tag, nbr=None):
        if tag == 'work_status':
            return np.array(
                list(map(lambda n: self.network.nodes[n].get('works', False), self.nodes)),
                dtype=np.float32)
        if tag == 'working':
            nbr_works = 1 if self.network.nodes[nbr].get('works', False) else 0
            return np.array(nbr_works, dtype=np.float32)
        else:
            return super()._getAddInput(tag, nbr)


class DQNRouterConveyor(LSConveyorMixin, ConveyorRewardAgent, ConveyorAddInputMixin, DQNRouter):
    pass

class DQNRouterOOConveyor(LSConveyorMixin, ConveyorRewardAgent, ConveyorAddInputMixin, DQNRouterOO):
    pass

class DQNRouterEmbConveyor(LSConveyorMixin, ConveyorRewardAgent, ConveyorAddInputMixin, DQNRouterEmb):
    pass

