import warnings
import networkx as nx
import numpy as np
import scipy.linalg as lg
import torch
import scipy.sparse as sp

from .node2vec import Node2Vec
from .sdne import DynGEM, DynGEMLoss, DynGEMBatchGenerator
from typing import Union
from ..utils import agent_idx


class Embedding(object):
    """
    Abstract class for graph node embeddings.
    """

    def __init__(self, dim, **kwargs):
        self.dim = dim

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], **kwargs):
        raise NotImplementedError()

    def transform(self, nodes):
        raise NotImplementedError()


class HOPEEmbedding(Embedding):
    def __init__(self, dim, proximity='katz', beta=0.01, **kwargs):
        if dim % 2 != 0:
            dim -= dim % 2
            print('HOPE supports only even embedding dimensions; falling back to {}'.format(dim))

        if proximity not in ('katz', 'common-neighbors', 'adamic-adar'):
            raise Exception('Unsupported proximity measure: ' + proximity)

        super().__init__(dim, **kwargs)
        self.proximity = proximity
        self.beta = beta
        self._W = None

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], weight='weight'):
        if type(graph) == nx.DiGraph:
            graph = nx.relabel_nodes(graph, agent_idx)
            A = nx.to_numpy_matrix(graph, nodelist=sorted(graph.nodes), weight=weight)
            n = graph.number_of_nodes()
        else:
            A = np.mat(graph)
            n = A.shape[0]

        if self.proximity == 'katz':
            M_g = np.eye(n) - self.beta * A
            M_l = self.beta * A
        elif self.proximity == 'common-neighbors':
            M_g = np.eye(n)
            M_l = A * A
        elif self.proximity == 'adamic-adar':
            M_g = np.eye(n)
            D = np.mat(np.diag([1 / (np.sum(A[:, i]) + np.sum(A[i, :])) for i in range(n)]))
            M_l = A * D * A

        S = np.dot(np.linalg.inv(M_g), M_l)

        # (Changed by Igor):
        # Added v0 parameter, the "starting vector for iteration".
        # Otherwise, the operation behaves nondeterministically, and as a result
        # different nodes may learn different embeddings. I am not speaking about
        # minor floating point errors, the problem was worse.

        # u, s, vt = sp.linalg.svds(S, k=self.dim // 2)
        u, s, vt = sp.linalg.svds(S, k=self.dim // 2, v0=np.ones(A.shape[0]))

        X1 = np.dot(u, np.diag(np.sqrt(s)))
        X2 = np.dot(vt.T, np.diag(np.sqrt(s)))
        self._W = np.concatenate((X1, X2), axis=1)

    def transform(self, idx):
        return self._W[idx]


class LaplacianEigenmap(Embedding):
    def __init__(self, dim, renormalize_weights=True, weight_transform='heat',
                 temp=1.0, **kwargs):
        super().__init__(dim, **kwargs)
        self.renormalize_weights = renormalize_weights
        self.weight_transform = weight_transform
        self.temp = temp
        self._X = None

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], weight='weight'):
        if type(graph) == np.ndarray:
            graph = nx.from_numpy_array(graph, create_using=nx.DiGraph)
            weight = 'weight'

        graph = nx.relabel_nodes(graph.to_undirected(), agent_idx)

        if weight is not None:
            if self.renormalize_weights:
                sum_w = sum([ps[weight] for _, _, ps in graph.edges(data=True)])
                avg_w = sum_w / len(graph.edges())
                for u, v, ps in graph.edges(data=True):
                    graph[u][v][weight] /= avg_w

            if self.weight_transform == 'inv':
                for u, v, ps in graph.edges(data=True):
                    graph[u][v][weight] = 1 / ps[weight]

            elif self.weight_transform == 'heat':
                for u, v, ps in graph.edges(data=True):
                    w = ps[weight]
                    graph[u][v][weight] = np.exp(-w * w)

        A = nx.to_scipy_sparse_matrix(graph, nodelist=sorted(graph.nodes),
                                      weight=weight, format='csr', dtype=np.float32)

        n, m = A.shape
        diags = A.sum(axis=1)
        D = sp.spdiags(diags.flatten(), [0], m, n, format='csr')
        L = D - A

        # (Changed by Igor):
        # Added v0 parameter, the "starting vector for iteration".
        # Otherwise, the operation behaves nondeterministically, and as a result
        # different nodes may learn different embeddings. I am not speaking about
        # minor floating point errors, the problem was worse.

        # values, vectors = sp.linalg.eigsh(L, k=self.dim + 1, M=D, which='SM')
        values, vectors = sp.linalg.eigsh(L, k=self.dim + 1, M=D, which='SM', v0=np.ones(A.shape[0]))

        # End (Changed by Igor)

        self._X = vectors[:, 1:]

        if weight is not None and self.renormalize_weights:
            self._X *= avg_w
        # print(self._X.flatten()[:3])

    def transform(self, idx):
        return self._X[idx]

    def transform_all(self):
        return self._X


class Node2VecWrapper(Embedding):
    def __init__(self, dim, walk_length=20, context_size=10,
                 walks_per_node=1, p=1, q=1, num_negative_samples=1,
                 num_nodes=None, sparse=False, **kwargs):
        super().__init__(dim, **kwargs)
        self.model = None
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.num_nodes = num_nodes
        self.sparse = sparse
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit(self, graph: Union[nx.DiGraph, np.ndarray], batch_size=128, num_workers=1, shuffle=True, lr=0.01,
            use_pretrained_laplacian=True, weight='weight', **kwargs):
        if type(graph) == np.ndarray:
            graph = nx.from_numpy_array(graph, create_using=nx.DiGraph)
        graph = nx.relabel_nodes(graph.to_undirected(), agent_idx)
        edge_index = torch.from_numpy(np.asarray(graph.edges()).T).to(self.device)
        pretrained_emb = None
        # if use_pretrained_laplacian:
        #     laplacian_emb = LaplacianEigenmap(self.dim)
        #     laplacian_emb.fit(graph, weight=None)
        #     pretrained_emb = torch.from_numpy(laplacian_emb.transform_all())

        self.model = Node2Vec(edge_index, self.dim, self.walk_length, self.context_size,
                              self.walks_per_node, self.p, self.q,
                              self.num_negative_samples, self.num_nodes, self.sparse, pretrained_emb).to(self.device)
        loader = self.model.loader(batch_size=batch_size, shuffle=shuffle, num_workers=0)
        optimizer = torch.optim.Adam(list(self.model.parameters()), lr=lr)
        self.model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    def transform(self, idx):
        return self.model.embedding.weight[idx].detach().numpy()


class SDNE(Embedding):
    def __init__(self, dim, **kwargs):
        super().__init__(dim, **kwargs)
        self.dim = dim
        self.model = None
        self.graph = None

    def fit(self, graph: Union[nx.DiGraph, np.ndarray],
            n_units=None, epoch=20, batch_size = 64,
            lr=0.01, weight_decay=0, alpha=1e-5, beta=10, nu1=1e-4, nu2=1e-4, **kwargs):
        if n_units is None:
            n_units = [15]
        if type(graph) == np.ndarray:
            graph = nx.from_numpy_array(graph, create_using=nx.DiGraph)
        self.graph = nx.adjacency_matrix(nx.relabel_nodes(graph.to_undirected(), agent_idx))
        self.model = DynGEM(self.graph.shape[0], self.dim, n_units=n_units)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer.zero_grad()
        loss_model = DynGEMLoss(alpha, beta, nu1, nu2)
        batch_generator = DynGEMBatchGenerator(batch_size, beta)
        for i in range(epoch):
            generator = batch_generator.generate(self.graph)
            loss_input_list = self._get_model_res(generator)
            loss = loss_model(self.model, loss_input_list)
            loss.backward()
            optimizer.step()
            self.model.zero_grad()
            # print("epoch", i + 1, ', loss:', loss.item())

    def propagate(self, graph, gradient, lr=0.01, weight_decay=0):
        if type(graph) == np.ndarray:
            graph = nx.from_numpy_array(graph, create_using=nx.DiGraph)
        self.graph = nx.adjacency_matrix(nx.relabel_nodes(graph.to_undirected(), agent_idx))
        optimizer = torch.optim.Adam(self.model.encoder.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer.zero_grad()
        output = self.model.encoder(torch.from_numpy(self.graph.toarray()).float())
        output.backward(gradient=gradient)
        optimizer.step()
        self.model.zero_grad()


    def transform(self, idx):
        row = torch.from_numpy(self.graph[idx].toarray()).float()
        return self.model(row)[0].detach().numpy()

    def _get_model_res(self, generator):
        [xi_batch, xj_batch], [yi_batch, yj_batch, value_batch] = next(generator)
        hx_i, xi_pred = self.model(xi_batch)
        hx_j, xj_pred = self.model(xj_batch)
        return [xi_pred, xi_batch, yi_batch, xj_pred, xj_batch, yj_batch, hx_i, hx_j, value_batch]


emb_classes = {
    'hope': HOPEEmbedding,
    'lap': LaplacianEigenmap,
    'node2vec': Node2VecWrapper,
    'sdne': SDNE
}


def get_embedding(alg: str, **kwargs):
    try:
        return emb_classes[alg](**kwargs)
    except KeyError:
        raise Exception('Unsupported embedding algorithm: ' + alg)
