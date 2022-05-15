# coding: utf-8
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class MLP(nn.Module):
    input_dim: int
    output_dim: int
    bias: bool
    layer_list: nn.ModuleList
    layer_num: int

    def __init__(self, input_dim, output_dim, n_units, bias=True):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Linear(input_dim, n_units[0], bias=bias))

        layer_num = len(n_units)
        for i in range(1, layer_num):
            self.layer_list.append(nn.Linear(n_units[i - 1], n_units[i], bias=bias))
        self.layer_list.append(nn.Linear(n_units[-1], output_dim, bias=bias))
        self.layer_num = layer_num + 1

    def forward(self, x):
        for i in range(self.layer_num):
            x = F.relu(self.layer_list[i](x))
        return x


class RegularizationLoss(nn.Module):
    nu1: float
    nu2: float

    def __init__(self, nu1, nu2):
        super(RegularizationLoss, self).__init__()
        self.nu1 = nu1
        self.nu2 = nu2

    @staticmethod
    def get_weight(model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                # print('name: ', name)
                weight_list.append(weight)
        return weight_list

    def forward(self, model):
        loss = Variable(torch.FloatTensor([0.]), requires_grad=True).cuda() if torch.cuda.is_available() else Variable(torch.FloatTensor([0.]), requires_grad=True)
        # No L1 regularization and no L2 regularization
        if self.nu1 == 0. and self.nu2 == 0.:
            return loss
        # calculate L1-regularization loss and L2-regularization loss
        weight_list = self.get_weight(model)
        weight_num = len(weight_list)
        # print('weight num', weight_num)
        l1_reg_loss, l2_reg_loss = 0, 0
        for name, weight in weight_list:
            if self.nu1 > 0:
                l1_reg = torch.norm(weight, p=1)
                l1_reg_loss = l1_reg_loss + l1_reg
            if self.nu2 > 0:
                l2_reg = torch.norm(weight, p=2)
                l2_reg_loss = l2_reg_loss + l2_reg
        l1_loss = self.nu1 * l1_reg_loss / weight_num
        l2_loss = self.nu2 * l2_reg_loss / weight_num
        return l1_loss + l2_loss


class DynGEM(nn.Module):
    input_dim: int
    output_dim: int
    bias: bool
    method_name: str
    encoder: MLP
    decoder: MLP

    def __init__(self, input_dim, output_dim, n_units=None, bias=True, **kwargs):
        super(DynGEM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.encoder = MLP(input_dim, output_dim, n_units, bias=bias)
        self.decoder = MLP(output_dim, input_dim, n_units[::-1], bias=bias)

    def forward(self, x):
        hx = self.encoder(x)
        x_pred = self.decoder(hx)
        return hx, x_pred


class DynGEMLoss(nn.Module):
    alpha: float
    beta: float
    regularization: RegularizationLoss

    def __init__(self, alpha, beta, nu1, nu2):
        super(DynGEMLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.regularization = RegularizationLoss(nu1, nu2)

    def forward(self, model, input_list):
        assert len(input_list) == 9
        xi_pred, x_i, penalty_i = input_list[0], input_list[1], input_list[2]
        xj_pred, x_j, penalty_j = input_list[3], input_list[4], input_list[5]
        hx_i, hx_j, edge_weight = input_list[6], input_list[7], input_list[8]

        node_num = xj_pred.shape[1]

        eps = 1e-5

        old_penalty_i = penalty_i
        old_penalty_j = penalty_j
        penalty_i = torch.ones(x_i.shape, device=x_i.device)
        penalty_j = torch.ones(x_j.shape, device=x_j.device)
        penalty_i[(x_i != 0) & ((xi_pred < eps) & (xi_pred > -eps))] = self.beta
        penalty_j[(x_j != 0) & ((xj_pred < eps) & (xj_pred > -eps))] = self.beta

        xi_loss = torch.mean(torch.sum(torch.square((xi_pred - x_i) * penalty_i[:, 0:node_num]), dim=1) / old_penalty_i[:, node_num])
        xj_loss = torch.mean(torch.sum(torch.square((xj_pred - x_j) * penalty_j[:, 0:node_num]), dim=1) / old_penalty_j[:, node_num])

        hx_loss = torch.mean(torch.sum(torch.square(hx_i - hx_j), dim=1) * edge_weight)

        reconstruction_loss = xi_loss + xj_loss + self.alpha * hx_loss
        regularization_loss = self.regularization(model)
        return reconstruction_loss + regularization_loss


# Batch generator used for DynGEM
class DynGEMBatchGenerator:
    batch_size: int
    beta: float
    shuffle: bool
    has_cuda: bool

    def __init__(self, batch_size, beta, shuffle=True, has_cuda=False):
        self.batch_size = batch_size
        self.beta = beta
        self.shuffle = shuffle
        self.has_cuda = has_cuda

    def generate(self, graph):
        if isinstance(graph, list):
            assert len(graph) == 1
            graph = graph[0]
        rows, cols, values = sp.find(graph)
        element_num = rows.shape[0]
        element_indices = np.arange(element_num)
        batch_num = element_num // self.batch_size
        if element_num % self.batch_size != 0:
            batch_num += 1

        if self.shuffle:
            np.random.shuffle(element_indices)
        counter = 0
        while True:
            batch_indices = element_indices[self.batch_size * counter: min(element_num, self.batch_size * (counter + 1))]
            xi_batch = torch.from_numpy(graph[rows[batch_indices], :].toarray()).float()
            xi_batch = xi_batch.cuda() if self.has_cuda else xi_batch
            xj_batch = torch.tensor(graph[cols[batch_indices], :].toarray(), device=xi_batch.device).float()
            yi_batch = torch.ones(xi_batch.shape, device=xi_batch.device)  # penalty tensor for x_i
            yj_batch = torch.ones(xj_batch.shape, device=xi_batch.device)  # penalty tensor for x_j
            value_batch = torch.tensor(values[batch_indices], device=xi_batch.device).unsqueeze(1).float()  # [batch_size * 1]

            yi_batch[xi_batch != 0] = self.beta
            yj_batch[xj_batch != 0] = self.beta
            xi_degree_batch = torch.sum(xi_batch, dim=1).unsqueeze(1)  # [batch_size * 1]
            xj_degree_batch = torch.sum(xj_batch, dim=1).unsqueeze(1)  # [batch_size * 1]

            yi_batch = torch.cat((yi_batch, xi_degree_batch), dim=1)
            yj_batch = torch.cat((yj_batch, xj_degree_batch), dim=1)
            counter += 1
            yield [xi_batch, xj_batch], [yi_batch, yj_batch, value_batch]

            if counter == batch_num:
                if self.shuffle:
                    np.random.shuffle(element_indices)
                counter = 0


# Batch Predictor used for DynGEM
class DynGEMBatchPredictor:
    node_list: list
    node_num: int
    batch_size: int
    has_cuda: bool

    def __init__(self, node_list, batch_size, has_cuda=False):
        self.node_list = node_list
        self.node_num = len(node_list)
        self.batch_size = batch_size
        self.has_cuda = has_cuda

    def get_predict_res(self, graph, model, batch_indices, counter, embedding_mat, x_pred):
        if isinstance(graph, list):
            assert len(graph) == 1
            graph = graph[0]
        x_batch = torch.tensor(graph[batch_indices, :].toarray()).float()
        x_batch = x_batch.cuda() if self.has_cuda else x_batch
        hx_batch, x_pred_batch = model(x_batch)
        if counter:
            embedding_mat = torch.cat((embedding_mat, hx_batch), dim=0)
            x_pred = torch.cat((x_pred, x_pred_batch), dim=0)
        else:
            embedding_mat = hx_batch
            x_pred = x_pred_batch
        return embedding_mat, x_pred

    def predict(self, model, graph):
        counter = 0
        embedding_mat, x_pred = 0, 0
        batch_num = self.node_num // self.batch_size

        while counter < batch_num:
            batch_indices = range(self.batch_size * counter, self.batch_size * (counter + 1))
            embedding_mat, x_pred = self.get_predict_res(graph, model, batch_indices, counter, embedding_mat, x_pred)

            counter += 1
        # has a remaining batch
        if self.node_num % self.batch_size != 0:
            remain_indices = range(self.batch_size * counter, self.node_num)
            embedding_mat, x_pred = self.get_predict_res(graph, model, remain_indices, counter, embedding_mat, x_pred)
        return embedding_mat, x_pred
