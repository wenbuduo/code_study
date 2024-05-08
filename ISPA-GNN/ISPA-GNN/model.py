import torch as t
from torch.nn import functional as F

import dgl as d
from dgl.nn import pytorch as dnn
import dgl.function as fn
from logger import *


# Node Info Scheme
# x   : one-hot
# in  : dim vector
# rev : message received
# out : for readout

class SubGraphConv(t.nn.Module):
    def __init__(self, dim, order=1, type='RT'):
        super(SubGraphConv, self).__init__()
        self.order = order  # 卷积的阶数
        self.dim = dim  # 特征维度
        self.type = type  # 边属性类型
        # 定义用户节点的线性变换层
        self.usermsg = t.nn.Linear(self.dim + 1, self.dim, bias=True)
        # 定义服务节点的线性变换层
        self.servmsg = t.nn.Linear(self.dim + 1, self.dim, bias=True)
        # 定义激活函数，使用LeakyReLU
        self.act = t.nn.LeakyReLU()

    # 定义用户到服务的消息传递函数
    def userToServ(self, edges):
        # 提取源节点（用户）的特征和边的属性
        userEmbs = edges.src['in']
        attrQoS = edges.data[self.type].reshape((-1, 1))
        # 将用户特征和边属性连接
        input = t.cat((userEmbs, attrQoS), -1)
        # 计算用户消息，并通过用户节点的线性变换层
        eventMsg = self.usermsg(input)
        return {'eventMsg': eventMsg}

    # 定义服务到用户的消息传递函数
    def servToUser(self, edges):
        # 提取源节点（服务）的特征和边的属性
        servEmbs = edges.src['in']
        attrQoS = edges.data[self.type].reshape((-1, 1))
        # 将服务特征和边属性连接
        input = t.cat((servEmbs, attrQoS), -1)
        # 计算服务消息，并通过服务节点的线性变换层
        eventMsg = self.servmsg(input)
        return {'eventMsg': eventMsg}

    # 定义前向传播函数
    def forward(self, bGraph):
        graph = bGraph  # 获取输入的图

        # 通过用户到服务的消息传递更新图
        graph.update_all(self.userToServ, fn.mean('eventMsg', 'out'), etype='us')
        # 通过服务到用户的消息传递更新图
        graph.update_all(self.servToUser, fn.mean('eventMsg', 'out'), etype='su')

        # 从图中读取用户节点的池化特征并进行激活
        user_readout = self.act(d.readout_nodes(graph, 'out', ntype='user', op='mean'))
        # 从图中读取服务节点的池化特征并进行激活
        serv_readout = self.act(d.readout_nodes(graph, 'out', ntype='serv', op='mean'))

        # 将用户和服务的池化特征连接起来
        user_serv_readout = t.cat((user_readout, serv_readout), -1)

        return user_serv_readout


# 信息传递
class MessagePassing(t.nn.Module):

    def __init__(self, dim, type='RT'):
        super(MessagePassing, self).__init__()
        self.dim = dim
        self.type = type
        # 定义用于消息传递的线性变换层
        self.W_us = t.nn.Linear(dim + 1, dim, bias=True)
        self.W_su = t.nn.Linear(dim + 1, dim, bias=True)
        # 定义激活函数
        self.act = t.nn.LeakyReLU()

    def userToServ(self, edges):
        userEmbs = edges.src['in']
        attrQoS = edges.data[self.type].reshape((-1, 1))
        input = t.cat((userEmbs, attrQoS), -1)
        # 根据边的 mask 属性，将部分输入置零
        input[edges.data['mask'] == 0] = 0
        eventMsg = self.W_us(input)
        return {'eventMsg': eventMsg}

    def servToUser(self, edges):
        servEmbs = edges.src['in']
        attrQoS = edges.data[self.type].reshape((-1, 1))
        input = t.cat((servEmbs, attrQoS), -1)
        input[edges.data['mask'] == 0] = 0
        eventMsg = self.W_su(input)
        return {'eventMsg': eventMsg}

    def forward(self, graph):
        # 输入图（批量图）
        batchGraph = graph  # type:d.DGLGraph

        # 通过用户到服务的消息传递更新图
        batchGraph['us'].update_all(self.userToServ, fn.mean('eventMsg', 'rev'), etype='us')
        # 通过服务到用户的消息传递更新图
        batchGraph['su'].update_all(self.servToUser, fn.mean('eventMsg', 'rev'), etype='su')

        # 对用户和服务节点的特征进行激活并更新
        batchGraph.nodes['user'].data['in'] += self.act(batchGraph.nodes['user'].data['rev'])
        batchGraph.nodes['serv'].data['in'] += self.act(batchGraph.nodes['serv'].data['rev'])

        # 获取更新后的用户和服务节点的特征
        newUserEmbedding = batchGraph.nodes['user'].data['in']
        newServEmbedding = batchGraph.nodes['serv'].data['in']

        return newUserEmbedding, newServEmbedding


# 信息聚合
class Interaction(t.nn.Module):

    def __init__(self, dim, order=1, type='RT'):
        super(Interaction, self).__init__()
        self.dim = dim
        self.order = order
        self.type = type
        self.layers = t.nn.ModuleDict()
        for i in range(order):
            self.layers[f'Layer_{i + 1}'] = MessagePassing(self.dim, self.type)

    def forward(self, graph):
        batchGraph = graph  # type:d.DGLGraph
        retUEmbeds = []
        retSEmbeds = []

        for i in range(self.order):
            ufeats, sfeats = self.layers[f'Layer_{i + 1}'](batchGraph)
            retUEmbeds += [ufeats]
            retSEmbeds += [sfeats]

        retUser = t.cat(retUEmbeds, -1)
        retServ = t.cat(retSEmbeds, -1)

        return retUser, retServ


# Use for Link Prediction， 预测
class LinkPrediction(t.nn.Module):
    def __init__(self, in_feats, neigh_feats):
        super(LinkPrediction, self).__init__()

        self.layers = t.nn.Sequential(
            t.nn.Linear(in_feats, 128, bias=False),
            t.nn.LayerNorm(128),
            t.nn.LeakyReLU()
        )

        self.readout_layers = t.nn.Sequential(
            t.nn.Linear((128 + neigh_feats), 128),
            t.nn.LeakyReLU(),
            t.nn.Linear(128, 1)
        )

    def forward(self, inputs, neigh_inputs=None):
        logits = self.layers(inputs)
        concatEmbeds = t.cat([logits, neigh_inputs], dim=1) if neigh_inputs is not None else logits
        pred = self.readout_layers(concatEmbeds).squeeze()
        return pred


# 双图，
class DualGNN(t.nn.Module):

    def __init__(self, **kwargs):
        super(DualGNN, self).__init__()

        self.dim = kwargs['dim']
        self.order = kwargs['order']  # Mo
        self.gpu = kwargs['gpu']
        self.ctx = kwargs['ctx']
        self.type = kwargs['type']

        # Initialize TESM Computing Components
        self.neighbourEncoder = SubGraphConv(self.dim, order=1, type=self.type)
        self.interactionEncoder = Interaction(self.dim, order=self.order, type=self.type)  # Mo

        # Initialize Node Features Transformation
        self.ufeats = t.nn.Linear(200, self.dim, bias=False)
        self.sfeats = t.nn.Linear(3800, self.dim, bias=False)

        # Link Prediction
        infeats = (2 * self.order) * self.dim
        print("infeats = ", infeats)
        ctx_infeats = (2 * self.dim) if self.ctx else 0
        print("ctx_infeats = ", ctx_infeats)

        self.Link = LinkPrediction(infeats, ctx_infeats)

        print("self.RTLink is \n", self.Link)
        if t.cuda.is_available():
            self.cuda()

        # 反馈
        logger.info('ISPA-GNN Model Loaded!')

    def setType(self, type):
        self.type = type
        self.interactionEncoder.type = type
        self.neighbourEncoder.type = type

    def forward(self, batchGraph, neighGraph=None):
        # Subgraph
        neigh_readout = None
        if neighGraph is not None:
            neighGraph.nodes['user'].data['in'] = self.ufeats(neighGraph.nodes['user'].data['x'])
            neighGraph.nodes['serv'].data['in'] = self.sfeats(neighGraph.nodes['serv'].data['x'])
            neigh_readout = self.neighbourEncoder(neighGraph)

        # Neighbor
        batchGraph.nodes['user'].data['in'] = self.ufeats(batchGraph.nodes['user'].data['x'])
        batchGraph.nodes['serv'].data['in'] = self.sfeats(batchGraph.nodes['serv'].data['x'])
        UserEmbeds, ServEmbeds = self.interactionEncoder(batchGraph)

        # 取Embedding
        users = batchGraph.ndata['label']['user'] == 0
        items = batchGraph.ndata['label']['serv'] == 0

        x = t.cat([UserEmbeds[users], ServEmbeds[items]], -1)

        rtLink = self.Link(x, neigh_readout)

        return rtLink


def PredictLoss(pred, label):
    return F.l1_loss(pred, label)
