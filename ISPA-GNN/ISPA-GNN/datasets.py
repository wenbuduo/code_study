import pandas as pd
import dgl as d
import torch as t
import numpy as np
from torch.utils.data import Dataset
from logger import *
from scipy.stats import boxcox
from scipy.special import inv_boxcox


def multi_hot(len, idx):
    vec = [0.] * len
    for id in idx:
        vec[id] = 1.
    return vec


class BoxCoxMinMax:
    def __init__(self, enable):
        self.boxcox = enable

    def fit(self, x):
        if self.boxcox:
            arr, lmbda = boxcox(x)
            self.lmbda = lmbda
            self.min = np.min(arr)
            self.max = np.max(arr)
            ret = (arr - self.min) / (self.max - self.min)
            return t.from_numpy(ret)
        else:
            self.min = t.min(x)
            self.max = t.max(x)
            ret = (x - self.min) / (self.max - self.min)
            return ret

    def inv(self, x):
        x = (self.max - self.min) * x + self.min
        if self.boxcox:
            return inv_boxcox(x, self.lmbda)
        else:
            return x


# 获得两个表格
def get_graph():
    user_table = pd.read_csv('./Dataset/userlist_table.csv')
    serv_table = pd.read_csv('./Dataset/wslist_table.csv')
    Uctx2Idx = {}
    Idx2Uctx = {}
    uidx = 0
    usrc = []
    udst = []
    ufeats = []

    for row in list(user_table.iterrows()):
        row = row[1]
        UserID = row['[User ID]']
        Country = row['[Country]']
        AS = row['[AS]']

        if Country not in Uctx2Idx:
            Uctx2Idx[Country] = uidx
            Idx2Uctx[uidx] = Country
            uidx += 1

        if AS not in Uctx2Idx:
            Uctx2Idx[AS] = uidx
            Idx2Uctx[uidx] = AS
            uidx += 1

        Country = Uctx2Idx[Country]
        AS = Uctx2Idx[AS]

        usrc.append(UserID)
        udst.append(Country)

        usrc.append(UserID)
        udst.append(AS)
        # Que1: What for the func. multi_hot? One-hot!
        ufeats.append(multi_hot(200, [Country, AS]))
    #         ugctry.append(multi_hot(150, [Gcountry]))
    #         ugas.append(multi_hot(1200, [GAS]))

    Sctx2Idx = {}
    Idx2Sctx = {}
    sidx = 0
    ssrc = []
    sdst = []
    sfeats = []
    for row in list(serv_table.iterrows()):
        row = row[1]
        ServID = row['[Service ID]']
        Country = row['[Country]']
        AS = row['[AS]']
        Provider = row['[Service Provider]']

        if Country not in Sctx2Idx:
            Sctx2Idx[Country] = sidx
            Idx2Sctx[sidx] = Country
            sidx += 1

        if AS not in Sctx2Idx:
            Sctx2Idx[AS] = sidx
            Idx2Sctx[sidx] = AS
            sidx += 1

        if Provider not in Sctx2Idx:
            Sctx2Idx[Provider] = sidx
            Idx2Sctx[sidx] = Provider
            sidx += 1

        Country = Sctx2Idx[Country]
        AS = Sctx2Idx[AS]
        Provider = Sctx2Idx[Provider]

        ssrc.append(ServID)
        sdst.append(Country)

        ssrc.append(ServID)
        sdst.append(AS)

        ssrc.append(ServID)
        sdst.append(Provider)
        sfeats.append(multi_hot(3800, [Country, AS, Provider]))

    logger.info('Loading Matrix Dataset')

    rtMatrix = np.loadtxt('./Dataset/rtMatrix.txt')
    tpMatrix = np.loadtxt('./Dataset/tpMatrix.txt')

    # Filtering the invalid entries.
    msrc, mdst = [], []

    rts = []
    tps = []

    # 建立总图
    for i in range(339):
        for j in range(5825):
            if 0 < rtMatrix[i][j] < 19.9 and 0 < tpMatrix[i][j] < 1000:
                msrc.append(i)
                mdst.append(j)
                rts.append(rtMatrix[i][j])
                tps.append(tpMatrix[i][j])

    data_dict = {
        ('uctx', 'cu', 'user'): (udst, usrc),
        ('user', 'uc', 'uctx'): (usrc, udst),
        ('sctx', 'cs', 'serv'): (sdst, ssrc),
        ('serv', 'sc', 'sctx'): (ssrc, sdst),
        ('user', 'us', 'serv'): (msrc, mdst),
        ('serv', 'su', 'user'): (mdst, msrc)
    }

    graph = d.heterograph(data_dict=data_dict)  # type:d.DGLGraph

    RTValue = t.tensor(rts).float()
    TPValue = t.tensor(tps).float()

    graph.edata['RT'] = {('user', 'us', 'serv'): RTValue,
                         ('serv', 'su', 'user'): RTValue}

    graph.edata['TP'] = {('user', 'us', 'serv'): TPValue,
                         ('serv', 'su', 'user'): TPValue}

    graph.ndata['x'] = {
        'user': t.tensor(ufeats),
        'serv': t.tensor(sfeats)
    }

    logger.info('Matrix Dataset Loaded!')
    return graph


def collateV3(data):
    g_list, n_list, label_list = map(list, zip(*data))
    g = d.batch(g_list)
    n = d.batch(n_list)
    g_label = t.stack(label_list)
    return g, n, g_label


#####
# 从满矩阵把密度极小的矩阵切出来
def train_test_split(graph, density):
    p = np.random.RandomState(seed=233).permutation(graph.number_of_edges('us'))
    trainsize = int(density * graph.number_of_edges('us'))
    trainIdx = p[: trainsize]
    testIdx = p[trainsize:][: trainsize]
    trainSubGraph = d.edge_subgraph(graph, edges={
        ('uctx', 'cu', 'user'): np.arange(0, graph.number_of_edges('cu')),
        ('user', 'uc', 'uctx'): np.arange(0, graph.number_of_edges('uc')),
        ('sctx', 'cs', 'serv'): np.arange(0, graph.number_of_edges('cs')),
        ('serv', 'sc', 'sctx'): np.arange(0, graph.number_of_edges('sc')),
        ('user', 'us', 'serv'): trainIdx,
        ('serv', 'su', 'user'): trainIdx},
    )

    testSubGraph = d.edge_subgraph(graph, edges={
        ('uctx', 'cu', 'user'): np.arange(0, graph.number_of_edges('cu')),
        ('user', 'uc', 'uctx'): np.arange(0, graph.number_of_edges('uc')),
        ('sctx', 'cs', 'serv'): np.arange(0, graph.number_of_edges('cs')),
        ('serv', 'sc', 'sctx'): np.arange(0, graph.number_of_edges('sc')),
        ('user', 'us', 'serv'): testIdx,
        ('serv', 'su', 'user'): testIdx},
    )
    return trainSubGraph, testSubGraph


# 主要采用这一个
class QoSDataSetV3(Dataset):

    def __init__(self, graph, boxcox, type='RT'):
        self.graph = graph  # type:d.DGLGraph
        self.bipart = d.node_type_subgraph(graph, ['user', 'serv'])
        self.links = self.graph.all_edges(etype='us')
        self.labels = self.graph.edata[type][('user', 'us', 'serv')]
        self.transform = BoxCoxMinMax(enable=boxcox)
        self.labels = self.transform.fit(self.labels)
        self._neighUsersCache = {}
        self._neighServsCache = {}
        self._fringeUsersCache = {}
        self._fringeServsCache = {}
        self._noPruningNeighUsersCache = {}
        self._noPruningNeighServsCache = {}
        self._gcache = {}

    def __len__(self):
        return len(self.links[0])

    def _fringeUsers(self, graph, user, serv):
        if serv not in self._fringeUsersCache:
            neighUser = graph.in_edges(serv, etype='us')[0]
            self._fringeUsersCache[serv] = neighUser
        neighUser = self._fringeUsersCache[serv]
        neighUser = np.setdiff1d(neighUser, user)
        return neighUser

    def _fringeServs(self, graph, user, serv):
        if user not in self._fringeServsCache:
            neighServ = graph.in_edges(user, etype='su')[0]
            self._fringeServsCache[user] = neighServ
        neighServ = self._fringeServsCache[user]
        neighServ = np.setdiff1d(neighServ, serv)
        return neighServ

    # _neighUsers和_neighServs这两个方法是用来？
    def _neighUsers(self, graph, user):
        if user not in self._neighUsersCache:
            uctxNodes = graph.out_edges(user, etype='uc')[1]
            neighUser = graph.out_edges(uctxNodes, etype='cu')[1]
            neighUser = np.setdiff1d(neighUser, user)
            self._neighUsersCache[user] = [uctxNodes, neighUser]
        uctxNodes, neighUser = self._neighUsersCache[user]

        neighSize = 35  # 60
        if len(neighUser) > neighSize:
            degrees = graph.out_degrees(neighUser, etype='us').numpy()
            sumd = np.sum(degrees)
            degrees = degrees / sumd
            non_zero = len(degrees[degrees > 0])
            sample_size = min(neighSize, non_zero)
            neighUser = np.random.choice(neighUser, sample_size, replace=False, p=degrees)
        return uctxNodes, neighUser

    def _neighServs(self, graph, serv):
        if serv not in self._neighServsCache:
            sctxNodes = graph.out_edges(serv, etype='sc')[1]
            neighServ = graph.out_edges(sctxNodes, etype='cs')[1]
            neighServ = np.setdiff1d(neighServ, serv)
            self._neighServsCache[serv] = [sctxNodes, neighServ]
        sctxNodes, neighServ = self._neighServsCache[serv]
        neighSize = 85  # 300
        if len(neighServ) > neighSize:
            degrees = graph.out_degrees(neighServ, etype='su').numpy()
            sumd = np.sum(degrees)
            degrees = degrees / sumd
            non_zero = len(degrees[degrees > 0])
            sample_size = min(neighSize, non_zero)
            neighServ = np.random.choice(
                neighServ, sample_size, replace=False, p=degrees)
        return sctxNodes, neighServ

    def extractSubgraph(self, **kwargs):
        u_fringe = kwargs['u']
        v_fringe = kwargs['s']

        neighUser = self._fringeUsers(self.bipart, u_fringe, v_fringe)
        neighServ = self._fringeServs(self.bipart, u_fringe, v_fringe)

        u_nodes = t.cat((t.tensor([u_fringe]), t.from_numpy(neighUser)))
        v_nodes = t.cat((t.tensor([v_fringe]), t.from_numpy(neighServ)))

        sg = d.node_subgraph(self.bipart, nodes={
            'user': u_nodes,
            'serv': v_nodes,
        })  # type:d.DGLGraph

        su = sg.nodes('user')[sg.ndata[d.NID]['user'] == u_fringe]
        sv = sg.nodes('serv')[sg.ndata[d.NID]['serv'] == v_fringe]

        sg.ndata['label'] = {
            'user': t.ones(sg.number_of_nodes('user')),
            'serv': t.ones(sg.number_of_nodes('serv'))
        }

        sg.ndata['label']['user'][su] = 0
        sg.ndata['label']['serv'][sv] = 0

        mask_us_edge = sg.edge_ids(su, sv, etype='us')
        mask_su_edge = sg.edge_ids(sv, su, etype='su')

        sg.edges['us'].data['mask'] = t.ones(sg.number_of_edges(etype='us'))
        sg.edges['su'].data['mask'] = t.ones(sg.number_of_edges(etype='su'))

        sg.edges['us'].data['mask'][mask_us_edge] = 0
        sg.edges['su'].data['mask'][mask_su_edge] = 0
        return sg

    def extractNeighSubgraph(self, **kwargs):
        u_fringe = kwargs['u']
        v_fringe = kwargs['s']
        graph = kwargs['graph']  # type:d.DGLGraph
        u_ctxNodes, neighUsers = self._neighUsers(graph, u_fringe)
        v_ctxNodes, neighServs = self._neighServs(graph, v_fringe)
        sg = d.node_subgraph(
            self.bipart,
            nodes={'user': neighUsers, 'serv': neighServs}
        )  # type:d.DGLGraph
        return sg


    def __getitem__(self, idx):
        centerUser, centerServ = int(self.links[0][idx]), int(self.links[1][idx])
        subgraph = self.extractSubgraph(graph=self.graph, u=centerUser, s=centerServ)
        ngraph = self.extractNeighSubgraph(graph=self.graph, u=centerUser, s=centerServ)
        label = self.labels[idx]
        return subgraph, ngraph, label
