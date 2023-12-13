import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, eignvalue):

        eignvalue_con = eignvalue * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000)/self.hidden_dim)).to(eignvalue.device)
        position = eignvalue_con.unsqueeze(1) * div
        eignvalue_pos = torch.cat((eignvalue.unsqueeze(1), torch.sin(position), torch.cos(position)), dim=1)

        return self.eig_w(eignvalue_pos)

class FaXGNN_OC(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FaXGNN_OC, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, feature):
        feature = self.layer1(feature)
        feature = self.gelu(feature)
        feature = self.layer2(feature)
        return feature


class FaXLayer(nn.Module):

    def __init__(self, nbases, ncombines, prop_dropout=0.0, norm='none'):
        super(FaXLayer, self).__init__()
        self.prop_dropout = nn.Dropout(prop_dropout)

        if norm == 'none':
            self.weight = nn.Parameter(torch.ones((1, nbases, ncombines)))
        else:
            self.weight = nn.Parameter(torch.empty((1, nbases, ncombines)))
            nn.init.normal_(self.weight, mean=0.0, std=0.01)

        if norm == 'layer':
            self.norm = nn.LayerNorm(ncombines)
        elif norm == 'batch':
            self.norm = nn.BatchNorm1d(ncombines)
        else:
            self.norm = None

    def forward(self, feature):
        feature = self.prop_dropout(feature) * self.weight
        feature = torch.sum(feature, dim=1)

        if self.norm is not None:
            feature = self.norm(feature)
            feature = F.relu(feature)

        return feature


class FaXGNN(nn.Module):

    def __init__(self, nclass, nfeat, nlayer=1, hidden_dim=128, nheads=1,
                tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none'):
        super(FaXGNN, self).__init__()

        self.norm = norm
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        # feat_encoder = mlp (with relu)
        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nclass),
        )

        # linear_encoder = onne layer Linear
        self.linear_encoder = nn.Linear(nfeat, hidden_dim)
        
        self.classify = nn.Linear(hidden_dim, nclass)

        # position encoding
        self.eignvalue_encoder = SineEncoding(hidden_dim)
        
        # linear decoder
        self.decoder = nn.Linear(hidden_dim, nheads)

        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.oc_norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(tran_dropout)
        self.oc_dropout = nn.Dropout(tran_dropout)
        self.mha = nn.MultiheadAttention(hidden_dim, nheads, tran_dropout)
        
        # mlp with gelu
        self.oc = FaXGNN_OC(hidden_dim, hidden_dim, nclass)
        # self.oc = FaXGNN_OC(hidden_dim, hidden_dim, hidden_dim)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)
        if norm == 'none': # nlayer
            self.layers = nn.ModuleList([FaXLayer(2, nclass, prop_dropout, norm=norm) for i in range(nlayer)])
        else:
            self.layers = nn.ModuleList([FaXLayer(2, hidden_dim, prop_dropout, norm=norm) for i in range(nlayer)])

    def forward(self, eignvalue, eignvector, feature):
        eignvector_T = eignvector.permute(1, 0)
        if self.norm == 'none':
            h = self.feat_dp1(feature)
            h = self.feat_encoder(h)
            h = self.feat_dp2(h)
        else:
            h = self.feat_dp1(feature)
            h = self.linear_encoder(h) 
            
        eig = self.eignvalue_encoder(eignvalue) 

        mha_eig = self.mha_norm(eig)
        mha_eig, _ = self.mha(mha_eig, mha_eig, mha_eig)
        eig = eig + self.mha_dropout(mha_eig)

        oc_eig = self.oc_norm(eig)
        oc_eig = self.oc(oc_eig)
        eig = self.oc_dropout(oc_eig)

        eig_faxgnn = eig 
        for conv in self.layers:
            basic_feats = [h]
            eignvector_conv = eignvector_T @ h
            for i in range(self.nheads):
                basic_feats.append(eignvector @ (eig_faxgnn[:, i].unsqueeze(1) * eignvector_conv)) 
            basic_feats = torch.stack(basic_feats, axis=1)
            h = conv(basic_feats)

        if self.norm == 'none':
            return h
        else:
            h = self.feat_dp2(h)
            h = self.classify(h)
            return h

class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.body = GraphConvolution(nfeat,nhid,dropout)
        self.fc = nn.Linear(nhid, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.body(x, edge_index)
        x = self.fc(x)
        return x
    