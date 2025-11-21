import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def AddClusterTemporal(INPUT): # [256, 325, 12, 8] -> [256, 325, 60, 8]
    
    batch_count = INPUT.shape[0]

    X = [torch.split(INPUT[i], 5, dim=0) for i in range(batch_count)]
    X = [list(split) for split in X]
    X = [[t.reshape(-1, t.size(-1)) for t in X[i]] for i in range(batch_count)]
    X = [[t for t in X[i] for _ in range(5)] for i in range(batch_count)]
    X = torch.stack([torch.stack(t) for t in X])

    return X

    # OLD IN CASE WE NEED

    # grouping = torch.tensor([
    #         43, 48, 5, 5, 42, 1, 1, 42, 10, 10,
    #         10, 60, 45, 45, 0, 32, 60, 44, 21, 61,
    #         24, 61, 24, 12, 12, 12, 12, 12, 21, 34,
    #         44, 51, 16, 34, 44, 44, 47, 34, 55, 41,
    #         45, 51, 55, 55, 51, 36, 50, 23, 64, 9,
    #         64, 57, 19, 30, 57, 19, 23, 50, 19, 60,
    #         23, 50, 46, 40, 14, 63, 47, 56, 50, 14,
    #         19, 55, 64, 60, 16, 49, 58, 41, 46, 2,
    #         14, 34, 54, 50, 18, 47, 11, 60, 40, 40,
    #         30, 47, 15, 51, 54, 11, 11, 63, 40, 0,
    #         0, 47, 49, 39, 49, 37, 49, 18, 59, 15,
    #         37, 2, 14, 0, 57, 61, 41, 40, 39, 59,
    #         59, 57, 0, 54, 59, 17, 23, 37, 18, 56,
    #         46, 34, 19, 4, 15, 4, 51, 39, 23, 56,
    #         32, 63, 61, 9, 5, 36, 54, 39, 58, 7,
    #         17, 27, 11, 38, 14, 22, 49, 4, 45, 15,
    #         55, 11, 29, 8, 46, 39, 17, 7, 38, 32,
    #         56, 9, 37, 53, 54, 64, 30, 32, 16, 32,
    #         25, 2, 37, 43, 63, 46, 25, 18, 1, 16,
    #         56, 61, 36, 64, 25, 33, 25, 25, 1, 15,
    #         27, 7, 30, 48, 7, 17, 20, 22, 43, 6,
    #         45, 4, 20, 9, 31, 28, 28, 59, 43, 42,
    #         20, 22, 7, 18, 26, 22, 30, 58, 42, 20,
    #         22, 10, 28, 29, 13, 20, 29, 9, 44, 28,
    #         63, 13, 17, 13, 4, 29, 31, 31, 28, 29,
    #         16, 24, 31, 26, 24, 53, 3, 53, 3, 62,
    #         13, 43, 6, 57, 36, 62, 2, 1, 62, 13,
    #         36, 8, 8, 26, 53, 53, 3, 21, 38, 24,
    #         62, 62, 58, 35, 52, 3, 31, 52, 3, 33,
    #         21, 35, 33, 35, 38, 41, 52, 26, 6, 6,
    #         10, 21, 41, 38, 35, 26, 48, 27, 27, 48,
    #         35, 6, 5, 27, 52, 8, 33, 48, 8, 52,
    #         33, 2, 42, 5, 58
    #     ])

    # # input -> arranged helper
    # transform = []
    # for number in range(65):
    #     indices = (grouping == number).nonzero(as_tuple=True)[0]
    #     transform.extend(indices.tolist())

    # # arranged -> output helper
    # count = {}
    # back_to_normal = [] 
    # for index, value in enumerate(grouping):
    #     if value.item() not in count:
    #         count[value.item()] = 0
            
    #     count[value.item()] += 1
    #     k = count[value.item()]  
    #     new_value = 5 * (value.item() - 1) + k - 1
    #     back_to_normal.append(new_value)
    
    # # Refer to archive/permute.ipynb
    # # input -> arranged
    # for i in range(batch_count):
    #     INPUT[i] = INPUT[i][transform]
    
    # X = [torch.split(INPUT[i], 5, dim=0) for i in range(batch_count)]
    # X = [list(split) for split in X]
    # X = [[t.reshape(-1, t.size(-1)) for t in X[i]] for i in range(batch_count)]
    # X = [[t for t in X[i] for _ in range(5)] for i in range(batch_count)]
    # X = torch.stack([torch.stack(t) for t in X])

    # # Create a new tensor of shape (325, 24, 8)
    # new_X= torch.zeros(batch_count, 325, 24, 8)

    # # Keep the first 12 slices the same
    # new_X[:, :, :12, :] = X[:, :, :12, :]

    # # Calculate the sums for the 13th to 24th slices
    # for i in range(12, 24):
    #     for j in range(4):
    #         new_X[:, :, i, :] += X[:, :, i + 12 * j, :]
    #     new_X[:, :, i, :] /= 4

    # del X, INPUT

    # # arranged -> output
    # for i in range(batch_count):
    #     new_X[i] = new_X[i][back_to_normal]
    
    # return new_X


class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,padding=0, bias=use_bias)
        self.batch_norm = nn.BatchNorm2d(output_dims, momentum=bn_decay)
        torch.nn.init.xavier_uniform_(self.conv.weight)

        if use_bias:
            torch.nn.init.zeros_(self.conv.bias)


    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = F.relu_(x)
        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class STEmbedding(nn.Module):
    '''
    spatio-temporal embedding
    SE:     [num_vertex, D]
    TE:     [batch_size, num_his + num_pred, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    return: [batch_size, num_his + num_pred, num_vertex, D]
    '''

    def __init__(self, D, bn_decay):
        super(STEmbedding, self).__init__()
        self.FC_se = FC(
            input_dims=[D, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)

        self.FC_te = FC(
            input_dims=[295, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)  # input_dims = time step per day + days per week=288+7=295

    def forward(self, SE, TE, T=288):
        # spatial embedding
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.FC_se(SE)
        # temporal embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7)
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 288, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        TE = TE.unsqueeze(dim=2)
        TE = self.FC_te(TE)
        del dayofweek, timeofday
        return SE + TE


class spatialAttention(nn.Module):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, K, d, bn_decay):
        super(spatialAttention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, STE):
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_step, num_vertex, K * d + D (from STE)]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # FC turned from 2D to D not 'd'
        # [K * batch_size, num_step, num_vertex, d]
        
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        
        # [K * batch_size, num_step, num_vertex, num_vertex]
        test = key.transpose(2,3)
        attention = torch.matmul(query, key.transpose(2, 3)) 
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        
        # Multiply the attention to the value for the final embedding
        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)  # orginal K, change to batch_size
        X = self.FC(X)
        del query, key, value, attention
        return X


class temporalAttention(nn.Module):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, num_vertex, D]
    STE:    [batch_size, num_step, num_vertex, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, K, d, bn_decay, mask=True):
        super(temporalAttention, self).__init__()
        D = K * d
        
        self.d = d
        self.K = K
        
        self.mask = mask
        self.FC_q = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=2 * D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)
          

    def forward(self, X, STE):
        batch_size_ = X.shape[0]
        # X - [batch_size, num_step, num_vertex, K * d + D]
        X = torch.cat((X, STE), dim=-1)
        
        # q,k,v - [32,12,325,64]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        
        # q,k,v - [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)

        # query: [K * batch_size, num_vertex, num_step, d]
        # key:   [K * batch_size, num_vertex, d, num_step]
        # value: [K * batch_size, num_vertex, num_step, d]
        # [256, 325, 12, 8]
        # [256, 325, 8, 12]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        
        query = AddClusterTemporal(query) # [256, 325, 12, 8] -> [256, 325, 60, 8]
        value = AddClusterTemporal(value) 
        
        # We create a new key which concats 5 permutations of X
        # we get Q = [256, 325, 60, 8] and K = [256, 325, 8, 12]
        
        # [K * batch_size, num_vertex, num_step, num_step]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        # we want A = [256, 325, 60, 12]
        
        
        # mask attention score
        if self.mask: # not used at the moment
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_vertex = X.shape[2]
            mask = torch.ones(num_step, num_step)
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, -2 ** 15 + 1)
        
        # softmax
        attention = F.softmax(attention, dim=-2)
        
        # [batch_size, num_step, num_vertex, D]
        attention = attention.permute(0,1,3,2)
        
        # we want A = [256, 325, 12, 60]
        # We want to change V to [256, 325, 60, 8]
        # We apply the same permutation we did to key except its in 3rd dim
        
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size_, dim=0), dim=-1)  # orginal K, change to batch_size
        X = self.FC(X)
        del query, key, value, attention
        return X


class gatedFusion(nn.Module):
    '''
    gated fusion
    HS:     [batch_size, num_step, num_vertex, D]
    HT:     [batch_size, num_step, num_vertex, D]
    D:      output dims
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, D, bn_decay):
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=False)
        self.FC_xt = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=True)
        self.FC_h = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)

    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.FC_h(H)
        del XS, XT, z
        return H


class STAttBlock(nn.Module):
    def __init__(self, K, d, bn_decay, mask=False):
        super(STAttBlock, self).__init__()
        self.spatialAttention = spatialAttention(K, d, bn_decay)
        self.temporalAttention = temporalAttention(K, d, bn_decay, mask=mask)
        self.gatedFusion = gatedFusion(K * d, bn_decay)

    def forward(self, X, STE):
        HS = self.spatialAttention(X, STE)
        HT = self.temporalAttention(X, STE)
        H = self.gatedFusion(HS, HT)
        del HS, HT
        return torch.add(X, H)


class transformAttention(nn.Module):
    '''
    transform attention mechanism
    X:        [batch_size, num_his, num_vertex, D]
    STE_his:  [batch_size, num_his, num_vertex, D]
    STE_pred: [batch_size, num_pred, num_vertex, D]
    K:        number of attention heads
    d:        dimension of each attention outputs
    return:   [batch_size, num_pred, num_vertex, D]
    '''

    def __init__(self, K, d, bn_decay):
        super(transformAttention, self).__init__()
        D = K * d
        self.K = K
        self.d = d
        self.FC_q = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_k = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC_v = FC(input_dims=D, units=D, activations=F.relu,
                       bn_decay=bn_decay)
        self.FC = FC(input_dims=D, units=D, activations=F.relu,
                     bn_decay=bn_decay)

    def forward(self, X, STE_his, STE_pred):
        batch_size = X.shape[0]
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(STE_pred)
        key = self.FC_k(STE_his)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # query: [K * batch_size, num_vertex, num_pred, d]
        # key:   [K * batch_size, num_vertex, d, num_his]
        # value: [K * batch_size, num_vertex, num_his, d]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_vertex, num_pred, num_his]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_pred, num_vertex, D]
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


class KMeans_GMAN(nn.Module):
    '''
    KMeans_GMAN
        X：       [batch_size, num_his, num_vertx]
        TE：      [batch_size, num_his + num_pred, 2] (time-of-day, day-of-week)
        SE：      [num_vertex, K * d]
        num_his： number of history steps
        num_pred：number of prediction steps
        T：       one day is divided into T steps
        L：       number of STAtt blocks in the encoder/decoder
        K：       number of attention heads
        d：       dimension of each attention head outputs
        return：  [batch_size, num_pred, num_vertex]
    '''

    def __init__(self, SE, args, bn_decay):
        super(KMeans_GMAN, self).__init__()
        L = args.L
        K = args.K
        d = args.d
        D = K * d
        
        self.num_his = args.num_his
        self.SE = SE
        self.STEmbedding = STEmbedding(D, bn_decay)
        self.STAttBlock_1 = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(L)])
        self.STAttBlock_2 = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(L)])
        self.transformAttention = transformAttention(K, d, bn_decay)
        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)
        self.FC_2 = FC(input_dims=[D, D], units=[D, 1], activations=[F.relu, None],
                       bn_decay=bn_decay)
        

    def forward(self, X, TE):
        # input
        X = torch.unsqueeze(X, -1)
        X = self.FC_1(X)
        # STE
        STE = self.STEmbedding(self.SE, TE)
        STE_his = STE[:, :self.num_his]
        STE_pred = STE[:, self.num_his:]
        # encoder
        for net in self.STAttBlock_1:
            X = net(X, STE_his)
        # transAtt
        X = self.transformAttention(X, STE_his, STE_pred)
        # decoder
        for net in self.STAttBlock_2:
            X = net(X, STE_pred)
        # output
        X = self.FC_2(X)
        del STE, STE_his, STE_pred
        return torch.squeeze(X, 3)