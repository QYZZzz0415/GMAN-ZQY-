import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
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
    retrun: [batch_size, num_his + num_pred, num_vertex, D]
    '''

    def __init__(self, D, bn_decay):
        super(STEmbedding, self).__init__()
        self.FC_se = FC(
            input_dims=[D, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)
        #D是多头注意力通道输出维度
        self.FC_te = FC(
            input_dims=[115, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)  # input_dims = time step per day + days per week=108+7=115

    def forward(self, SE, TE, T=108):
        # spatial embedding
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.FC_se(SE)
        # temporal embedding
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7)#长期趋势
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T)#当天时间（早高峰/晚高峰)
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 108, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1).to(device)#最后一维
        TE = TE.unsqueeze(dim=2).to(device)#第二维
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
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # [K * batch_size, num_step, num_vertex, num_vertex]
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)  # orginal K, change to batch_size 分割再拼接
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
        X = torch.cat((X, STE), dim=-1)
        # [batch_size, num_step, num_vertex, K * d]
        query = self.FC_q(X)
        key = self.FC_k(X)
        value = self.FC_v(X)
        # [K * batch_size, num_step, num_vertex, d]
        query = torch.cat(torch.split(query, self.K, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.K, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.K, dim=-1), dim=0)
        # query: [K * batch_size, num_vertex, num_step, d]
        # key:   [K * batch_size, num_vertex, d, num_step]
        # value: [K * batch_size, num_vertex, num_step, d]
        #重新排列向量维度信息
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        # [K * batch_size, num_vertex, num_step, num_step]
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        # mask attention score 屏蔽未来时间步信息
        if self.mask:
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
        attention = F.softmax(attention, dim=-1)
        # [batch_size, num_step, num_vertex, D]
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
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))#z大 空间特征占比高
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
        HS = self.spatialAttention(X, STE)#空间注意力机制计算
        HT = self.temporalAttention(X, STE)#时间注意力机制计算
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
        X = torch.matmul(attention, value)#根据注意力分数对值进行加权求和
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        del query, key, value, attention
        return X


class GMAN(nn.Module):
    '''
    GMAN
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

    def __init__(self, SE, args, bn_decay): #空间注意力机制
        super(GMAN, self).__init__()
        L = args.L #number of STatt blocks
        K = args.K #number of attention heads
        d = args.d # dims of each attention head outputs
        D = K * d #sum dims
        self.num_his = args.num_his #history
        self.SE = SE
        self.STEmbedding = STEmbedding(D, bn_decay)
        self.STAttBlock_1 = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(L)])
        self.STAttBlock_2 = nn.ModuleList([STAttBlock(K, d, bn_decay) for _ in range(L)])
        self.transformAttention = transformAttention(K, d, bn_decay)
        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)#升维
        self.FC_2 = FC(input_dims=[D, D], units=[D, 1], activations=[F.relu, None],
                       bn_decay=bn_decay)#降维
        self.linear = nn.Linear(11, 150)
        #self.linear_1 = nn.Linear(100, 300)
        # self.linear_2 = nn.Linear(2, 1)
        self.linear_3 = nn.Linear(276 + 150 , 276)#和节点数对齐
    def forward(self, X, TE, Input):
        Input_1 = self.linear(Input)
        # Input_1 = self.linear_1(Input_1)
        # Input_1 = self.linear_2(Input_1)
        X_in=torch.cat((X, Input_1), dim=-1)#把交通流量数据和外部因素拼接
        X = self.linear_3(X_in)
        X = torch.unsqueeze(X, -1)#调整输入数据的形状 增加一个维度方便在卷积层升维
        X = self.FC_1(X)#升维 输入到FC层 进行非线性变换
        # STE
        STE = self.STEmbedding(self.SE, TE)#时空嵌入
        STE_his = STE[:, :self.num_his]#历史时间步长 num_his是历史时间步数
        STE_pred = STE[:, self.num_his:]#预测时间步长 num_pred是预测时间步数
        # encoder
        for net in self.STAttBlock_1:
            X = net(X, STE_his)#STAttBlock forward
        # transAtt
        X = self.transformAttention(X, STE_his, STE_pred)
        # decoder
        for net in self.STAttBlock_2:
            X = net(X, STE_pred)#STAttBlock forward
        # output
        X = self.FC_2(X)
        del STE, STE_his, STE_pred
        return torch.squeeze(X, 3)#移除X张量上第三维尺寸为 1 的维度


