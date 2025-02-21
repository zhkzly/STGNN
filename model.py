


import torch
import numpy as np
import torch.nn as nn
from utils import generate_embedding_matrix, attention_with_dot_scale, calculate_multi_head_attention
import math
import torch.nn.functional as F
import sys
from copy import deepcopy


class GCN(nn.Module):
    '''this is used to calculate the gcn '''

    def __init__(self, d: int, dtype=torch.float32):
        super(GCN, self).__init__()
        self.linear = nn.Linear(in_features=d, out_features=d, bias=False, dtype=dtype)
        self.dtype = dtype

    def forward(self, X, A):
        '''
        this is used to calculate the gcn
        :param A:  tensor(N,N) calculated by adja matrix
        :param X: tensor(b,T,N,d)
        :return:output:tensor (b,T,N,d)
        '''
        A = A.type(self.dtype)
        X = X.type(self.dtype)
        # (N,N)@(b,T,N,d)->(b,T,N,d)->(b,T,N,d)
        output = self.linear(torch.matmul(A, X))
        return torch.relu(output)

#######################################
# ##check GCN
# A=torch.randn(9,9)
# X=torch.randn(3,4,9,7)
# print(X.shape)
# gcn=GCN(7)
# output=gcn(X,A)
# print(f'output.shape:{output.shape}')

#####################################3

class Temporal_embedding(nn.Module):
    '''
    this is used to calculate the embedding of Spatial and time
    lookup_index:array,shape(1,)
    '''

    def __init__(self, d, max_len, lookup_index=None, dropout=0., dtype=torch.float32,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(Temporal_embedding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.dtype = dtype
        self.max_len = max_len
        self.d = d
        self.lookup_index = lookup_index
        self.device = device

    def forward(self, X):
        '''
        this is used to return the embedding of the input
        :param X: tensor(b,T,N,d),raw input
        :return embedding_X:tensor(b,T,N,d) the result of the embedding input (embedding with position embedding,spatial embedding)
        '''
        with torch.no_grad():

            X = X.type(self.dtype)
            # (max_len,d)
            # (max_len,d)
            Etp = generate_embedding_matrix(self.max_len, self.d, dtype=self.dtype).to(self.device)
            # (b,T,N,d)+(N,d)
            if self.lookup_index is not None:
                # (N,d)+(b,T,N,d)
                tp_x = X + Etp[self.lookup_index].unsqueeze(dim=0).unsqueeze(dim=0)
            # (b,T,N,d)
            else:
                tp_x = X + Etp[:X.shape[-2], :].unsqueeze(dim=0).unsqueeze(dim=0)

            embedding_X = tp_x

        # 以此为新的起点,等价于embedding_X.detach()
        return self.dropout(embedding_X)

###########################################
###check Temporal_embedding
# X=torch.randn(3,4,5,6)
# t_embed=Temporal_embedding(d=6,max_len=18)
# output=t_embed(X)
# print(f'the shape of the output:{output.shape}')

##########################################


class Spatial_embedding(nn.Module):

    def __init__(self, N, d, A, dtype=torch.float32, dropout=0.0, gcn=None, num_of_smooth_layers=0):
        '''

        :param N: the number of the nodes
        :param d:
        :param A: norm_adja
        :param dtype:
        :param dropout:
        :param gcn:
        :param num_of_smooth_layers:the times of passing the X into gcn
        '''
        super(Spatial_embedding, self).__init__()
        # self.embed=nn.Embedding(N,d,dtype=dtype)
        self.embed = nn.Parameter(torch.empty((N, d), dtype=dtype))
        nn.init.xavier_uniform_(self.embed)
        self.N = N
        self.d = d
        self.dtype = dtype
        self.A = A.type(self.dtype)
        self.dropout = nn.Dropout(dropout)
        assert gcn is not None and num_of_smooth_layers > 0
        self.gcns = nn.ModuleList([gcn(self.d) for _ in range(num_of_smooth_layers)])     #可以保证会进行更新W

    def forward(self, X):
        X = X.type(self.dtype)

        embedding = self.embed.unsqueeze(0).unsqueeze(0)  # (1,1,N,d)
        for gcn in self.gcns:
            embedding = gcn(embedding, self.A)

        X = X + embedding
        return self.dropout(X)

######################################
###check Spatial_embedding


# A=torch.randn(9,9)
# X=torch.randn(3,4,9,7)
# spatial_embedding=Spatial_embedding(N=9,d=7,A=A,gcn=GCN,num_of_smooth_layers=3)
# output=spatial_embedding(X)
# print(X.shape)
#
# print(f'output.shape:{output.shape}')

#########################################



config_of_SDGCN = {
    'd': None,
    'norm_adja': None,
    'dropout': None,
    'use_ln': None,
    'use_res': None,
    'dtype': None,

}


class SDGCN_with_ln_res(nn.Module):
    '''
    this is used to calculate the dynamic graph convolution
    :param d,int,the size of the hidden dim
    '''

    def __init__(self, config: dict):
        super(SDGCN_with_ln_res, self).__init__()
        self.d = config['d']
        self.dtype = config.setdefault('dtype', torch.float32)
        self.p = config.setdefault('dropout', 0.0)
        self.dropout = nn.Dropout(p=self.p)
        self.ordinary_gcn = GCN(self.d, dtype=self.dtype)
        self.A = config['norm_adja']
        self.use_ln = config.setdefault('use_ln', True)
        self.use_res = config.setdefault('use_res', True)
        if self.use_ln:
            self.ln = nn.LayerNorm(self.d,dtype=self.dtype)

    def forward(self, X):
        '''

        :param A:tensor(N,N),calculated from adja
        :param X: tensor(b,T,N,d),the output of the last layer
        :return: output,tensor(b,T,N,d),
        '''
        # (b,T,N,d)@(b,T,d,N)->(b,T,N,N)
        St = self.dropout(
            torch.softmax(torch.matmul(X, X.permute(dims=(0, 1, 3, 2))), dim=-1) / math.sqrt(self.d )+ 1e-8)
        output = self.ordinary_gcn(X,self.A * St)
        if self.use_ln:
            output = self.ln(output)
        if self.use_res:
            output = output + X

        return output

###############################################
######check SDGCN_with_ln_res
#
# A=torch.randn(9,9)
# X=torch.randn(3,4,9,7)
#
# config_of_SDGCN = {
#     'd': 7,
#     'norm_adja': A,
#     'dropout': 0.,
#     'use_ln': True,
#     'use_res': True,
#     'dtype': torch.float32,
#
# }
#
# sdgcn=SDGCN_with_ln_res(config=config_of_SDGCN)
#
# output=sdgcn(X)
# print(f'the shape of the output:{output.shape}')
#
#
#######################################################



class SDGCN_with_ln_res_scaled(nn.Module):
    '''
    this is used to calculate the dynamic graph convolution
    :param d,int,the size of the hidden dim
    '''

    def __init__(self, config: dict):
        super(SDGCN_with_ln_res_scaled, self).__init__()
        self.d = config['d']
        self.dtype = config.setdefault('dtype', torch.float32)
        self.p = config.setdefault('dropout', 0.0)
        self.dropout = nn.Dropout(p=self.p)
        self.ordinary_gcn = GCN(self.d, dtype=self.dtype)
        self.A = config['norm_adja']
        self.use_res = config.setdefault('use_res', True)
        self.use_ln = config.setdefault('use_ln', True)
        if self.use_ln:
            self.ln = nn.LayerNorm(self.d)

    def forward(self, X):
        '''

        :param A:tensor(N,N),calculated from adja
        :param X: tensor(b,T,N,d),the output of the last layer
        :return: output,tensor(b,T,N,d),
        '''
        # (b,T,N,d)@(b,T,d,N)->(b,T,N,N)
        St = self.dropout(
            torch.softmax(torch.matmul(X, X.permute(dims=(0, 1, 3, 2))), dim=-1) / math.sqrt(self.d + 1e-8))
        output = self.ordinary_gcn(X,self.A * St) / math.sqrt(self.d + 1e-9)
        if self.use_ln:
            output = self.ln(output)
        if self.use_res:
            output = output + X
        return output

#########################################
### check
# A=torch.randn(9,9)
# X=torch.randn(3,4,9,7)
#
# config_of_SDGCN = {
#     'd': 7,
#     'norm_adja': A,
#     'dropout': 0.,
#     'use_ln': True,
#     'use_res': True,
#     'dtype': torch.float32,
#
# }
#
# sdgcn=SDGCN_with_ln_res_scaled(config=config_of_SDGCN)
# output=sdgcn(X)
# print(f'the shape of the output:{output.shape}')

################################################



class CausalConv(nn.Module):
    '''
    this is used to calculate the causal convolution which is to avoid the layers to use the future input
    '''

    def __init__(self, in_channels, out_channels, kernel_size=(1, 3), padding=(1, 1), stride=(1, 1),
                 dtype=torch.float32):
        super(CausalConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        # self.W=nn.Parameter(self.empty((self.out_channels,self.in_channels,*kernel_size) if len(kernel_size)==2 else (self.out_channels,self.in_channels,1,self.kernel_size),dtype=dtype))
        # nn.init.xavier_uniform_(self.W)
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                              kernel_size=self.kernel_size, stride=self.stride
                              , padding=self.padding, dtype=dtype)
        self.dtype = dtype

    def forward(self, X):
        '''
        conv(input),input:(b,d,N,T)
        :param X: tensor(b,T,N,d)
        :return:
        '''
        b, T, N, d = X.shape
        # kernel_size=self.kernel_size[-1] if len(self.kernel_size)==2 else self.kernel_size
        # stride=self.stride[-1] if len(self.stride)==2 else self.stride
        # #padding default equal,1D conv (kernel)(1,kernel_size)
        # padding=self.padding[-1] if len(self.padding)==2 else self.padding
        # Tout=(2*self.padding+T-kernel_size)//stride+1
        # # N_next=()
        # padding=self.padding
        # HH,WW=self.kernel_size
        # H,W=(N+2*padding[0]-HH)//self.stride[0],(T+2*padding[1]-WW)//self.stride[1]
        # output=torch.zeros((b,self.out_channels,H,W),dtype=self.dtype)
        # #(b,d,N,T)
        # X=X.permute(dims=(0,-1,2,1))
        # padded_x=F.pad(X,pad=(padding,padding),value=0)#在H,W方向进行padding
        # calculate convolution,can use broadcast all,but maybe out of memory
        # W(out_channels,in_channels,H,W),仅有沿着T的方向有mask,也就是需要考虑不能利用过去的结果
        # for ba in range(b):     #批量
        #     for c in range(self.out_channels):  #通道
        #         for h in range(H):      #高
        #             for w in range(W):      #宽
        #                 output[ba,c,h,w]=output[ba,c,h,w]+torch.sum(X[ba,:,h*self.stride[0]:h*self.stride+HH,w*self.stride[1]:w*self.stride[1]+WW]*self.W).item()
        # ok 本质上causal_convolution 与standard convolution 没什么大的区别，仅仅是进行了一定的平移,可以通过padding进行简单的结果变换
        # 其实必须使用两个进行占位才能显示出来仅仅使用了之前的时间步，否则就是普通的卷积操作
        X = X.type(self.dtype)
        # (b,T,N,d)->(b,d,N,T)
        X = X.permute(dims=(0, -1, 2, 1))
        # (b,c,N,Tr)
        output = self.conv(X)
        output = F.pad(output, pad=(math.floor(self.kernel_size[-1] / 2), 0), value=0.)
        # (b,T,N,d)
        output = output.permute(dims=(0, -1, 2, 1))
        return output


#
# class TrSelfAttention_with_ln_res(nn.Module):
#     '''
#     this is used to calculate the temporary trend aware self-attention,for each head
#     we will use the traditional convolution to calculate the attention
#
#     '''
#
#     # def __init__(self,d:int,n_head:int,q_norm_conv=False,k_norm_conv=False,dtype=torch.float32,kernel_size=(1,3),out_channels=64,padding=(1,1),stride=(1,1),mask=False):
#     def __init__(self, config:dict):
#         '''
#
#         :param config: dict('q_norm_conv','k_norm_conv',...)
#
#         '''
#         super(TrSelfAttention_with_ln_res, self).__init__()
#         # for key, value in kwargs.items():
#         #     setattr(self, key, value)
#         self.q_norm_conv=config.setdefault('q_norm_conv',False)
#         self.k_norm_conv=config.setdefault('k_norm_conv',False)
#         self.dtype=config.setdefault('dtype',torch.float32)
#         self.kernel_size=config.setdefault('kernel_size',(1,3))
#         self.out_channels=config.setdefault('out_channels',64)
#         self.padding=config.setdefault('padding',(1,1))
#         self.stride=config.setdefault('stride',(1,1))
#         self.mask=config.setdefault('mask',False)
#         self.d=config['d']
#         self.n_head=config['n_head']
#         self.dim_per_head = self.d // self.n_head
#         self.Wv = nn.Parameter(torch.empty((self.d, self.d), dtype=self.dtype))
#         # self.Wk=nn.Parameter(torch.empty((d,d),dtype=self.dtype))
#         # self.W=nn.Parameter(torch.empty((3,d,d),dtype=self.dtype))
#         self.Wo = nn.Parameter(torch.empty((self.d,self.d)))
#         self.use_res=config.setdefault('use_res',True)
#         self.use_ln=config.setdefault('use_ln',True)
#         if self.use_ln:
#             self.ln=nn.LayerNorm(self.d)
#
#         if self.q_norm_conv:
#             # 使用ModuleList可以放置某些参数不会出现在self.parameters()中导致无法更新，为了保证最后的d_model 不变大小，实际上out_channels=d_model//self.n_head
#             self.conv_q_list = nn.ModuleList([nn.Conv2d(in_channels=self.dim_per_head, out_channels=self.out_channels,
#                                                         kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
#                                               for i in range(self.n_head)])
#         else:
#             self.conv_k_list = nn.ModuleList([CausalConv(in_channels=self.dim_per_head, out_channels=self.out_channels,
#                                                          kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
#                                               for i in range(self.n_head)])
#         if self.k_norm_conv:
#             self.conv_q_list = nn.ModuleList(
#                 [nn.Conv2d(in_channels=self.dim_per_head, out_channels=self.out_channels, kernel_size=self.kernel_size \
#                            , stride=self.stride, padding=self.padding) for i in range(self.n_head)])
#         else:
#             self.conv_k_list = nn.ModuleList(
#                 [CausalConv(in_channels=self.dim_per_head, out_channels=self.out_channels, kernel_size=self.kernel_size \
#                             , stride=self.stride, padding=self.padding) for i in range(self.n_head)])
#         self.ln = nn.LayerNorm(self.d)
#         for param in self.parameters():
#             if len(param.shape) >= 2:
#                 torch.nn.init.xavier_uniform_(param)
#             else:
#                 torch.nn.init.uniform_(param)
#
#     def forward(self, X):
#         '''
#         because this is self-attention ,so i don't need to input the query,key,v
#         :param X: tensor(b,N,T,d)
#         :return:output,tensor(b,N,T,d)
#         '''
#         b, N, T, d = X.shape
#         X = X.type(self.dtype)
#
#         # Q,K are all the X,in the paper ,they try to 保持input与output形状相同
#         trself_atten = torch.empty((self.n_head, b, T, N, d // self.n_head))
#         # (b,T,N,d)@(d,d)->(b,T,N,d)->(b,T,N,self.n_head,self.nun_per_head)->(n_head,b,T,N,num_per_head)
#         V = torch.matmul(X, self.Wv).reshape((b, N, T, self.n_head, -1)).permute(dims=(3, 0, 2, 1, 4))
#         # Q=torch.matmul(X,self.Wq)
#         # (b,T,N,d)->(b,T,N,n_head,num_per_head)->(n_head,b,T,N,num_per_head)
#         tempt_X = X.reshape((b, N, T, self.n_head, -1)).permute(dims=(3, 0, 1, 2, 4))
#         for i, conv in enumerate(zip(self.conv_q_list, self.conv_k_list)):
#             # X:(b,T,N,dim_per_head)->input:(b,num_per_head,N,T)
#             # conv(input)->(b,c,N,Tr)
#             trself_atten[i] = attention_with_dot_scale(conv[0](tempt_X[i]).permute(dims=(0, -1, 1, 2)),
#                                                        conv[1](tempt_X[i].permute(dims=(0, -1, 1, 2)), mask=self.mask),
#                                                        V[i])
#
#         # (n_head,b,Tr,N,num_per_head)->(b,Tr,N,dnext)
#         atten = trself_atten.permute(dims=(1, 2, 3, 0, 4)).flatten(start_dim=-2)
#         # (b,T,N,d)
#         output = torch.matmul(atten, self.Wo)
#         if self.use_res:
#             output = output + X
#         if self.use_ln:
#             output=self.ln(output)
#         return output

config_of_TrSelf = {
    'd': None,
    'n_head': None,
    'q_norm_conv': None,
    'k_norm_conv': None,
    'kernel_size': None,
    'out_channels': None,
    'stride': None,
    'padding': None,

    'num_of_weeks': None,
    'num_of_days': None,
    'num_of_hours': None,
    'points_per_hour': None,

    'use_ln': None,
    'use_res': None,
    'dropout': None,

    'mask': None,
    'dtype': None,

}


# 源码中采用的是在所有的输入之中采用卷积，而不是论文中提到的分为不同的head
class TrSelfAttention_with_ln_res(nn.Module):
    '''
    this is used to calculate the temporary trend aware self-attention,for each head
    we will use the traditional convolution to calculate the attention

    '''

    # def __init__(self,d:int,n_head:int,q_norm_conv=False,k_norm_conv=False,dtype=torch.float32,kernel_size=(1,3),out_channels=64,padding=(1,1),stride=(1,1),mask=False):
    def __init__(self, config: dict):
        '''

        :param config: dict('q_norm_conv','k_norm_conv',...)

        '''
        super(TrSelfAttention_with_ln_res, self).__init__()
        # for key, value in kwargs.items():
        #     setattr(self, key, value)
        self.q_norm_conv = config.setdefault('q_norm_conv', False)
        self.k_norm_conv = config.setdefault('k_norm_conv', False)
        self.dtype = config.setdefault('dtype', torch.float32)
        self.kernel_size = config.setdefault('kernel_size', (1, 3))
        self.out_channels = config.setdefault('out_channels', 64)
        self.padding = config.setdefault('padding', (1, 1))
        self.stride = config.setdefault('stride', (1, 1))
        self.mask = config.setdefault('mask', False)
        self.d = config['d']
        self.n_head = config['n_head']
        self.num_per_head = self.d // self.n_head
        self.linearWv = nn.Linear(in_features=self.d, out_features=self.d, dtype=self.dtype)
        self.linearWo = nn.Linear(in_features=self.d, out_features=self.d, dtype=self.dtype)
        # self.Wk=nn.Parameter(torch.empty((d,d),dtype=self.dtype))
        # self.W=nn.Parameter(torch.empty((3,d,d),dtype=self.dtype))

        # this is used to calculate conv for different slices,Xg,Xl
        self.num_of_weeks = config.setdefault('num_of_weeks', 0)
        self.num_of_days = config.setdefault('num_of_days', 0)
        self.points_per_hour = config.setdefault('num_of_hours', 0)
        self.num_of_hours = config.setdefault('num_of_hours', 0)
        self.week_length = self.num_of_weeks * self.num_of_hours * self.points_per_hour
        self.day_length = self.num_of_days * self.num_of_hours * self.points_per_hour
        self.hours_length = self.num_of_hours * self.points_per_hour
        self.dropout = config.setdefault('dropout', 0.)
        self.use_res = config.setdefault('use_res', True)
        self.use_ln = config.setdefault('use_ln', True)
        if self.use_ln:
            self.ln = nn.LayerNorm(self.d)

        if self.q_norm_conv:
            # 使用ModuleList可以放置某些参数不会出现在self.parameters()中导致无法更新，为了保证最后的d_model 不变大小，实际上out_channels=d_model//self.n_head
            self.conv_q = nn.Conv2d(in_channels=self.dim_per_head, out_channels=self.out_channels,
                                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        else:
            self.conv_q = CausalConv(in_channels=self.dim_per_head, out_channels=self.out_channels,
                                     kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        if self.k_norm_conv:
            self.conv_k = nn.Conv2d(in_channels=self.dim_per_head, out_channels=self.out_channels,
                                    kernel_size=self.kernel_size \
                                    , stride=self.stride, padding=self.padding)
        else:
            self.conv_k = CausalConv(in_channels=self.dim_per_head, out_channels=self.out_channels,
                                     kernel_size=self.kernel_size \
                                     , stride=self.stride, padding=self.padding)

        # for param in self.parameters():
        #     if len(param.shape) >= 2:
        #         torch.nn.init.xavier_uniform_(param)
        #     else:
        #         torch.nn.init.uniform_(param)

    def forward(self, X, Q, K, V):
        '''
        because this is self-attention ,so i don't need to input the query,key,v
        :param X: tensor(b,N,T,d)
        Q:(b,T,N,d)
        :return:output,tensor(b,N,T,d)
        '''

        b, T, N, d = V.shape
        # Q,K are all the X,in the paper ,they try to 保持input与output形状相同
        trself_atten = torch.empty((self.n_head, b, T, N, d // self.n_head))
        # (b,T,N,d)@(d,d)->(b,T,N,d)->(b,T,N,self.n_head,self.nun_per_head)->(n_head,b,T,N,num_per_head)
        V = self.linearWv(V).reshape((b, N, T, self.n_head, -1)).permute(dims=(3, 0, 2, 1, 4))

        # 进行卷积，也就是和普通的是一样的

        assert self.points_per_hour != 0 and self.num_of_hours != 0
        # 仅需要一段，
        if self.num_of_weeks == 0 and self.num_of_days == 0:
            # (b,T,N,d)->permute(0,3,1,2)->(b,d,N,T)->reshape(b,n_head,num_per_head,N,T)
            Q = self.conv_q(Q.permute(dims=(0, 3, 1, 2))).reshape((b, self.n_head, -1, N, T))
            K = self.conv_k(K.permute(dims=(0, 3, 1, 2))).reshape((b, self.n_head, -1, N, T))

        if self.num_of_weeks > 0 and self.num_of_days > 0:
            # 用来存放不同的T slice
            Q_index = [Q[:, :self.week_length, :, :], Q[:, self.week_length:self.week_length + self.day_length, :, :],
                       Q[:, -self.hours_length, :, :]]
            Q_list = map(
                lambda x: self.conv_q(x.permute(dims=(0, 3, 1, 2))).reshape((b, self.n_head, self.num_per_head, N, -1)),
                Q_index)
            K_index = [K[:, :self.week_length, :, :], K[:, self.week_length:self.week_length + self.day_length, :, :],
                       K[:, -self.hours_length, :, :]]
            K_list = map(
                lambda x: self.conv_k(x.permute(dims=(0, 3, 1, 2))).reshape((b, self.n_head, self.num_per_head, N, -1)),
                K_index)
            Q = torch.cat([*Q_list], dim=-1)
            # (b,n_head,num_per_head,N,T)
            K = torch.cat([*K_list], dim=-1)

        if self.week_length > 0 and self.num_of_days <= 0:
            Q_index = [Q[:, :self.week_length, :, :],
                       Q[:, -self.hours_length, :, :]]
            Q_list = map(
                lambda x: self.conv_q(x.permute(dims=(0, 3, 1, 2))).reshape((b, self.n_head, self.num_per_head, N, -1)),
                Q_index)
            K_index = [K[:, :self.week_length, :, :],
                       K[:, -self.hours_length, :, :]]
            K_list = map(
                lambda x: self.conv_k(x.permute(dims=(0, 3, 1, 2))).reshape((b, self.n_head, self.num_per_head, N, -1)),
                K_index)
            Q = torch.cat([*Q_list], dim=-1)
            # (b,n_head,num_per_head,N,T)
            K = torch.cat([*K_list], dim=-1)

        if self.week_length < 0 and self.day_length > 0:
            Q_index = [Q[:, :self.day_length, :, :],
                       Q[:, -self.hours_length, :, :]]
            Q_list = map(
                lambda x: self.conv_q(x.permute(dims=(0, 3, 1, 2))).reshape((b, self.n_head, self.num_per_head, N, -1)),
                Q_index)
            K_index = [K[:, :self.day_length, :, :],
                       K[:, -self.hours_length, :, :]]
            K_list = map(
                lambda x: self.conv_k(x.permute(dims=(0, 3, 1, 2))).reshape((b, self.n_head, self.num_per_head, N, -1)),
                K_index)
            Q = torch.cat([*Q_list], dim=-1)
            # (b,n_head,num_per_head,N,T)
            K = torch.cat([*K_list], dim=-1)

        # calculate the multi head
        # Q,shape(b,n_head,num_per_head,N,T)->(n_head,b,T,N,d_k)
        Q = Q.permute(dims=(1, 0, 4, 3, 2))
        K = K.permute(dims=(1, 0, 4, 3, 2))
        output = calculate_multi_head_attention(Q, K, V, mask=self.mask, dropout=self.dropout)
        output = self.linearWo(output)
        if self.use_res:
            output = output + X
        if self.use_ln:
            output = self.ln(output)
        return output


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config_of_MHSelf = {
    'd': None,
    'd_head': None,
    'device': None,
    'droupout': None,
    'use_ln': None,
    'use_res': None,
    'dtype': None,
    'mask': None

}


class MHSelfAttention_with_ln_res(nn.Module):
    def __init__(self, d_model: int, n_head: int, use_res=True, use_ln=True, dtype=torch.float32, device=device,
                 mask=False, dropout=0.):
        super(MHSelfAttention_with_ln_res, self).__init__()
        self.d_model = d_model
        self.device = device
        self.dropout = nn.Dropout(p=dropout)
        # 其实device是没必要的
        self.ln = torch.nn.LayerNorm(self.d_model, device=device, dtype=dtype)
        self.W_head = nn.Parameter(torch.empty((3, d_model, d_model), dtype=dtype))
        self.bias = nn.Parameter(torch.empty((3, d_model), dtype=dtype))
        self.W_out = nn.Parameter(torch.empty((d_model, d_model), dtype=dtype))
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.xavier_uniform_(self.W_out)
        torch.nn.init.xavier_uniform_(self.bias)
        self.Wq, self.Wk, self.Wv = self.W_head
        self.n_head = n_head
        self.dtype = dtype
        self.num_per_head = self.d_model // self.n_head
        self.mask = mask
        self.use_res = use_res
        self.use_ln = use_ln
        if self.use_ln:
            self.ln = nn.LayerNorm(self.d)

    def forward(self, X, Q, K, V):
        '''
        :param X:tensor,b,T,N,d_model
        :return:b,T,N,d_model
        '''
        X = X.type(self.dtype)
        b, T, N, d = X.shape
        # (b,T,N,d)@(d,num_per_head*n_head)->(b,T,N,num_per_head*n_head)->(n_head,b,T,N,num_per_head)
        # 原作者使用的是linear(),所以有bias
        Q = (torch.matmul(Q, self.Wq) + self.bias[0][None, None, None, :]).view(
            (b, T, N, self.num_per_head, -1)).permute(dims=(-1, 0, 1, 2, 3))
        K = (torch.matmul(K, self.Wk) + self.bias[1][None, None, None, :]).view(
            (b, T, N, self.num_per_head, -1)).permute(dims=(-1, 0, 1, 2, 3))
        V = (torch.matmul(V, self.Wv) + self.bias[2][None, None, None, :]).view(
            (b, T, N, self.num_per_head, -1)).permute(dims=(-1, 0, 1, 2, 3))
        # (n_head,b,T,N,d)@(n_head,b,T,d,N)->(n_head,b,T,N,N)
        output = calculate_multi_head_attention(Q, K, V, mask=self.mask, dropout=self.dropout)

        output = torch.matmul(output, self.W_out)
        if self.use_res:
            output = output + X
        if self.use_ln:
            output = self.ln(output)
        return output


# design encoder
'''
config:dict
configs:[config,config]
'''

configs = {
    'TrSelf':
        {

        },
    'MHSelf':
        {

        },
    'SDGCN':
        {

        }
}


class EncoderLayer(nn.Module):
    def __init__(self, configs: dict, sdgcn_with_scale=True, tr_self_att=True):
        super(EncoderLayer, self).__init__()
        if tr_self_att:
            atten = TrSelfAttention_with_ln_res(configs['TrSelf'])

        else:
            atten = MHSelfAttention_with_ln_res(configs['MHSelf'])

        if sdgcn_with_scale:
            sdgcn = SDGCN_with_ln_res_scaled(configs['SDGCN'])
        else:
            sdgcn = SDGCN_with_ln_res(configs['SDGCN'])

        self.coder_layer = nn.Sequential(atten, sdgcn)

        self.d = configs['SDGCN']['d']

    def forward(self, X):
        '''

        :param X:
        :return:
        '''

        output = self.coder_layer(X, X, X)
        return output


# define Encoder
class Encoder(nn.Module):
    def __init__(self, encoder_layer, N: int):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(*[deepcopy(encoder_layer) for _ in range(N)])
        self.ln = nn.LayerNorm(encoder_layer.d)

    def forward(self, x):
        output = self.encoder(x)
        return self.ln(output)


# design decoder

# 同decoder
configs_of_decoderlayer = {

}


class DecoderLayer(nn.Module):
    def __init__(self, configs: dict, sdgcn_with_scale=True, tr_self_att=True):
        super(DecoderLayer, self).__init__()
        if tr_self_att:
            atten_1 = TrSelfAttention_with_ln_res(configs['TrSelf'])
            atten_2 = TrSelfAttention_with_ln_res(configs['MHself'])

        else:
            atten_1 = MHSelfAttention_with_ln_res(configs['MHSelf'])
            atten_2 = MHSelfAttention_with_ln_res(configs['MHSelf'])

        if sdgcn_with_scale:
            sdgcn = SDGCN_with_ln_res_scaled(configs['SDGCN'])
        else:
            sdgcn = SDGCN_with_ln_res(configs['SDGCN'])
        self.tr_self_atten = atten_1  # 这个是self-attention
        self.tr_atten = atten_2  # 这个不是self_attention
        self.sdgcn = sdgcn
        self.d = configs['SDGCN']['d']

    def forward(self, X, memory):

        x = self.tr_self_atten(X, X, X, X)  # (x,Q,K,V)
        x = self.tr_atten(x, x, memory, memory)
        return self.sdgcn(x)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.decoders = nn.ModuleList([deepcopy(layer) for _ in range(N)])
        self.d = layer.d
        self.ln = nn.LayerNorm(self.d)

    def forward(self, x, memory):
        for decoder in self.decoders:
            x = decoder(x, memory)
        return self.ln(x)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embedding, tar_embedding, final_project):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embedding
        self.tar_embed = tar_embedding
        self.final_project = final_project

    def forward(self, src, tar):
        encoder_output = self.encode(src)
        return self.decode(tar, encoder_output)

    def encode(self, src):
        h = self.src_embed(src)
        return self.encoder(h)

    def decode(self, tar, encoder_output):
        return self.final_project(self.decoder(self.tar_embed(tar), encoder_output))


def search_index(max_len, num_of_depend, num_for_predict, points_per_hours, units):
    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = max_len - points_per_hours * units * i
        for j in range(num_for_predict):
            end_idx = start_idx + j
            x_idx.append(end_idx)
    return x_idx


# 创建完整的模型：
# model_config
'''
{

}

'''


# 源码中的num_for_predict 其实可以理解为说是：假如需要预测2个小时，一个小时有12个数据，那么num_for_predict=24,采用记下索引的目的就是
# 保证在Xl,Xg,X在cat的时候就能保证含有时间步的信息，因为直接cat，就会丧失时间步了，因为是简单的cat

def make_model(encoder_configs, decoder_configs, num_of_decoder_layers, src_size, num_of_encoder_layers, N, d,
               num_for_final_output, num_of_weeks, num_of_days, num_of_hours, points_per_hour,
               norm_adja, num_for_predict, smooth_layers_num, sdgcn_with_scale=True, tr_self_att=True, TE=True, SE=True,
               dropout=0.):
    # 需要完成词嵌入，模型返回
    encoder_layer = EncoderLayer(configs=encoder_configs, sdgcn_with_scale=sdgcn_with_scale, tr_self_att=tr_self_att)
    decoder_layer = DecoderLayer(configs=decoder_configs, sdgcn_with_scale=sdgcn_with_scale, tr_self_att=tr_self_att)
    encoder = Encoder(encoder_layer, num_of_encoder_layers)
    decoder = Decoder(decoder_layer, num_of_encoder_layers)
    # 属实看不懂这个max_len为什么这样去取，和我想得不一样。
    max_len = max(num_of_weeks * 7 * 24 * num_for_predict, num_of_days * 24 * num_for_predict,
                  num_of_hours * num_for_predict)
    w_index = search_index(max_len, num_of_weeks, num_for_predict, points_per_hour, 7 * 24)
    d_index = search_index(max_len, num_of_days, num_for_predict, points_per_hour, 24)
    h_index = search_index(max_len, num_of_hours, num_for_predict, points_per_hour, 1)
    en_lookup_index = h_index + d_index + w_index

    src_project = nn.Linear(in_features=src_size, out_features=d)
    tar_project = nn.Linear(num_for_predict, d)

    if TE and SE:
        encoder_temporal_embedding = Temporal_embedding(d=d, max_len=max_len, lookup_index=en_lookup_index,
                                                        dropout=dropout)
        decoder_temporal_embedding = Temporal_embedding(d=d, max_len=num_for_predict, dropout=dropout)
        spatial_embedding = Spatial_embedding(N=N, d=d, A=norm_adja, num_of_smooth_layers=smooth_layers_num)
        encoder_embedding = nn.Sequential(src_project, encoder_temporal_embedding, deepcopy(spatial_embedding))
        decoder_embedding = nn.Sequential(tar_project, decoder_temporal_embedding, spatial_embedding)
    elif TE and not SE:
        encoder_temporal_embedding = Temporal_embedding(d=d, max_len=max_len, lookup_index=en_lookup_index,
                                                        dropout=dropout)
        decoder_temporal_embdedding = Temporal_embedding(d=d, max_len=num_for_predict, dropout=dropout)
        # spatial_embedding=Spatial_embedding(N=N,d=d,A=norm_adja,num_of_smooth_layers=smooth_layers_num)
        encoder_embedding = nn.Sequential(src_project, encoder_temporal_embedding)
        decoder_embedding = nn.Sequential(tar_project, decoder_temporal_embdedding)

    elif not TE and SE:

        spatial_embedding = Spatial_embedding(N=N, d=d, A=norm_adja, num_of_smooth_layers=smooth_layers_num)
        encoder_embedding = nn.Sequential(src_project, deepcopy(spatial_embedding))
        decoder_embedding = nn.Sequential(tar_project, spatial_embedding)

    else:  # (not TE and not SE)
        encoder_embedding = src_project
        decoder_embedding = tar_project
    # if decoder_embedding is None:
    #     a=4
    # else:
    #     a: int=5
    # c = a

    final_project = nn.Linear(in_features=d, out_features=num_for_final_output)

    encoder_decoder = EncoderDecoder(encoder=encoder, decoder=decoder, src_embedding=encoder_embedding,
                                     tar_embedding=decoder_embedding, final_project=final_project)

    return encoder_decoder
