
import functools

import torch.nn.functional
import numpy as np
import math
from configparser import ConfigParser
import os
from tqdm import tqdm
from typing import Callable,Optional,TypeVar,Union,Type
a=list
import argparse
#
# config=ConfigParser()
# q=config.sections()

def re_normalization(x, mean, std):
    x = x * std + mean
    return x
functools.cached_property

def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min) / (_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def generate_embedding_matrix(rows: int, cols: int, dtype=torch.float32):
    '''
    this func is used to calculate the embedding matrix  and E_tp
    :param rows: int ,rows of the matrix
    :param cols: int ,cols of the matrix,this is d_model
    :return:Ep:tensor,(rows,cols)
    '''
    Ep = torch.empty((rows, cols), dtype=dtype)
    for row in range(rows):
        for col in range(cols):
            if col % 2 == 0:
                Ep[row, col] = math.sin(row / (10000 ** (2 * col / cols)))
            else:
                Ep[row, col] = math.cos(row / (10000 ** (2 * col / cols)))
    return Ep


#####################################

# check generate_embedding_matrix
# output = generate_embedding_matrix(3, 3)
# print(f'output[1,1]:{math.cos(1/(10000)**(2*1/3))}:{output[1,1]},output[2,2]:{math.sin(2/(10000)**(2*2/3))}:{output[2,2]}')
# # print(output)
# import matplotlib.pyplot as plt
# plt.plot(output[:,0])
# plt.show()
######################################

def attention_with_dot_scale(Q, K, V, mask=False, dropout:Optional[Callable]=None):
    '''
    Optional[X]==Union[X,None],前者表明一个可以选择类型
    this is used to calculate the atten
    :param Q: tensor(b,T,N_1,c)
    :param K: tensor(b,T,N_2,c)
    :param V: tensor(b,T,N_2,c)
    :return:output:tensor(b,T,N_1,c)
    '''
    # (b,T,N,c)@(b,T,c,N)->(b,T,N,N)
    N_1 = Q.shape[-2]
    N_2 = K.shape[-2]
    a = torch.matmul(Q, K.permute(dims=(0, 1, 3, 2)))
    if mask:
        mask = torch.ones((N_1, N_2), dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)
        a[:, :, mask] = 1e-9
        print(mask)
    E = torch.softmax(a, dim=-1)
    if dropout is not None:
        E = dropout(E)
    output = torch.matmul(E, V) / math.sqrt(V.shape[-1])
    return output


###########################
# check the attention_with_dot_scale
# Q=torch.ones((2,3,6,5)) #查询六个
# K=torch.ones((2,3,4,5))
# V=torch.ones((2,3,4,5))
# output=attention_with_dot_scale(Q,K,V)
# #shape:(2,3,6,5),output[0,0,0,0]:1/sqrt(5)
# print(f'output[0,0,0,0]={1/math.sqrt(5)}')
# print(f'the shape of output:{output.shape},output[0,0,0,0]:{output[0,0,0,0]}')
# #check mask
# output=attention_with_dot_scale(Q,K,V,True)
# print(output)
##################################

def calculate_multi_head_attention(Q, K, V, mask=False, dropout:Optional[Callable] = None):
    '''
    calculate the multi_head_attention
    :param Q: tensor,shape(n_head,b,T,N_1,num_per_head)
    :param K: tensor,shape(n_head,b,T,N_2,num_per_head)
    :param V: tensor,shape(n_head,b,T,N_2,num_per_head)
    :param mask: bool,for decode
    :param dropout: None,or object
    :return:tensor ,shape(b,T,N_1,N_2*num_per_head)
    '''

    # (n_head,b,T,N,num_per_head)
    # return (b,T,N,d)
    n_head, b, T, N_1, num_per_head = Q.shape
    N_2 = K.shape[3]
    a = torch.matmul(Q, K.permute(dims=(0, 1, 2, 4, 3)))
    if mask:
        mask = torch.ones((N_1, N_2), dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)
        a[:, :, :, mask] = 1e-9
        print(mask)
    # E = self.dropout(torch.softmax(a, dim=-1) / math.sqrt(d))
    if dropout is not None:
        # E = torch.dropout(torch.softmax(a, dim=-1) / math.sqrt(num_per_head),p=dropout,train=True)    #无法在test的时候不使用
        E = dropout(torch.softmax(a, dim=-1) / math.sqrt(num_per_head))
    else:
        E = torch.softmax(a, dim=-1) / math.sqrt(num_per_head)
    # (n_head,b,T,N_1,d)
    output = torch.matmul(E, V)
    # catcontate,->(b,T,N,d*n_head)
    output = output.permute((1, 2, 3, 0, -1)).reshape(b, T, N_1, -1)
    return output


# torch.nn.functional.dropout()     #不同于torch.dropout

##################################
# check the calculate_multi_head_attention()
# Q = torch.randn(size=(5, 3, 4, 5, 3))
# K = torch.randn(5, 3, 4, 7, 3)
# V = torch.randn(5, 3, 4, 7, 3)
# # shape of the output:(3,4,5,15)
#
# output = calculate_multi_head_attention(Q, K, V, dropout=torch.nn.Dropout(.5))
# print(f'the shape of output:{output.shape}')

##################################


def calculate_interaction_of_node(A, directed:bool=False):
    '''
    this is used to calculate the interaction relationship among nodes
    :param A: numpy.ndarray,(N,N),this is the graph adjacency matrix
    :param directed: bool,directed or not
    :return: np.ndarray,(N,N),this is interaction relationship among nodes
    '''
    A = A + np.identity(A.shape[0])
    D = np.sum(A, axis=-1)
    print(D)
    if not directed:
        _D = np.diag(1 / (np.sqrt(D) + 1e-9))
        output = np.matmul(np.matmul(_D, A), _D)
        return output
    _D = np.diag(1 / (D + 1e-9))
    output = np.matmul(_D, A)
    return output

#######################################

#check calculate_interaction_of_node

# A=np.random.permutation(np.arange(1,13))
# print(A)
# output=calculate_interaction_of_node(A,False)
# print(f'the shape of the output:{output.shape}')

#####################################


##基本的数据处理与ASTGCN是一致的，所以基本可以抄之前的,哎，不能直接抄了,序号不是从零开始的

def calculate_adj_matrix(X, directed=False, dtype=np.float32):
    '''

    :param X: array,(N,3)->from,to,distance
    :return: Y:array,(N,N)
    '''
    N = int(np.max(X) + 1)
    A = np.zeros((N, N), dtype=dtype)
    for data in X:
        A[int(data[0]), int(data[1])] = 1
    return A

##################################
#check calculate_adj_matrix


#####################################




def generate_one_data(data, start_idx, end_idx, step, num_of_predict):
    '''
    give a start index and end_index,return some sample data from the data
    that is return x and y
    :param data:
    :param start_idx:
    :param end_idx:
    :param step:
    :param num_of_predict:需要的采样数，也就是，如果是之前5个小时的，那么就是5
    :return:
    '''
    output = []

    for i in range(start_idx, end_idx, step):
        output.append(data[i:i + num_of_predict])


    output = np.concatenate(output, axis=0)

    return output

##############################
## check generate_one_data



def load_data_from_file(data_path, header=0, dtype=np.float32, usecols=[0, 1, 2]):
    '''
    this is used to load the data from the file,and return the adja matrix
    :param data_path:
    :param file_type:
    :param start_idx:
    :return:
    '''
    if 'npz' in data_path:
        # 加载最终的original data
        X = np.load(data_path)['data']
        return X.astype(dtype)
    else:
        if '.csv' in data_path:
            import pandas as pd
            data = pd.read_csv(data_path, sep=',', header=header, usecols=usecols)
            data = data.values()
            return calculate_adj_matrix(data, dtype)


import pandas as pd


def get_adja_matrix(data_path, num_of_points, index_file=None, dtype=np.float32, directed=False, skiprows=0 ):
    '''
    this is get information of the nodes from the file,the index of the node may be specified by the other file
    :param data_path: string,the file path stored the data
    :param num_of_points: the total number of the nodes
    :param index_file: str,file stored the index of the node
    :param dtype: object
    :param directed:
    :param skiprows:
    :return:
    '''
    data = pd.read_csv(data_path, sep=',', skiprows=skiprows)
    datas = data.to_numpy().astype(np.int32)
    A = np.zeros((num_of_points, num_of_points), dtype=dtype)
    if index_file is not None:
        with open(index_file, mode='r') as f:
            num2ind = {int(num.rstrip('\n')): idx for idx, num in enumerate(f.readlines())}
            # print(num2ind)
        if directed:
            for data in datas:
                A[num2ind[data[0]], num2ind[data[1]]] = 1

        else:
            for data in datas:
                A[num2ind[data[0]], num2ind[data[1]]] = 1
                A[num2ind[data[1]], num2ind[data[0]]] = 1

    else:
        A = np.zeros((num_of_points, num_of_points), dtype=dtype)
        if directed:
            for data in datas:
                A[int(data[0]), int(data[1])] = 1
        else:
            for data in datas:
                A[int(data[0]), int(data[1])] = 1
                A[int(data[1]), int(data[0])] = 1
    return A


# a=np.ones((3,3))
# a.astype()
# a.data
# a.
# np.asarray()

######################################
# #check the get_adja_matrix
# output_1=get_adja_matrix(index_file="D:\i'a'm'z'k'l\AI\ASTGNN-main\data\PEMS03\PEMS03.txt",data_path="D:\i'a'm'z'k'l\AI\ASTGNN-main\data\PEMS03\PEMS03.csv",num_of_points=358)
#
# output_2=get_adja_matrix(directed=True,index_file="D:\i'a'm'z'k'l\AI\ASTGNN-main\data\PEMS03\PEMS03.txt",data_path="D:\i'a'm'z'k'l\AI\ASTGNN-main\data\PEMS03\PEMS03.csv",num_of_points=358)
#
# # output_1[0,357]=1,output_1[357,0]=1,output_2[0,357]=1,output_2[357,0]=0
#
# print(f'the output of the output_1[0,357]:{output_1[0,357]}:output_1[357,0]{output_1[357,0]}output_2[0,357]:{output_2[0,357]}:output_2[357,0]{output_2[357,0]}')

########################################


def generate_dataset(original_data, num_of_weeks=0., num_of_days=0., num_of_hours=0., num_of_predict=0., points_per_hours=12):
    '''

    :param original_data: array,(T,N,d)
    :param num_of_weeks:int,需要几个星期(Tw)
    :param num_of_days:需要几天(Td)
    :param num_of_hours:需要几个小时(Th)
    :param num_of_predict:预测多少个时间步（Tp*points_per_hours）
    :param points_per_hours:每个小时产生多少个时间步
    :return:output,list of tuple,[X,y],X:np.ndarray,shape(b,Th+Td+Tw,N,d),y:np.ndarray,shape(b,T,d)
    '''
    assert num_of_predict and num_of_days+num_of_weeks+num_of_hours>=0.1
    min_len = max(num_of_weeks * 7 * 24 * points_per_hours, num_of_hours * points_per_hours,
                  num_of_days * 24 * points_per_hours) + num_of_predict
    assert min_len <= original_data.shape[0]
    index = []
    results = []
    list_y = []
    end_idx = min_len

    while end_idx <= original_data.shape[0]:
        result = []

        if num_of_hours > 0:
            h_start_idx = end_idx - num_of_predict - num_of_hours * points_per_hours
            Xh = generate_one_data(original_data, h_start_idx, end_idx - num_of_predict, points_per_hours,
                                               points_per_hours)
            result.append(Xh)  # (T,N,d)

        if num_of_days > 0:
            d_start_idx = end_idx - num_of_days * 24 * points_per_hours-num_of_predict
            Xd = generate_one_data(original_data, d_start_idx, end_idx - num_of_predict, 24 * points_per_hours,
                                               num_of_predict)
            result.append(Xd)

        if num_of_weeks > 0:
            w_start_idx = end_idx - num_of_weeks* 7 * 24 * points_per_hours-num_of_predict
            Xw = generate_one_data(original_data, w_start_idx, end_idx - num_of_predict, 7 * 24 * points_per_hours,
                                               num_of_predict)
            result.append(Xw)

        # results:用作将数据集转化为批量(B,T,N,d)
        results.append(np.concatenate(result, axis=0))
        y = original_data[end_idx - num_of_predict: end_idx]
        list_y.append(y)
        end_idx += 1

    output = []
    output.append(np.stack(results, axis=0))  # (b,T,N,d):T=Tw+Td+Th
    output.append(np.stack(list_y, axis=0))

    return output


####################################################
###
### check generate_dataset

# original_data=np.random.randn(7000,30,7)
# output=generate_dataset(original_data,num_of_weeks=3,num_of_days=3,num_of_hours=7,num_of_predict=12,points_per_hours=12)
# ##X.shape:(941, 156, 30, 7),y.shape:(941, 12, 30, 7) ,156=3*12+3*12+7*12
# print(f'X.shape:{output[0].shape},y.shape:{output[1].shape}')

###################################################





def generate_train_val_test_set(dataset: list, ratio=(6, 8, 10), shuffle=False, merge=False):
    '''
    用来产生所需要的数据集
    :param dataset: list[X,y],也就是return of generate_dataset  #(num_of_data,T,N,C)
    :param ratio: 划分的比例，
    :param shuffle:boolean,数据项是否随机的选取放置在相应的train_set,val_set,test_set中
    :param merge: 是否需要val_set,if len(ratio)==2,也就是相当于merge==True
    :return:dataset:(train_set,val_set,test_set),train_set:tuple(num_of_dataset,Xh,Xd,Xw,y)
    '''
    num_of_data = dataset[0].shape[0]
    # if len(ratio) == 3 and merge:
    #     print('1')
    # ~Fasle==-2,so this code is wrong
    # if len(ratio) == 2 and ~merge:
    #     print('2')
    if len(ratio) == 3 and merge or len(ratio) == 2 and not merge:
        raise ValueError('the shape of ratio and the value of the merge are not consistent')
    if not merge:
        if shuffle:
            index_ = np.arange(num_of_data)
            np.random.shuffle(index_)
            train_set = [data[index_[0:int(num_of_data * ratio[0] / 10)]] for data in dataset]
            val_set = [data[index_[int(num_of_data * ratio[0] / 10):int((num_of_data * ratio[1]) / 10)]] for data in
                       dataset]
            test_set = [data[index_[int(num_of_data * ratio[1] / 10):]] for data in dataset]
        else:
            index_ = np.arange(dataset[0].shape[0])
            train_set = [data[index_[0:int(num_of_data * ratio[0] / 10)]] for data in dataset]
            val_set = [data[index_[int(num_of_data * ratio[0] / 10):int((num_of_data * ratio[1]) / 10)]] for data in
                       dataset]
            test_set = [data[index_[int(num_of_data * ratio[1] / 10):]] for data in dataset]
        print(
            f'train_set:X:{train_set[0].shape},y:{train_set[1].shape}')
        print(f'val_set:X:{val_set[0].shape},y:{val_set[1].shape}')
        print(f'test_set:X:{test_set[0].shape},y:{test_set[1].shape}')
        return train_set, val_set, test_set
    else:
        if shuffle:
            index_ = np.arange(dataset[0].shape[0])
            np.random.shuffle(index_)
            train_set = [data[index_[0:int(num_of_data * ratio[0] / 10)]] for data in dataset]
            test_set = [data[index_[int(num_of_data * ratio[1] / 10):]] for data in dataset]
        else:
            index_ = np.arange(dataset[0].shape[0])
            train_set = [data[index_[0:int(num_of_data * ratio[0] / 10)]] for data in dataset]
            test_set = [data[index_[int(num_of_data * ratio[1] / 10):]] for data in dataset]
        print(
            f"train_set:X:{train_set[0].shape},y:{train_set[1].shape}")
        print(f"test_set:Xh:{test_set[0].shape},y:{test_set[1].shape}")
        return train_set, test_set

###################################################
### check generate_train_val_test_set

# original_data=np.random.randn(7000,30,7)
# output=generate_dataset(original_data,num_of_weeks=3,num_of_days=3,num_of_hours=7,num_of_predict=12,points_per_hours=12)
# train_set,val_set,test_set=generate_train_val_test_set(output,shuffle=True)

###################################################



def MinMaxnormalization(train: list, test, val=None):
    '''
    归一化仅仅在各特征维度，类似与通道上d上进行,其实默认就是三个，没必要去考虑val是否需要了,
    input is the return of generate_train_val_test_set
    :param train: [X,y]:其中X:shape(b,T,N,d),np.array
    :param test:
    :param val:
    :return:(dict('max_value','min_value'),train_set,val_set,test_val)
    '''

    if val is not None:
        assert train[0].shape[-2] == test[0].shape[-2] and train[0].shape[-2] == val[0].shape[-2]   #判断N
        max_value = np.max(train[0], axis=(0, 1, 2), keepdims=True)       #torch.max,没有这个按照维度进行聚合
        min_value = np.min(train[0], axis=(0, 1, 2), keepdims=True)
        print(f'max_value:{max_value},min_value:{min_value}')
        def normalize(x):
            x = 1. * (x - min_value) / (max_value - min_value)
            x = 2. * x - 1.
            return x

        train[0] = normalize(train[0])
        train[1] = normalize(train[1])
        print(f'end of train')
        val[0] = normalize(val[0])
        val[1] = normalize(val[1])
        print(f'end of val')
        test[0] = normalize(test[0])
        test[1] = normalize(test[1])
        print(f'end of test')
        return {'max_value': max_value, 'min_value': min_value}, (train, val, test)

    else:
        assert train[0].shape[-2] == test[0].shape[-2]
        max_value = np.max(train[0], axis=(0, 1, 2), keepdims=True)
        min_value = np.min(train[0], axis=(0, 1, 2), keepdims=True)

        def normalize(x):
            x = 1. * (x - min_value) / (max_value - min_value)
            x = 2. * x - 1.
            return x

        train[0] = normalize(train[0])
        train[1] = normalize(train[1])
        print(f'end of train')
        test[0] = normalize(test[0])
        test[1] = normalize(test[1])
        print(f'end of test')
        return {'max_value': max_value, 'min_value': min_value}, (train, test)


#################################################
##check MinMaxnormalization

# original_data=np.random.randn(7000,30,7)
# output=generate_dataset(original_data,num_of_weeks=3,num_of_days=3,num_of_hours=7,num_of_predict=12,points_per_hours=12)
# data_set=generate_train_val_test_set(output,shuffle=True)
# data=MinMaxnormalization(*data_set)
# print(f'train_set:{data[1][0].shape},{data[1][1].shape}'
#       f'train_set[0,0]:{data[1][0][0,0]}')

#####################################################


def data_procession(data_file_path, adja_path, num_of_weeks, num_of_hours, num_of_predict, num_of_days,num_of_points,
                    points_per_hours=12,index_file=None,dtype=np.float32,
                    shuffle=False, directed=False, merge=False, ratio=(6, 8, 10), save=False):
    '''

    :param data_file_path: str,where to load data
    :param adja_path: str ,where to load the adja data
    :param num_of_weeks:
    :param num_of_hours:
    :param num_of_predict:
    :param num_of_days:
    :param num_of_points: int,the number of the total nodes of the input adja
    :param points_per_hours:
    :param index_file:
    :param shuffle:boolean
    :param directed:boolean
    :param merge:boolean
    :param ratio:tuple
    :param save:boolean
    :return:
    '''
    original_data = load_data_from_file(data_file_path,dtype=dtype)
    norm_adja = get_adja_matrix(data_path=adja_path,index_file=index_file,num_of_points=num_of_points)

    dataset: list = generate_dataset(original_data, num_of_weeks, num_of_days, num_of_hours, num_of_predict,
                                     points_per_hours)
    before_normalization = generate_train_val_test_set(dataset, ratio, shuffle, merge)
    if len(before_normalization) == 2:
        stats, norm_set = MinMaxnormalization(before_normalization[0], before_normalization[1])
        norm_dataset = {
            'stats':
                stats,
            'norm_adja':
                norm_adja,
            'train_norm_set':
                {
                    'X': norm_set[0][0],    #(num_of_datas,T,N,d)
                    'y': norm_set[0][1][:, :, :, 0] #(num_of_datas,T,N),也是被归一化的
                },
            'test_norm_set':
                {
                    'X': norm_set[1][0],
                    'y': norm_set[1][1][:, :, :, 0]
                }

        }

    else:
        stats, norm_set = MinMaxnormalization(train=before_normalization[0], val=before_normalization[1],
                                              test=before_normalization[2])
        print('1')
        norm_dataset = {
            'stats':
                stats,
            'norm_adja':
                norm_adja,
            'train_norm_set':
                {
                    'X': norm_set[0][0],
                    'y': norm_set[0][1][:, :, :, 0]
                },
            'val_norm_set':
                {
                    'X': norm_set[1][0],
                    'y': norm_set[1][1][:, :, :, 0]
                },
            'test_norm_set':
                {
                    'X': norm_set[2][0],
                    'y': norm_set[2][1][:, :, :, 0]
                }

        }
    if save:
        file = os.path.basename(data_file_path).split('.')[0]
        dir_path = os.path.dirname(data_file_path)
        # 等价于os.path.split(data_file_path)[0]
        filename = os.path.join(dir_path, file + f'_r{num_of_hours}_d{num_of_days}_w{num_of_weeks}_norm_data_max_value_min_value_norm_adja_directed_({directed})')
        print('save file', filename)
        # if os.path.exists(r'{}'.format(filename)):
        #     os.remove(path=filename)
        #     print(f'remove:{filename}')
        if len(before_normalization) == 2:
            np.savez_compressed(filename,
                                train_x=norm_dataset['train_norm_set']['X'], train_y=norm_dataset['train_norm_set']['y'],
                                test_x=norm_dataset['test_norm_set']['X'], test_y=norm_dataset['test_norm_set']['y'],
                                norm_adja=norm_adja,max_value=stats['max_value'],min_value=stats['min_value']
                                )
        else:
            np.savez_compressed(filename,
                                train_x=norm_dataset['train_norm_set']['X'], train_y=norm_dataset['train_norm_set']['y'],
                                val_x=norm_dataset['val_norm_set']['X'], val_y=norm_dataset['val_norm_set']['y'],
                                test_x=norm_dataset['test_norm_set']['X'], test_y=norm_dataset['test_norm_set']['y'],
                                norm_adja=norm_adja, max_value=stats['max_value'], min_value=stats['min_value']
                                )
            print('end')
    return norm_dataset, norm_dataset['train_norm_set']['X'].shape[0]

############################################

## check data_procession
# norm_dataset,num_of_nodes=data_procession(data_file_path=r"D:\i'a'm'z'k'l\AI\recurence\ASGNN\data\PEMS03\PEMS03.npz",adja_path=r"D:\i'a'm'z'k'l\AI\recurence\ASGNN\data\PEMS03\PEMS03.csv",
#                                    num_of_points=358,num_of_predict=12,num_of_hours=7,num_of_weeks=3,num_of_days=3,points_per_hours=12,
#                                    index_file=r"D:\i'a'm'z'k'l\AI\recurence\ASGNN\data\PEMS03\PEMS03.txt",save=True)
# print(f'num_of_nodes:{num_of_nodes}')

##############################################










# MinMaxnormalization()

# torch.save()

def min_max_normalize(data, _min, _max):
    '''

    :param data: numpy.ndarray,(b,T,N)
    :param _min: (1,1,1)
    :param _max:
    :return:
    '''
    data = 2. * (data - _min) / _max - 1.
    return data


def get_encoder_decoder_input_target(dataset, num_of_encoder_input=1, dtype=np.float32):
    '''

    :param dataset: dataset from the data_preprocession
    :param num_of_encoder_input:
    :return:
    '''
    _max, _min = dataset['stats'].values()
    if len(dataset) == 5:
        # train_data
        train_data = dataset['train_norm_set']
        # (b,T,N,d)
        train_x = train_data['X'][:, :, :, 0:num_of_encoder_input]
        train_y = train_data['y']
        # normalize the y
        train_y = min_max_normalize(train_y, _min, _max)

        # val_data
        val_data = dataset['val_norm_set']
        # (b,T,N,d)
        val_x = val_data['X'][:, :, :, 0:num_of_encoder_input]
        val_y = val_data['y']
        # normalize the y
        val_y = min_max_normalize(val_y, _min, _max)

        # test_data
        test_data = dataset['test_norm_set']
        # (b,T,N,d)
        test_x = test_data['X'][:, :, :, 0:num_of_encoder_input]
        test_y = test_data['y']
        # normalize the y
        test_y = min_max_normalize(test_y, _min, _max)

        # get encoder and decoder input for the model
        train_decoder_input = np.concatenate([train_x[:, :, -1:], train_y], axis=-1).astype(dtype=dtype)
        val_decoder_input = np.concatenate([val_x[:, :, -1:], val_y], axis=-1).astype(dtype=dtype)
        test_decoder_input = np.concatenate([test_x[:, :, -1:], test_y], axis=-1).astype(dtype)

        train_x = torch.from_numpy(train_x.astype(dtype))
        train_y = torch.from_numpy(train_y.astype(dtype))
        train_decoder_input = torch.from_numpy(train_decoder_input)

        val_x = torch.from_numpy(val_x.astype(dtype))
        val_y = torch.from_numpy(val_y.astype(dtype))
        val_decoder_input = torch.from_numpy(val_decoder_input)

        test_x = torch.from_numpy(test_x.astype(dtype))
        test_y = torch.from_numpy(test_y.astype(dtype))
        test_decoder_input = torch.from_numpy(test_decoder_input)
        dataset['stats']['max_value'] = torch.from_numpy(dataset['stats']['max_value'].astype(dtype))
        dataset['stats']['min_value'] = torch.from_numpy(dataset['stats']['min_value'].astype(dtype))
        dataset['norm_adja'] = torch.from_numpy(dataset['norm_adja'].astype(dtype))
        dataset['train_norm_set'] = {'X': train_x, 'y': train_y, 'train_decoder_input': train_decoder_input}
        dataset['val_norm_set'] = {'X': val_x, 'y': val_y, 'val_decoder_input': val_decoder_input}
        dataset['test_norm_set'] = {'X': test_x, 'y': test_y, 'test_decoder_input': test_decoder_input}

        return dataset
    # torch.FloatTensor ,

    else:
        # train_data
        train_data = dataset['train_norm_set']
        # (b,T,N,d)
        train_x = train_data['X'][:, :, :, 0:num_of_encoder_input]
        train_y = train_data['y']
        # normalize the y
        train_y = min_max_normalize(train_y, _min, _max)

        # test_data
        test_data = dataset['test_norm_set']
        # (b,T,N,d)
        test_x = test_data['X'][:, :, :, 0:num_of_encoder_input]
        test_y = test_data['y']
        # normalize the y
        test_y = min_max_normalize(test_y, _min, _max)

        # get encoder and decoder input for the model
        train_decoder_input = np.concatenate([train_x[:, :, -1:], train_y], axis=-1).astype(dtype=dtype)

        test_decoder_input = np.concatenate([test_x[:, :, -1:], test_y], axis=-1).astype(dtype)  # (b,T,1,1)

        train_x = torch.from_numpy(train_x.astype(dtype))
        train_y = torch.from_numpy(train_y.astype(dtype))
        train_decoder_input = torch.from_numpy(train_decoder_input)

        test_x = torch.from_numpy(test_x.astype(dtype))
        test_y = torch.from_numpy(test_y.astype(dtype))
        test_decoder_input = torch.from_numpy(test_decoder_input)

        dataset['stats']['max_value'] = torch.from_numpy(dataset['stats']['max_value'].astype(dtype))
        dataset['stats']['min_value'] = torch.from_numpy(dataset['stats']['min_value'].astype(dtype))
        dataset['norm_adja'] = torch.from_numpy(dataset['norm_adja'].astype(dtype))
        dataset['train_norm_set'] = {'X': train_x, 'y': train_y, 'train_decoder_input': train_decoder_input}
        dataset['test_norm_set'] = {'X': test_x, 'y': test_y, 'test_decoder_input': test_decoder_input}
        return dataset


def re_min_max_normalize(data, _min, _max):
    data = (data + 1) / 2
    data = data * (_max - _min) + _min
    return data


# np.savetxt
# os.path.basename

## get configs of the layers of model

# config_of_SDGCN = {
#     'd_model': None,
#     'norm_adja': None,
#     'dropout': None,
#     'use_ln': None,
#     'use_res': None,
#     'dtype': None,
#
# }

# config_of_TrSelf = {
#     'd_model': None,
#     'n_head': None,
#     'q_norm_conv': None,
#     'k_norm_conv': None,
#     'kernel_size': None,
#     'out_channels': None,
#     'stride': None,
#     'padding': None,
#
#     'num_of_weeks': None,
#     'num_of_days': None,
#     'num_of_hours': None,
#     'points_per_hour': None,
#
#     'use_ln': None,
#     'use_res': None,
#     'dropout': None,
#
#     'mask': None,
#     'dtype': None,
#
# }
#
# config_of_MHSelf = {
#     'd_model': None,
#     'd_head': None,
#     'device': None,
#     'droupout': None,
#     'use_ln': None,
#     'use_res': None,
#     'dtype': None,
#     'mask': None
#
# }

# encoder 与encoderlayer 是一致的，也就是层数的关系,decoder与encoder是一样的
# configs_of_encoderlayer = {
#     'TrSelf':
#         {
#
#         },
#     'MHSelf':
#         {
#
#         },
#     'SDGCN':
#         {
#
#         }
# }

from copy import deepcopy


def get_config_of_layers(args, norm_adja):
    d = args.d
    dropout = args.dropout
    use_ln = args.use_ln
    use_res = args.use_res
    dtype = args.dtype
    config_of_SDGCN = {
        'd': d,
        'norm_adja': norm_adja,
        'dropout': dropout,
        'use_ln': use_ln,
        'use_res': use_res,
        'dtype': dtype,
    }

    n_head = args.n_head
    q_norm_conv = args.q_norm_conv
    k_norm_conv = args.k_norm_conv
    kernel_size = (1, args.kernel_size)
    out_channels = args.out_channels
    stride = (1, args.stride)
    padding = args.paddiing  # 有待考究
    num_of_weeks = args.num_of_weeks
    num_of_days = args.num_of_days
    num_of_hours = args.num_of_hours
    points_per_hour = args.points_per_hour
    mask = args.mask

    config_of_TrSelf = {
        'd': d,
        'n_head': n_head,
        'q_norm_conv': q_norm_conv,
        'k_norm_conv': k_norm_conv,
        'kernel_size': kernel_size,
        'out_channels': out_channels,
        'stride': stride,
        'padding': padding,

        'num_of_weeks': num_of_weeks,
        'num_of_days': num_of_days,
        'num_of_hours': num_of_hours,
        'points_per_hour': points_per_hour,

        'use_ln': use_ln,
        'use_res': use_res,
        'dropout': dropout,

        'mask': mask,
        'dtype': dtype,

    }

    config_of_MHSelf = {
        'd': d,
        'n_head': n_head,
        'dropout': dropout,
        'use_ln': use_ln,
        'use_res': use_res,
        'dtype': dtype,
        'mask': mask

    }

    configs_of_encoderlayer = {
        'TrSelf':
            config_of_TrSelf,
        'MHSelf':
            config_of_MHSelf
        ,
        'SDGCN':
            config_of_SDGCN
    }

    configs_of_decoderlayer = deepcopy(configs_of_encoderlayer)
    congigs_of_decoder = configs_of_decoderlayer
    configs_of_encoderlayer = configs_of_encoderlayer

    return configs_of_encoderlayer, configs_of_encoderlayer


from collections import deque
import time


def evalueta_model(model, data_loader, loss_fn, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                   writer=None):
    '''

    :param model:
    :param data: (X,y,decoder_input)
    :return:
    '''
    model.eval()  # 等价于model.train(False),dropout的缘故
    pbar = tqdm(iterable=data_loader, desc='evaluate the model', total=len(data_loader))
    with torch.no_grad():
        model.to(device)
        loss = 0.
        loss_history = deque()
        start_time = time.time()
        for it, data in enumerate(pbar):
            X, y, decoder_input = data
            X = X.to(device)
            y = y.to(device)
            decoder_input = decoder_input.to(device)
            # y_hat = model(X,decoder_input)
            # loss+=loss_fn(y_hat,y).item()
            # pbar.set_description(desc=f'it:{it}')
            predict_length = y.shape[1]
            encoder_output = model.encode(X)
            decoder_start_inputs = decoder_input[:, :1, :, :]
            decoder_input_list = [decoder_start_inputs]

            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=1)
                predit_output = model.decode(decoder_inputs,
                                             encoder_output)  # (b,1,N,1)->(b,2,N,1)->(b,3,N,1)->...(b,T,N,1)
                decoder_input_list = [decoder_start_inputs, predit_output]
                # pbar.set_description(desc=f'it:{it},step:{step}')
            loss = loss_fn(predit_output, y)
            loss_history.append(loss.item())
            if it % 100 == 0:
                pbar.set_description(desc='it:{it+1},loss:{loss.item():.3f}')
            print(f'validation cost time:{time.time() - start_time:.3f}s')
            validation_loss = sum(loss_history) / len(loss_history)


from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


# 直接copy的
def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                                y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


# torch.nn.Module.load_state_dict()
# torch.save()
def predict_and_save_result(model, dataloader, num_for_predict, _min, _max, save_prediction_to, label_of_data=None,
                            param_of_model=None):
    if param_of_model is not None:
        model.load_state_dict(torch.load(param_of_model))

    with torch.no_grad():
        start_time = time.time()
        pbar = tqdm(iterable=dataloader, desc='Predicting', total=len(dataloader))
        prediction_history = deque()
        for it, encoder_inputs in enumerate(pbar):
            encoder_ouput = model.encode(encoder_inputs)
            decoder_inputs = encoder_inputs[:, :1, :, :]
            for i in range(num_for_predict):
                decoder_outputs = model.decode(decoder_inputs)
                decoder_inputs = torch.cat([encoder_inputs[:, :1, :, :], decoder_outputs], dim=1)

            prediction = decoder_inputs.cpu().detach().numpy()
            prediction_history.append(prediction)
            pbar.set_description(f'it:{it + 1}')
            if it % 100 == 0:
                print(f'the time of the prediction:{time.time() - start_time:.3f}s')

        print(f'the total time is:{time.time() - start_time:0.3f}s')
        # 评估结果的性能
        if save_prediction_to is None:
            file_name = 'prediction_of_model'
            save_prediction_to = os.path.abspath(file_name)

        prediction_array = np.concatenate(prediction_history, axis=0)
        if label_of_data is None:
            # 返回原来的结果
            original_prediction = re_max_min_normalization(prediction_array, _max[:, :, :, 0:1], _min[:, :, :, 0:1])
            np.savez(save_prediction_to, prediction=original_prediction)

        else:
            labels = label_of_data.cpu().detach().numpy()
            original_prediction = re_max_min_normalization(prediction_array, _max[:, :, :, 0:1], _min[:, :, :, 0:1])
            loss_of_each = deque()
            for i in range(prediction_array.shape[2]):
                mae = masked_mape_np(label_of_data[:, :, i, :], prediction_array[:, :, i, :])
                rmse = mean_squared_error(label_of_data[:, :, i, :], prediction_array[:, :, i, :]) ** 0.5
                mape = masked_mape_np(label_of_data[:, :, i, :], prediction_array[:, :, i, :])
                print(f'the loss of point_{i}:\nrmse:{rmse:3f},mae:{mae:3f},mape:{mape:3f}')
                loss_of_each.append([rmse, mae, mape])

            mas = mean_absolute_error(labels, prediction)
            mae = masked_mape_np(y_true=labels, y_pred=prediction)
            rmse = mean_squared_error(labels, prediction) ** 0.5
            print(f'the total rmse:{rmse:3f},mas:{mas:3f},mae:{mae:3f}')

            np.savez(save_prediction_to, prediction=original_prediction)
