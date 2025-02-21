
import shutil

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import matplotlib as mpl
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from mydataset import MyDataset
import os
from collections import deque
import numpy as np


def trainer(model, dataset, optimizer, loss_fn=torch.nn.MSELoss(), batch_size=12, start_epoch=0, epochs=100,
            save_params_to='params_of_model', loss_history_path='loss_history_path',
            device=('cuda' if torch.cuda.is_available() else 'cpu'), shuffle=False, use_distributed=True):
    dataset = MyDataset(dataset)
    # dataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle)
    if use_distributed:
        dist.init_process_group(backend='nccl')
        data_sampler = DistributedSampler(dataset=dataset, shuffle=shuffle)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=data_sampler)
        local_rank = int(os.environ['LOCAL_RANK'])
        model.to(local_rank)  # 先加载进入cuda,DDP中的self.device=list(module.parameters())[0].device
        model = DDP(module=model, device_ids=[local_rank],
                    output_device=local_rank)  # device_ids 可能是None，所以model.to必须先进行
        pbar = tqdm(data_loader, desc=f'start of process {dist.get_rank()}', total=len(data_loader))
        total_loss = deque()
        if dist.get_rank() == 0:
            if start_epoch == 0 and not (os.path.exists(save_params_to)):
                os.makedirs(save_params_to)
                print(f'create params directory:{save_params_to}')
            elif start_epoch == 0 and os.path.exists(save_params_to):
                shutil.rmtree(save_params_to)
                os.makedirs(save_params_to)
                print(f'delete the old one and create parmas directory:{save_params_to}')
            elif start_epoch > 0 and os.path.exists(save_params_to):
                print(f'train_params from the {save_params_to}')
            else:
                raise SystemExit('Wrong type of model')


        if start_epoch > 0:
            model.load_state_dict(torch.load(save_params_to))



        for epoch in range(epochs):
            l = deque()
            data_sampler.set_epoch(epoch)
            for it, data in enumerate(pbar):
                X, y = data
                X = X.to(local_rank)
                y = y.to(local_rank)
                y_hat = model(X, X)
                loss = loss_fn(y, y_hat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                l.append(loss.item())
                if dist.get_rank() == 0:
                    pbar.set_description(
                        f'epoch:{epoch},loss:{loss.item()},it:{it + 1},lr:{optimizer.param_group[0]["lr"]}')
            if dist.get_rank() == 0:
                total_loss.append(l)
        if dist.get_rank() == 0:
            loss_array = np.array(total_loss)
            np.save(loss_history_path, loss_array)
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), save_params_to)
            ax = plt.subplot()
            ax.plot(loss_array[0])
            plt.savefig(fname='the_loss_picture')

    else:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        model.to(device)
        pbar = tqdm(data_loader, total=len(data_loader))
        total_loss = deque()
        for epoch in range(epochs):
            l = deque()
            for it, data in enumerate(pbar):
                X, y = data
                X = X.to(device)
                y = y.to(device)
                y_hat = model(X, X)
                loss = loss_fn(y, y_hat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                l.append(loss.item())
                pbar.set_description(
                    f'epoch:{epoch},loss:{loss.item()},it:{it + 1},lr:{optimizer.param_group[0]["lr"]}')
            total_loss.append(l)
        loss_array = np.array(total_loss)
        np.save(loss_history_path, loss_array)
        torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), save_params_to)
        ax = plt.subplot()
        ax.plot(loss_array[0])
        plt.savefig(fname='the_loss_picture')


from configparser import ConfigParser
# a=torch.optim.Adam
