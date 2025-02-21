from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class MyDataset(Dataset):
    '''
    dataset:(train_X,train_y),train_x:(num_of_samples,N,C)
    '''

    def __init__(self, dataset):
        self.data = dataset

    def __getitem__(self, item):
        return self.data[0][item], self.data[1][item]

    def __len__(self):
        return self.data[0].shape[0]


def mi():
    a = {'3': 3}
