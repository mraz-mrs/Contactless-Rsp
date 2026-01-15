import numpy as np
import os
import torch
from torch.utils.data import Dataset

class Xinchuan_Dataset(Dataset):
    def __init__(self, configs, split = 'train', transform=None):
        self.data_root = configs.DATA.ROOT
        self.transform = transform
        self.data_list = os.listdir(self.data_root)
        # 过滤出stmap和rsp文件
        self.stamap_npy = 'stmap.npy'
        self.data_list = [os.path.join(self.data_root, f) for f in self.data_list if f.endswith(self.stamap_npy)]

        data_split = configs.DATA.SPLIT

        if split == 'train':
            self.data_list = self.data_list[:int(len(self.data_list) * data_split)]
        elif split == 'val':
            self.data_list = self.data_list[int(len(self.data_list) * data_split):]

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_path = self.data_list[index]
        stmap = np.load(data_path)
        stmap = np.transpose(stmap, (2, 0, 1))
        rsp_path = data_path.replace(self.stamap_npy, 'rsp.npy')
        rsp = np.load(rsp_path)

        if self.transform:
            stmap = self.transform(stmap)
        return torch.from_numpy(np.float32(stmap)), torch.from_numpy(np.float32(rsp))

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    class config:
        def __init__(self):
            class DATA:
                ROOT = 'H:/chunks/meta_data/RPPG'
                SPLIT = 0.8
            self.DATA = DATA()
    configs = config()
    train_dataset = Xinchuan_Dataset(configs, split='train')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    for stmap, rsp in train_loader:
        print(stmap.shape, rsp.shape)