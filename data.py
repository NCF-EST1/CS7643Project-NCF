import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, label_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.label_tensor = label_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.label_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)