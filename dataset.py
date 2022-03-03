import torch
from torch.utils.data import Dataset

class TimeGANDataset(Dataset):
    def __init__(self, time_features):
        self.features = torch.FloatTensor(time_features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], torch.LongTensor(self.features[idx].size()[1])
