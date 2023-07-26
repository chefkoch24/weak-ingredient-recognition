from torch.utils.data import Dataset
import torch
class MultiTaskDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return {key: torch.tensor(self.data[key][index]) for key in self.data.keys()}

    def __len__(self):
        return len(self.data['input_ids'])

    def num_token_classes(self):
        return 7 # 0: not ingredient, 1: ingredient

    def num_classification_classes(self):
        return len(self.data['classification_labels'][0])
