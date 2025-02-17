from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        data, target = self.subset[index]
        if self.transform:
            data = self.transform(data)
        return data, target

    def __len__(self):
        return len(self.subset)