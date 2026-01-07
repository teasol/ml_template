from torch.utils.data import Dataset

class MyDataset(Dataset):
    """
    Standard PyTorch Dataset for individual data loading.
    """
    def __init__(self, data_path):
        super().__init__()
        # Load data from data_path
        with open(data_path, 'r') as f:
            self.samples = [] 

    def __len__(self):
        # Return total number of samples
        return len(self.samples)

    def __getitem__(self, idx):
        # Load one sample
        sample = self.samples[idx]

        return sample