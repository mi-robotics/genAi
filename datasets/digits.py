from torch.utils.data import Dataset
import numpy as np
from sklearn.datasets import load_digits
from sklearn import datasets

class Digits(Dataset):
    """
    Scikit-Learn Digits dataset.
    8x8 pixels, each pixel is in the range 0 - 16
    """

    def __init__(self, mode='train', transforms=None):
        digits = load_digits()
        if mode == 'train':
            self.data = digits.data[:1000].astype(np.float32)
        elif mode == 'val':
            self.data = digits.data[1000:1350].astype(np.float32)
        else:
            self.data = digits.data[1350:].astype(np.float32)

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transforms:
            sample = self.transforms(sample)
        return sample
    


if __name__ == "__main__":
    ds = Digits(mode='train')
    print(np.max(ds.data))
