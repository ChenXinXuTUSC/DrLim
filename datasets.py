import torch
import numpy as np

class FashionMnist(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: str="train",
        transforms=None
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.transforms = transforms

        npzdata = np.load(f"{self.root}/{self.split}.npz")
        self.images = torch.from_numpy(npzdata["images"]).unsqueeze(dim=1).float() # [n,h,w]=>[n,c,h,w]
        self.labels = torch.from_numpy(np.reshape(npzdata["labels"], (len(npzdata["labels"]), 1))).int()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        for T in self.transforms:
            image = T(image)

        return image, label

class Cifar10(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: str="train",
        transforms=None
    ) -> None:
        self.root = root

    
    def unpickle(self, path):
        import pickle
        with open(path, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        return data_dict
    
    
    def
