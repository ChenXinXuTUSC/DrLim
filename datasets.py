import torch
import torchvision
import numpy as np

class FashionMnist(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str,
            split: str="train",
            transforms: torchvision.transforms=None,
            misc_args=None
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
            transforms: torchvision.transforms=None,
            misc_args=None
        ) -> None:
        self.root = root
        self.transforms = transforms

        data = np.load(f"{root}/npz/{split}_batch.npz")
        self.images = data["images"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        for T in self.transforms:
            image = T(image)
        
        return image, label

VALID_DATASETS=[
    FashionMnist, Cifar10
]

def make_torchloader(
        data_type: str,
        data_root: str,
        split: str,
        transforms: torchvision.transforms,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        misc_args
    ):
    dataset_str_map = {d.__name__:d for d in VALID_DATASETS}
    if data_type not in dataset_str_map:
        raise Exception(f"{data_type} not implement yet...")
    dataset = dataset_str_map[data_type]

    thisset = dataset(
        root=data_root,
        split=split,
        transforms=transforms,
        misc_args=misc_args
    )

    thisloader = torch.utils.data.DataLoader(
        dataset=thisset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

    return thisloader
