import os
import argparse
import time
from tqdm import tqdm

import torch
from torchvision import transforms
from tensorboardX import SummaryWriter
import utils
import datasets
import model

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True, help="path to the root dir of dataset")
parser.add_argument("--tfxw_root", type=str, default=f"{utils.PROJECT_SOURCE_DIR}/runs/tfxw", help="dir to store metric logs")
parser.add_argument("--stat_root", type=str, default=f"{utils.PROJECT_SOURCE_DIR}/runs/stat", help="dir to store model state dict")
parser.add_argument("--stat_dict", type=str, help="existing state dict that can be resumed")
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_root = args.data_root
num_epochs = args.num_epochs
batch_size = args.batch_size
lr = args.lr

timestamp = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
tfxw_root = f"{args.tfxw_root}/{timestamp}_e{num_epochs}_b{batch_size}_lr{lr:.2f}"
tfxw = SummaryWriter(tfxw_root)
stat_root = f"{args.stat_root}/{timestamp}_e{num_epochs}_b{batch_size}_lr{lr:.2f}"
if not os.path.exists(stat_root):
    os.makedirs(stat_root, mode=0o775)

trainset = datasets.FashionMnist(
    root=data_root,
    split="train",
    transforms=[transforms.Resize([32 ,32], antialias=False)]
)
testset = datasets.FashionMnist(
    root=data_root,
    split="test",
    transforms=[transforms.Resize([32, 32], antialias=False)]
)
trainloader = torch.utils.data.DataLoader(
    dataset=trainset,
    batch_size=batch_size,
    shuffle=True
)
testloader = torch.utils.data.DataLoader(
    dataset=testset,
    batch_size=batch_size,
    shuffle=True
)

classifer = model.DrLim(1, 2)
optimizer = torch.optim.Adam(classifer.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
if args.stat_dict is not None:
    classifer.load_state_dict(torch.load(args.stat_dict))


classifer.to(device)
classifer.train()
for epoch in range(1, num_epochs + 1):
    for iter, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), ncols=100, desc=f"{utils.redd('train')} {epoch:3d}/{num_epochs}"):
        images = images.to(device)
        labels = labels.to(device)
        output = classifer(images)

        optimizer.zero_grad()

        loss = classifer.ctloss(output, labels)
        loss.backward()

        optimizer.step()

        # metric logs
        if iter % 100 == 0:
            tqdm.write(f"{utils.redd('loss')}: {loss.item():.3f}")
            tfxw.add_scalar(tag="train/loss", scalar_value=loss.item(), global_step=epoch * len(trainloader) + iter)
    if epoch % 10 == 0:
        torch.save(classifer.state_dict(), f"{stat_root}/{epoch}.pth")
    scheduler.step()
