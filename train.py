import os
import argparse
import time
from tqdm import tqdm

import torch
from torchvision import transforms
from tensorboardX import SummaryWriter

import config
import utils
import datasets
import model

args = config.args

data_type = args.data_type
data_root = args.data_root
data_splt = args.data_splt
batch_size = args.batch_size
num_workers = args.num_workers
trainloader = datasets.make_torchloader(
    data_type=data_type,
    data_root=data_root,
    split=data_splt,
    transforms=[transforms.Resize([32, 32], antialias=False)],
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    misc_args=args
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifer = model.DrLim(args.in_channels, args.out_channels)
optimizer = torch.optim.Adam(classifer.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
if args.stat_dict is not None:
    classifer.load_state_dict(torch.load(args.stat_dict))


timestamp = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
tfxw_root = f"{args.tfxw_root}/{timestamp}_{data_type}_i{args.in_channels}o{args.out_channels}e{args.num_epochs}b{args.batch_size}lr{args.lr:.2f}"
tfxw = SummaryWriter(tfxw_root)
stat_root = f"{args.stat_root}/{timestamp}_{data_type}_i{args.in_channels}o{args.out_channels}e{args.num_epochs}b{args.batch_size}lr{args.lr:.2f}"
if not os.path.exists(stat_root):
    os.makedirs(stat_root, mode=0o775)


num_epochs = args.num_epochs
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
