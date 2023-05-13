import os
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

trans = [transforms.Resize([args.trans_rsiz, args.trans_rsiz], antialias=False)]
if args.trans_gray:
    trans.append(transforms.Grayscale(num_output_channels=1))
trainloader = datasets.make_torchloader(
    data_type=data_type,
    data_root=data_root,
    split=data_splt,
    transforms=trans,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    misc_args=args
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# classifer = model.Contrastive(args.in_channels, args.out_channels)
classifer = model.load_model(args.loss_type)(args.in_channels, args.out_channels)
optimizer = torch.optim.Adam(classifer.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
if args.stat_dict is not None:
    classifer.load_state_dict(torch.load(args.stat_dict))


timestamp = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
logd_root = os.path.join(
    args.logd_root, 
    f"{data_type}@{classifer.__class__.__name__}_i{args.in_channels}o{args.out_channels}e{args.num_epochs}b{args.batch_size}lr{args.lr:.2f}_{timestamp}"
)
tfxw_root = os.path.join(logd_root, "tfxw")
tfxw = SummaryWriter(tfxw_root)
stat_root = os.path.join(logd_root, "stat")
if not os.path.exists(stat_root):
    os.makedirs(stat_root, mode=0o775)


num_epochs = args.num_epochs
save_freq  = args.save_freq
info_freq  = args.info_freq
classifer.to(device)
classifer.train()
utils.log_info(f"training on {device}")
for epoch in range(1, num_epochs + 1):
    for iter, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), ncols=100, desc=f"{utils.redd('train')} {epoch:3d}/{num_epochs}"):
        images = images.to(device)
        labels = labels.to(device)
        output = classifer(images)

        optimizer.zero_grad()

        loss = classifer.loss(output, labels, iter % 1000 == 0)
        loss.backward()

        optimizer.step()

        # metric logs
        if iter % info_freq == 0:
            tqdm.write(f"{utils.redd('loss')}: {loss.item():.3f}")
            tfxw.add_scalar(tag="train/loss", scalar_value=loss.item(), global_step=epoch * len(trainloader) + iter)
    if epoch % save_freq == 0:
        torch.save(classifer.state_dict(), f"{stat_root}/{epoch}.pth")
    
    scheduler.step()
