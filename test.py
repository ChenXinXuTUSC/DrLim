import argparse
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib

import torch
from torchvision import transforms
import numpy as np

import datasets
import model
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, required=True, help="path to the root dir of dataset")
parser.add_argument("--rslt_root", type=str, default=f"{utils.PROJECT_SOURCE_DIR}/results", help="dir to store visualization resutls")
parser.add_argument("--stat_dict", type=str, required=True, help="existing state dict that can be resumed")
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

data_root = args.data_root
rslt_root = args.rslt_root
if not os.path.exists(rslt_root):
    os.makedirs(rslt_root, mode=0o775)
stat_dict = args.stat_dict
batch_size = args.batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

testset = datasets.FashionMnist(
    root=data_root,
    split="test",
    transforms=[transforms.Resize([32, 32], antialias=False)]
)
testloader = torch.utils.data.DataLoader(
    dataset=testset,
    batch_size=batch_size,
    shuffle=True
)

classifier = model.DrLim(1, 2)
classifier.load_state_dict(torch.load(stat_dict))
classifier.eval()
classifier.to(device)

x = None
y = None
for images, labels in tqdm(testloader, total=len(testloader), ncols=100, desc=f"{utils.gren('testset')}"):
    images = images.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        output = classifier(images)
        if x is None:
            x = output
            y = labels
        else:
            x = torch.cat([output, x], dim=0)
            y = torch.cat([labels, y], dim=0)

plot_data = torch.cat([x, y], dim=1)

x = x.detach().cpu().numpy()
y = y.detach().cpu().numpy()

with open(f"{data_root}/classname.txt", 'r') as f:
    tagmap = {int(item[0]):item[1] for item in [line.strip().split(' ') for line in f.readlines()]}

cmap = matplotlib.cm.get_cmap("plasma")
norm = matplotlib.colors.Normalize(vmin=0, vmax=9)
ax = plt.figure()
plt.tight_layout()
for id, cls in tagmap.items():
    clsmask = np.nonzero((y[:, 0] == id))
    # len(color) must match len(coords), to assign each point a color
    # 'color' kwarg must be a color or sequence of color specs. For a 
    # sequence of values to be color-mapped,  use  the  'c'  argument
    plt.scatter(x[clsmask, 0], x[clsmask, 1], color=[cmap(norm(id))]*len(clsmask), label=cls)
# plt.scatter(x[:, 0], x[:, 1], c=y)
plt.legend(loc="upper right")
plt.savefig(f"{rslt_root}/result.png")
