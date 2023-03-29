import os
from tqdm import tqdm

import torch
from torchvision import transforms
import numpy as np

import config
import datasets
import model
import utils

def draw_points_2d(
    x: np.ndarray,
    y: np.ndarray,
    id2cls: dict,
    out_root: str,
    out_name: str
):
    import matplotlib.pyplot as plt
    import matplotlib

    cmap1 = matplotlib.colormaps["plasma"]
    cmap2 = matplotlib.colormaps["viridis"]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=9)
    fig = plt.figure(figsize=plt.figaspect(0.4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for id, cls in id2cls.items():
        clsmask = np.nonzero((y[:, 0] == id))
        # len(color) must match len(coords), to assign each point a color
        # 'color' kwarg must be a color or sequence of color specs. For a 
        # sequence of values to be color-mapped,  use  the  'c'  argument
        ax1.scatter(x[clsmask, 0], x[clsmask, 1], color=[cmap1(norm(id))]*len(clsmask), alpha=0.2, label=cls)
        ax1.title.set_text("DrLim-plasma")
        ax1.legend(loc="upper right")
        ax2.scatter(x[clsmask, 0], x[clsmask, 1], color=[cmap2(norm(id))]*len(clsmask), alpha=0.2, label=cls)
        ax2.title.set_text("DrLim-viridi")
        ax2.legend(loc="upper right")
    
    plt.legend(loc="upper right")
    plt.savefig(f"{out_root}/{out_name}.png")

def draw_points_3d(
    x: np.ndarray,
    y: np.ndarray,
    id2cls: dict,
    out_root: str,
    out_name: str
):
    import matplotlib.pyplot as plt
    import matplotlib

    cmap1 = matplotlib.colormaps["plasma"]
    cmap2 = matplotlib.colormaps["viridis"]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=9)
    fig = plt.figure(figsize=plt.figaspect(0.4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for id, cls in id2cls.items():
        clsmask = np.nonzero((y[:, 0] == id))
        # len(color) must match len(coords), to assign each point a color
        # 'color' kwarg must be a color or sequence of color specs. For a 
        # sequence of values to be color-mapped,  use  the  'c'  argument
        ax1.scatter(x[clsmask, 0], x[clsmask, 1], x[clsmask, 2], color=[cmap1(norm(id))]*len(clsmask), alpha=0.2, label=cls)
        ax1.title.set_text("DrLim-plasma")
        ax1.legend(loc="upper right")
        ax2.scatter(x[clsmask, 0], x[clsmask, 1], x[clsmask, 2], color=[cmap2(norm(id))]*len(clsmask), alpha=0.2, label=cls)
        ax2.title.set_text("DrLim-viridi")
        ax2.legend(loc="upper right")
    
    plt.legend(loc="upper right")
    plt.savefig(f"{out_root}/{out_name}.png")

def dump_points_3d(
    x: np.ndarray,
    y: np.ndarray,
    out_root: str,
    out_name: str
):
    import matplotlib.pyplot as plt
    import matplotlib
    cmap = matplotlib.colormaps["plasma"]
    norm = matplotlib.colors.Normalize(vmin=y.min(), vmax=y.max())
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    pcd.colors = o3d.utility.Vector3dVector(cmap(norm(y[:, 0]))[:, :3])
    o3d.io.write_point_cloud(f"{out_root}/{out_name}.ply", pcd)



if __name__ == "__main__":
    args = config.args

    if not os.path.exists(args.rslt_root):
        os.makedirs(args.rslt_root, mode=0o775)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testloader = datasets.make_torchloader(
        data_type=args.data_type,
        data_root=args.data_root,
        split=args.data_splt,
        transforms=[transforms.Resize([32, 32], antialias=False)],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        misc_args=args
    )

    classifier = model.DrLim(args.in_channels, args.out_channels)
    classifier.load_state_dict(torch.load(args.stat_dict))
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

    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    with open(f"{args.data_root}/classname.txt", 'r') as f:
        tagmap = {int(item[0]):item[1] for item in [line.strip().split(' ') for line in f.readlines()]}
    
    draw_points_3d(x, y, tagmap, args.rslt_root, f"{args.data_type}_i1o3")
    dump_points_3d(x, y, args.rslt_root, f"{args.data_type}_i1o3")
