import os
from tqdm import tqdm
import time

import torch
from torchvision import transforms

import config
import datasets
import model
import utils


if __name__ == "__main__":
    args = config.args

    if not os.path.exists(args.rslt_root):
        os.makedirs(args.rslt_root, mode=0o775)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trans = [transforms.Resize([args.trans_rsiz, args.trans_rsiz], antialias=False)]
    if args.trans_gray:
        trans.append(transforms.Grayscale(num_output_channels=1))
    testloader = datasets.make_torchloader(
        data_type=args.data_type,
        data_root=args.data_root,
        split=args.data_splt,
        transforms=trans,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        misc_args=args
    )

    # classifier = model.Contrastive(args.in_channels, args.out_channels)
    classifier = model.load_model(args.loss_type)(args.in_channels, args.out_channels)
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
    
    timestamp = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
    if args.out_channels == 2:
        utils.draw_points_2d(x, y, tagmap, args.rslt_root, f"{timestamp}_{args.data_type}@{classifier.__class__.__name__}_i{args.in_channels}o{args.out_channels}")
    if args.out_channels == 3:
        utils.draw_points_3d(x, y, tagmap, args.rslt_root, f"{timestamp}_{args.data_type}@{classifier.__class__.__name__}_i{args.in_channels}o{args.out_channels}")
        utils.dump_points_3d(x, y, args.rslt_root, f"{args.data_type}@{classifier.__class__.__name__}_i{args.in_channels}o{args.out_channels}")
