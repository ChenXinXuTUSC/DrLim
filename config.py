import argparse

import utils

parser = argparse.ArgumentParser()
# configuration dataset
parser.add_argument("--data_type", type=str, required=True, help="dataset type, [FashionMnist | Cifar10]")
parser.add_argument("--data_root", type=str, required=True, help="path to the root dir of dataset")
parser.add_argument("--data_splt", type=str, default="train", help="which split of the dataset")
parser.add_argument("--tfxw_root", type=str, default=f"{utils.PROJECT_SOURCE_DIR}/runs/tfxw", help="dir to store metric logs")
parser.add_argument("--stat_root", type=str, default=f"{utils.PROJECT_SOURCE_DIR}/runs/stat", help="dir to store model state dict")
parser.add_argument("--rslt_root", type=str, default=f"{utils.PROJECT_SOURCE_DIR}/results", help="dir to store visualization resutls")
# configuration model
parser.add_argument("--loss_type", type=str, help="available loss type to choose: [Contrastive|Triplet]")
parser.add_argument("--stat_dict", type=str, help="existing state dict that can be resumed")
parser.add_argument("--in_channels", type=int, default=1, help="input space dimension")
parser.add_argument("--out_channels", type=int, default=2, help="output space dimension")
# configuration loader & optimizer & scheduler
parser.add_argument("--trans_rsiz", type=int, default=32, help="length of resized squared image")
parser.add_argument("--trans_gray", action="store_true", help="whether to transform rgb to gray scale")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
# configuration training
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--save_freq", type=int, default=10, help="state dict saving frequency")
parser.add_argument("--info_freq", type=int, default=100, help="info printing frequency")
args = parser.parse_args()
