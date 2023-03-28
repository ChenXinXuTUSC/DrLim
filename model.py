import torch
import torch.nn as nn
from torchsummary import summary

import utils

class DrLim(nn.Module):
    def __init__(self, in_channels=1, out_channels=2) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=7, stride=2, padding=3, bias=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2, bias=True)
        self.fucn1 = nn.Linear(32, out_channels, bias=True)
        self.norm1 = nn.BatchNorm2d(16)
        self.norm2 = nn.BatchNorm2d(32)
        self.pool1 = nn.AdaptiveAvgPool2d((1,1))
        self.activ = nn.ReLU(inplace=True)

        self.init_weights()
    
    def init_weights(self):
        def init_module(m):
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.kaiming_normal_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            if isinstance(m, nn.Linear):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.normal_(m.weight, mean=0.0, std=1.0)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            if isinstance(m, nn.BatchNorm2d):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.normal_(m.weight, mean=0.0, std=1.0)
        
        self.apply(init_module)
    
    def forward(self, x: torch.Tensor):
        x = self.norm1(self.activ(self.conv1(x)))
        x = self.norm2(self.activ(self.conv2(x)))
        x = self.pool1(x).reshape(x.size(0), -1)
        out = self.fucn1(x)
        # do not apply softmax or sigmoid, the output
        # does not stand for probalistic, but coordi-
        # nates on manifold in x-dimesion.
        return out
    
    def ctloss(self, manifold_coords: torch.Tensor, labels: torch.Tensor):
        '''
        perform correlation computation between samples
        |(x1,x1) (x1,x2) (x1,x3) ... (x1,xn)|
        |(x2,x1) (x2,x2) (x2,x3) ... (x2,xn)|
        |   .       .       .           .   |
        |   .       .       .           .   |
        |(xn,x1) (xn,x2) (xn,x3) ... (xn,xn)|
        '''
        # this makes each sample compare to every other
        # sample in the same batch, if t hey  have  the
        # same label, mask at this position  should  be
        # true.
        
        # (n,c)=>(n,1,c) and (n,c)=>(1,n,c), operations
        # between (1,n,c) and  (n,1,c)  will  boradcast
        # automatically.
        num_samples = len(labels)
        eq_mask = labels.unsqueeze(dim=0) == labels.unsqueeze(dim=1) # torch.BoolTensor
        mutual1 = manifold_coords.unsqueeze(dim=0).repeat(num_samples, 1, 1)
        mutual2 = manifold_coords.unsqueeze(dim=1).repeat(1, num_samples, 1)

        manifold_dists = torch.norm((mutual1 - mutual2), dim=-1, p=2, keepdim=True)

        loss_same = torch.sum(manifold_dists[eq_mask] ** 2)
        loss_diff = torch.sum(torch.clamp(1.0 - manifold_dists[~eq_mask], min=0) ** 2)
        # loss is contributed by distance of every sample
        # don't forget to redistribute it.
        loss_totl = (loss_same + loss_diff) / (num_samples * (num_samples - 1))
        return loss_totl

if __name__ == "__main__":
    '''
    output information of model structure
    '''
    model = DrLim(1, 2)
    summary(model, input_data=(1, 32, 32), batch_dim=0)
