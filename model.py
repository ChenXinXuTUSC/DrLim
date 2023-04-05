import torch
import torch.nn as nn
from torchsummary import summary

class Contrastive(nn.Module):
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
    
    def loss(self, manifold_coords: torch.Tensor, labels: torch.Tensor, centeralized: bool=False):
        '''compute contrastive loss
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

        mutual_dists = torch.norm((mutual1 - mutual2), dim=-1, p=2, keepdim=True)

        loss_same = torch.sum(mutual_dists[eq_mask] ** 2)
        loss_diff = torch.sum(torch.clamp(1.0 - mutual_dists[~eq_mask], min=0) ** 2)
        
        # loss is contributed by distance of every sample
        # don't forget to redistribute it.
        ctaloss  = (loss_same + loss_diff) / (num_samples * (num_samples - 1))
        if centeralized:
            distloss = (manifold_coords ** 2).sum(dim=1).sqrt().mean()   # distant converge
            centloss = ((manifold_coords.mean(dim=1)) ** 2).sum().sqrt() # center to origin
            ctaloss = ctaloss + distloss * 1e-2 + centloss * 1e-3
        return ctaloss

class Triplet(Contrastive):
    def __init__(
        self, 
        in_channels=1, 
        out_channels=2,
        margin=5.0
    ) -> None:
        super().__init__(in_channels, out_channels)
        self.margin = margin
    
    def loss(self, manifold_coords: torch.Tensor, labels: torch.Tensor, centeralized: bool=False):
        '''triplet loss proposed in FaceNet CVPR 2015
        
        params
        ----------
        * manifold_coords: mapping results of the original input features
        * labels: labels of each sample in the batch
        
        return
        ----------
        * triplet loss
        '''
        eps = 1e-8
        # step1: get distance matrix
        distmat = self.euclidean_distmat(manifold_coords)
        
        # step2: compute triplet loss for all possible combinations
        #        a - anchor sample
        #        p - positive sample
        #        n - negative sample
        ap_dist = distmat.unsqueeze(dim=2)
        an_dist = distmat.unsqueeze(dim=1)
        trploss = ap_dist - an_dist + self.margin
        
        # step3: filter out invalid triplet by setting their values
        #        to zero
        valid_mask = self.valid_triplet_mask(labels)
        trploss = trploss * valid_mask
        trploss = torch.nn.functional.relu(trploss)
        
        # step4: compute scalar loss value  by  averaging  positive
        #        values
        num_positive_losses = (trploss > 0.0).float().sum()
        trploss = trploss.sum() / (num_positive_losses + eps)
        if centeralized:
            distloss = (manifold_coords ** 2).sum(dim=1).sqrt().mean()   # distant converge
            centloss = ((manifold_coords.mean(dim=1)) ** 2).sum().sqrt() # center to origin
            trploss = trploss + distloss * 1e-2 + centloss * 1e-3
        
        return trploss

    def euclidean_distmat(self, batch: torch.Tensor):
        '''efficiently compute Euclidean distance matrix 
        
        params
        ----------
        * batch: input tensor of shape(batch_size, feature_dims)

        return
        ----------
        * Distance matrix of shape (batch_size, batch_size) 
        '''
        eps = 1e-8
        
        # step1: compute self cartesian product
        self_product = torch.mm(batch, batch.T)
        
        # step2: extract the squared euclidean distance of each
        #        sample from diagonal
        squared_diag = torch.diag(self_product)
        
        # step3: compute squared euclidean dists using the formula
        #        (a - b)^2 = a^2 - 2ab + b^2
        distmat = squared_diag.unsqueeze(dim=0) - 2*self_product + squared_diag.unsqueeze(dim=1)
        
        # get rid of negative distances due to numerical instabilities
        distmat = torch.nn.functional.relu(distmat)
        
        # step4: take the squared root of distance matrix and handle
        #        the numerical instabilities
        mask = (distmat == 0.0).float()
        
        # use the zero-mask to set those zero values to epsilon
        distmat = distmat + eps * mask
        
        distmat = torch.sqrt(distmat)
        
        # undo the trick for numerical instabilities
        # do not use *= operator, as it's an inplace operation
        # which will break the backward chain on sqrt function
        distmat = distmat * (1.0 - mask)
        
        return distmat

    def valid_triplet_mask(self, labels: torch.Tensor):
        '''efficiently compute valid triplet mask
        
        params
        ----------
        * labels: labels of samples in shape(batch_size, label_dims)
        
        return
        ----------
        * mask: valid triplet mask in shape(batch_size, batch_size, 
            batch_size).
            A triplet is valid only if  labels[i]  ==  labels[j]  &&
            labels[j] != labels[k] and indices 'i', 'j', 'k' are not
            the same value.
        '''
        
        # step1: mask of unique indices
        indices_eql = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        indices_neq = torch.logical_not(indices_eql)
        
        i_neq_j = indices_neq.unsqueeze(dim=2)
        i_neq_k = indices_neq.unsqueeze(dim=1)
        j_neq_k = indices_neq.unsqueeze(dim=0)
        
        indices_unq = torch.logical_and(i_neq_j, torch.logical_and(i_neq_k, j_neq_k))
        
        # step2: mask of valid triplet(labels[i],labels[j],labels[k])
        labels_eql = (labels.unsqueeze(dim=0) == labels.unsqueeze(dim=1)).squeeze()
        li_eql_lj = labels_eql.unsqueeze(dim=2)
        li_eql_lk = labels_eql.unsqueeze(dim=1)
        labels_vld = torch.logical_and(li_eql_lj, torch.logical_not(li_eql_lk))
        
        return torch.logical_and(indices_unq, labels_vld)

ALL_MODELS = [Contrastive, Triplet]
def load_model(model_name: str):
    mdict = {m.__name__:m for m in ALL_MODELS}
    if model_name not in mdict:
        raise Exception("model not implemented")
    return mdict[model_name]

if __name__ == "__main__":
    '''
    output information of model structure
    '''
    model = Contrastive(1, 2)
    summary(model, input_data=(1, 32, 32), batch_dim=0)
