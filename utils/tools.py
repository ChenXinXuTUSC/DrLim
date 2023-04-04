import numpy as np
import torch

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
        ax1.scatter(x[clsmask, 0], x[clsmask, 1], color=[cmap1(norm(id))]*len(clsmask), alpha=0.5, label=cls)
        ax1.title.set_text("DrLim-plasma")
        ax1.legend(loc="upper right")
        ax2.scatter(x[clsmask, 0], x[clsmask, 1], color=[cmap2(norm(id))]*len(clsmask), alpha=0.5, label=cls)
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
        ax1.scatter(x[clsmask, 0], x[clsmask, 1], x[clsmask, 2], color=[cmap1(norm(id))]*len(clsmask), alpha=0.5, label=cls)
        ax1.title.set_text("DrLim-plasma")
        ax1.legend(loc="upper right")
        ax2.scatter(x[clsmask, 0], x[clsmask, 1], x[clsmask, 2], color=[cmap2(norm(id))]*len(clsmask), alpha=0.5, label=cls)
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

def euclidean_distmat_numpy(batch: np.ndarray):
    '''Efficient computation of Euclidean distance matrix 
    
    params
    ----------
    * batch: input tensor of shape(batch_size, feature_dims)

    return
    ----------
    * Distance matrix of shape (batch_size, batch_size) 
    '''
    
    eps = 1e-8
    
    # step1: compute self cartesian product
    self_product = batch @ batch.T # (batch_size, batch_size)
    
    # step2: extract the squared euclidean distance of each
    #        sample from diagonal
    squared_dig = np.diag(self_product)
    
    # step3: compute squared euclidean dists using the formula
    #        (a - b)^2 = a^2 - 2ab + b^2
    distmat = np.expand_dims(squared_dig, 0) - 2 * self_product + np.expand_dims(squared_dig, 1)
    
    # get rid of negative distances
    distmat = distmat * (distmat > 0.0).astype(np.float32)
    
    # step4: take the square root of distence matrix
    distmat = np.sqrt(distmat)
    
    return distmat

def euclidean_distmat_torch(batch: torch.Tensor):
    '''Efficient computation of Euclidean distance matrix 
    
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
    distmat += eps * mask
    
    distmat = torch.sqrt(distmat)
    
    # undo the trick for numerical instabilities
    distmat *= (1.0 - mask)
    
    return distmat
