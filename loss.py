import numpy as np
import torch
import torch.nn as nn

'''
We refer to the official implementation of alignment and uniformity in 
<Understanding contrastive representation learning through alignment and uniformity on the hypersphere>
and the official implementation of decoupled contrastive learning in
<Decoupled Contrastive Learning>

x : Tensor, shape=[bsz, d]
normalized latents for one side of positive pairs
y : Tensor, shape=[bsz, d]
normalized latents for the other side of positive pairs
device : String, "cuda" or "cpu"
the device where the latents are located
sim : Similarity
the instance of the class Similarity
'''

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

# the implementation of dcl
def decouple_contrastive_loss(x, y, device, sim):
    cos_sim = sim(x.unsqueeze(1), y.unsqueeze(0))
    ones = torch.ones(cos_sim.shape, dtype=torch.float32).to(device)
    diag = torch.eye(cos_sim.shape[0], dtype=torch.float32).to(device)
    mask = ones - diag
    align_sim = torch.mul(cos_sim, diag).sum(dim=-1)
    uniform_sim = (torch.mul(cos_sim, mask).exp().sum(dim=-1) - 1).log()
    loss = (-align_sim + uniform_sim).mean()
    return loss

# the implementation of dcl^+
def decouple_contrastive_positive_loss(x, y, device, sim):
    cos_sim = sim(x.unsqueeze(1), y.unsqueeze(0))
    ones = torch.ones(cos_sim.shape, dtype=torch.float32).to(device)
    diag = torch.eye(cos_sim.shape[0], dtype=torch.float32).to(device)
    mask = ones - diag
    align_sim = torch.mul(cos_sim, diag).sum(dim=-1)
    uniform_sim = (torch.mul(cos_sim, mask).exp().sum(dim=-1) - 1).log()
    loss = -align_sim + uniform_sim
    loss_mask = loss > 0
    loss = (loss * loss_mask).mean()
    return loss


# the implementation of mat
def margin_loss(x, y, device, margin=0.15):
    margin = margin * np.pi
    align_sq_vec = (2 - (x - y).norm(dim=1).pow(2)) / 2
    align_angle_vec = torch.arccos(align_sq_vec)
    uniform_sq_vec = torch.mm(x, y.permute(1, 0))
    ones = torch.ones(uniform_sq_vec.shape, dtype=torch.float32).to(device)
    diag = torch.eye(uniform_sq_vec.shape[0], dtype=torch.float32).to(device)
    mask = ones - diag
    uniform_sq_vec = torch.mul(uniform_sq_vec, mask)
    uniform_angle_vec = torch.arccos(uniform_sq_vec)
    uniform_min_vec = torch.min(uniform_angle_vec, dim=0).values
    angle_dist = margin + align_angle_vec - uniform_min_vec
    angle_mask = angle_dist > 0
    loss = (angle_dist * angle_mask).mean()
    return loss


# the implementation of mpt
def margin_p_loss(x, y, device, margin=0.09):
    cos_sim = torch.mm(x, y.permute(1, 0))
    ones = torch.ones(cos_sim.shape, dtype=torch.float32).to(device)
    diag = torch.eye(cos_sim.shape[0], dtype=torch.float32).to(device)
    mask = ones - diag
    align_sim = torch.mul(cos_sim, diag).sum(dim=-1)
    uniform_sim = torch.mul(cos_sim, mask)
    uniform_max_sim = torch.max(uniform_sim, dim=0).values
    cos_dist = margin - align_sim + uniform_max_sim
    cos_mask = cos_dist > 0
    loss = (cos_dist * cos_mask).mean()
    return loss

# the implementation of met
def margin_e_loss(x, y, device, margin=0.45):
    align_distance_vec = (x - y).norm(dim=1)
    uniform_sq_vec = (2 - 2 * torch.mm(x, y.permute(1, 0))).sqrt()
    ones = torch.ones(uniform_sq_vec.shape, dtype=torch.float32).to(device)
    diag = torch.eye(uniform_sq_vec.shape[0], dtype=torch.float32).to(device)
    mask = ones - diag
    uniform_dist_vec = torch.mul(uniform_sq_vec, mask) + 2 * diag
    uniform_min_vec = torch.min(uniform_dist_vec, dim=0).values
    angle_dist = margin + align_distance_vec - uniform_min_vec
    angle_mask = angle_dist > 0
    loss = (angle_dist * angle_mask).mean()
    return loss

# the implementation of alignment
def lalign(x, y, alpha=2):
    loss = (x - y).norm(dim=1).pow(alpha).mean()
    return loss

# the implementation of uniformity
def lunif(x, y, t=2):
    x_sq_pdist = torch.pdist(x, p=2).pow(2)
    x_loss = x_sq_pdist.mul(-t).exp().mean().log()
    y_sq_pdist = torch.pdist(y, p=2).pow(2)
    y_loss = y_sq_pdist.mul(-t).exp().mean().log()
    return 2 * t + (x_loss + y_loss) / 2