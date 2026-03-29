import torch

def compute_ade(pred, gt):
    """
    pred: [batch, fut_len, 2]
    gt: [batch, fut_len, 2]
    Returns ADE (Average Displacement Error) across all timesteps in meters.
    """
    err = torch.norm(pred - gt, dim=-1) # [batch, fut_len]
    return err.mean(dim=1) # [batch]

def compute_fde(pred, gt):
    """
    pred: [batch, fut_len, 2]
    gt: [batch, fut_len, 2]
    Returns FDE (Final Displacement Error) in meters.
    """
    err = torch.norm(pred[:, -1, :] - gt[:, -1, :], dim=-1) # [batch]
    return err

def compute_min_ade_fde(preds, gt, K=3):
    """
    preds: [batch, K, fut_len, 2]
    gt: [batch, fut_len, 2]
    Returns the minimum ADE and minimum FDE among the K predicted trajectories.
    """
    batch = preds.size(0)
    
    # Expand gt to match preds K dim: [batch, K, fut_len, 2]
    gt_expanded = gt.unsqueeze(1).expand(batch, K, gt.size(1), 2)
    
    # Compute L2 error at all timesteps for all K modes
    errs = torch.norm(preds - gt_expanded, dim=-1) # [batch, K, fut_len]
    
    # ADE per mode -> [batch, K]
    ade_k = errs.mean(dim=-1)
    
    # min ADE over modes -> [batch]
    min_ade, best_ade_idx = ade_k.min(dim=1)
    
    # FDE per mode -> [batch, K]
    fde_k = errs[:, :, -1]
    
    # We report the FDE of the trajectory that had the minimum ADE (standard approach)
    # or the minimum FDE directly. Standard minFDE_3 reports min over K.
    min_fde, best_fde_idx = fde_k.min(dim=1)
    
    return min_ade, min_fde, best_ade_idx

