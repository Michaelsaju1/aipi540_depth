import torch


def compute_metrics(pred, target, count=None):
    """
    Compute depth estimation metrics.
    
    Returns raw sums and pixel counts to allow correct aggregation 
    over batches/patches.
    preds/tgt: B, C, H, W
    """
    pred = pred.detach()
    target = target.detach()
    
    eps = 1e-6
    pred = pred.clamp(min=eps)
    target = target.clamp(min=eps)
    
    # Pixel-wise error sums
    abs_rel_sum = torch.sum(torch.abs(pred - target) / target)
    sq_diff_sum = torch.sum((pred - target) ** 2)
    
    # Threshold counts
    ratio = torch.max(pred / target, target / pred)
    delta1_sum = (ratio < 1.25).float().sum()
    delta2_sum = (ratio < 1.25 ** 2).float().sum()
    delta3_sum = (ratio < 1.25 ** 3).float().sum()
    
    return {
        "abs_rel_sum": abs_rel_sum.item(),
        "sq_diff_sum": sq_diff_sum.item(),
        "delta1_sum": delta1_sum.item(),
        "delta2_sum": delta2_sum.item(),
        "delta3_sum": delta3_sum.item(),
        "n_pixels": pred.numel()
    }


