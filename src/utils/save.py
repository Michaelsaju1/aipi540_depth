"""Checkpoint saving utilities for training."""

import torch
import logging


def save_checkpoint(cfg, net, probe, opt_encoder, opt_probe, epoch, global_step, acc, reg, dataset, V=4, save_prefix="checkpoint"):
    """
    Save model checkpoint.
    
    Args:
        cfg: Configuration object
        net: Encoder network
        probe: Probe network (optional)
        opt_encoder: Encoder optimizer
        opt_probe: Probe optimizer (optional)
        epoch: Current epoch
        global_step: Global training step
        acc: Accuracy metric
        reg: Regularization metric
        dataset: Dataset name
        V: Number of views
        save_prefix: Prefix for checkpoint filename
    """
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "encoder_state_dict": net.state_dict(),
        "optimizer_encoder_state_dict": opt_encoder.state_dict(),
        "accuracy": acc,
        "regularization": reg,
        "dataset": dataset,
        "views": V,
    }
    
    if probe is not None:
        checkpoint["probe_state_dict"] = probe.state_dict()
    
    if opt_probe is not None:
        checkpoint["optimizer_probe_state_dict"] = opt_probe.state_dict()
    
    checkpoint_path = f"{save_prefix}_epoch{epoch}_step{global_step}.pt"
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved to {checkpoint_path}")
