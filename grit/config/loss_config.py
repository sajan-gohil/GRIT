"""
Configuration for auxiliary loss functions in transformer training.
"""
from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_loss')
def set_cfg_loss(cfg):
    """Configuration for auxiliary loss functions during transformer training.
    
    These losses help improve transformer layer training by:
    - Encouraging better structural associations in attention
    - Reconstructing graph structure from embeddings
    """
    
    cfg.loss = CN()
    
    # Attention improvement loss settings
    cfg.loss.attention_improvement = CN()
    cfg.loss.attention_improvement.enable = False
    cfg.loss.attention_improvement.weight = 1.0
    cfg.loss.attention_improvement.tau = 0.2  # Temperature for sigmoid
    
    # Structure reconstruction loss settings
    cfg.loss.structure_reconstruction = CN()
    cfg.loss.structure_reconstruction.enable = False
    cfg.loss.structure_reconstruction.weight = 1.0
