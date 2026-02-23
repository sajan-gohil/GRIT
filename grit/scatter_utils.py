"""
Replacement functions for torch_scatter operations using native PyTorch.
These functions provide the same interface as torch_scatter functions.
"""

import torch


def scatter(src, index, dim=0, dim_size=None, out=None, reduce='add'):
    """
    PyTorch native replacement for torch_scatter.scatter.
    
    Args:
        src: Source tensor with values to scatter
        index: Index tensor indicating where to scatter values
        dim: Dimension along which to scatter
        dim_size: Size of the output dimension (optional, inferred if None)
        out: Output tensor to scatter into (optional)
        reduce: Reduction operation - 'add', 'mean', 'mul', 'min', 'max', 'sum'
    
    Returns:
        Output tensor with scattered values
    """
    # Handle the case where dim_size is not provided
    if dim_size is None:
        if index.numel() > 0:
            dim_size = index.max().item() + 1
        else:
            dim_size = 0
    
    # Create output tensor if not provided
    if out is None:
        out_shape = list(src.shape)
        out_shape[dim] = dim_size
        
        if reduce == 'mul':
            # For multiplication, start with ones
            out = torch.ones(out_shape, dtype=src.dtype, device=src.device)
        else:
            # For all other operations, start with zeros
            out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    
    # Apply the scatter operation based on the reduce type
    if reduce in ['add', 'sum']:
        out.scatter_add_(dim, index, src)
    elif reduce == 'mul':
        # For multiplication, we need a different approach
        # Create a temporary output filled with ones
        temp_out = torch.ones(list(src.shape), dtype=src.dtype, device=src.device)
        temp_out.mul_(src)
        # Use scatter with multiply operation
        out_temp = torch.ones(out.shape, dtype=src.dtype, device=src.device)
        out_temp.scatter_(dim, index, temp_out, reduce='add')
        # This is tricky - multiply operation in scatter
        for i in range(out.shape[dim]):
            mask = (index == i)
            if mask.any():
                out.select(dim, i).mul_(src[mask].prod(dim=0))
    elif reduce == 'mean':
        out.scatter_add_(dim, index, src)
        # Count occurrences for mean
        counts = torch.zeros(out.shape[dim], dtype=torch.float32, device=src.device)
        if dim == 0:
            counts.scatter_add_(0, index, torch.ones(index.shape[0], device=src.device))
            out = out / (counts.unsqueeze(-1) + 1e-16)
        else:
            counts.scatter_add_(0, index, torch.ones(index.shape[0], device=src.device))
            out = out / (counts.view([-1 if i == dim else 1 for i in range(len(out.shape))]) + 1e-16)
    elif reduce == 'max':
        out.scatter_(dim, index, src, reduce='amax')
    elif reduce == 'min':
        out = torch.full(out.shape, float('inf'), dtype=src.dtype, device=src.device)
        out.scatter_(dim, index, src, reduce='amin')
    else:
        raise ValueError(f"Unknown reduce operation: {reduce}")
    
    return out


def scatter_add(src, index, dim=0, dim_size=None):
    """
    PyTorch native replacement for torch_scatter.scatter_add.
    Adds values from src into a tensor of size dim_size at indices specified.
    
    Args:
        src: Source tensor with values to add
        index: Index tensor indicating where to add values
        dim: Dimension along which to scatter
        dim_size: Size of the output dimension
    
    Returns:
        Output tensor with aggregated values
    """
    if dim_size is None:
        if index.numel() > 0:
            dim_size = index.max().item() + 1
        else:
            dim_size = 0
    
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    out.scatter_add_(dim, index, src)
    
    return out


def scatter_max(src, index, dim=0, dim_size=None):
    """
    PyTorch native replacement for torch_scatter.scatter_max.
    Returns the maximum values from src for each index.
    
    Args:
        src: Source tensor with values
        index: Index tensor indicating grouping
        dim: Dimension along which to scatter
        dim_size: Size of the output dimension
    
    Returns:
        Tuple of (output tensor with max values, indices of max values)
    """
    if dim_size is None:
        if index.numel() > 0:
            dim_size = index.max().item() + 1
        else:
            dim_size = 0
    
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    
    # Create output for values
    out = torch.full(out_shape, float('-inf'), dtype=src.dtype, device=src.device)
    out.scatter_(dim, index, src, reduce='amax')
    
    # Create output for indices - track which element gave the max
    out_indices = torch.zeros(out_shape, dtype=index.dtype, device=src.device)
    
    # For each index value, find which src element gave the max
    if dim == 0:
        for i in range(dim_size):
            mask = (index == i)
            if mask.any():
                max_idx = torch.nonzero(mask)[src[mask].argmax()].item()
                out_indices[i] = max_idx
    
    return out, out_indices


# For compatibility, these are the same as above
def scatter_sum(src, index, dim=0, dim_size=None):
    """Alias for scatter with reduce='add'."""
    return scatter_add(src, index, dim=dim, dim_size=dim_size)


def scatter_mean(src, index, dim=0, dim_size=None):
    """Compute mean instead of sum."""
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce='mean')


def scatter_min(src, index, dim=0, dim_size=None):
    """Compute minimum values."""
    if dim_size is None:
        if index.numel() > 0:
            dim_size = index.max().item() + 1
        else:
            dim_size = 0
    
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.full(out_shape, float('inf'), dtype=src.dtype, device=src.device)
    out.scatter_(dim, index, src, reduce='amin')
    
    return out
