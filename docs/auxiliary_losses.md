# Auxiliary Loss Functions for Transformer Training

This document describes the auxiliary loss functions implemented for training transformer layers in GRIT.

## Overview

Two new loss functions have been added to improve transformer training:

1. **Attention Improvement Loss** - Encourages better structural associations in attention mechanisms
2. **Structure Reconstruction Loss** - Helps reconstruct graph structure from learned embeddings

These losses are computed during training and added to the main task loss with configurable weights.

## Loss Functions

### Attention Improvement Loss

**Purpose:** Encourages the transformer to maintain and improve structural associations through the attention mechanism.

**Implementation:** `grit.losses.attention_improvement_loss()`

**Parameters:**
- `node_embeddings` (Tensor): Initial node embeddings [N, D]
- `denoised_embeddings` (Tensor): Embeddings after QKV transformation [N, D]
- `edge_index` (Tensor): Edge indices [2, E]
- `batch` (Tensor): Batch assignment for each node [N]
- `tau` (float): Temperature for sigmoid (default=0.2)
- `weight` (float): Scalar loss weight (default=1.0)

**Returns:** Weighted scalar loss

### Structure Reconstruction Loss

**Purpose:** Reconstructs graph structure from embeddings using predicted adjacency matrix.

**Implementation:** `grit.losses.structure_reconstruction_loss()`

**Parameters:**
- `node_embeddings` (Tensor): Node embeddings after transformer layer [N, D]
- `edge_index` (Tensor): Original edge indices [2, E]
- `batch` (Tensor): Batch assignment for each node [N]
- `weight` (float): Scalar loss weight (default=1.0)

**Returns:** Weighted scalar loss

## Configuration

Add the following to your YAML configuration file:

```yaml
loss:
  attention_improvement:
    enable: True
    weight: 0.1
    tau: 0.2
  
  structure_reconstruction:
    enable: True
    weight: 0.1
```

## Backward Compatibility

The implementation maintains full backward compatibility:
- If `loss` config is not present, no auxiliary losses are computed
- If `enable` is False, the loss returns 0.0 immediately
- Existing training pipelines work without modification
