# Quick Start Guide: Transformer Loss Functions

## What Was Added

This implementation adds two new auxiliary loss functions to improve transformer training in GRIT:

1. **Attention Improvement Loss** - Helps maintain and improve structural associations
2. **Structure Reconstruction Loss** - Reconstructs graph structure from learned embeddings

## How to Use

### Step 1: Enable in Configuration

Add this to your YAML configuration file:

```yaml
loss:
  attention_improvement:
    enable: True      # Enable attention improvement loss
    weight: 0.1       # Loss weight (start small, tune as needed)
    tau: 0.2          # Temperature parameter
  
  structure_reconstruction:
    enable: True      # Enable structure reconstruction loss
    weight: 0.1       # Loss weight (start small, tune as needed)
```

### Step 2: Train as Normal

No code changes needed! Just run your training:

```bash
python main.py --cfg configs/your_config.yaml
```

The losses will automatically be computed and added to your training loss.

### Step 3: Monitor and Tune

- Start with small weights (0.01-0.1)
- Monitor validation performance
- Adjust weights based on your task and dataset

## Example Configurations

### Minimal (Only Attention Loss)
```yaml
loss:
  attention_improvement:
    enable: True
    weight: 0.05
  structure_reconstruction:
    enable: False
```

### Balanced (Both Losses)
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

### Disabled (Backward Compatible)
```yaml
loss:
  attention_improvement:
    enable: False
  structure_reconstruction:
    enable: False
```

Or simply omit the `loss:` section entirely.

## Files Added

- `grit/losses.py` - Loss function implementations
- `grit/config/loss_config.py` - Configuration structure
- `unittests/test_losses.py` - Unit tests
- `docs/auxiliary_losses.md` - Detailed documentation
- `configs/loss_config_example.yaml` - Example config

## Files Modified

- `grit/network/grit_model.py` - Tracks embeddings during training
- `grit/train/custom_train.py` - Computes auxiliary losses

## Features

✓ **Backward Compatible** - No impact when disabled  
✓ **Memory Efficient** - Only tracks embeddings when needed  
✓ **Gradient Safe** - Proper gradient flow through all losses  
✓ **Error Handling** - Handles edge cases gracefully  
✓ **Configurable** - Easy to enable/disable and tune weights  

## Need Help?

- See `docs/auxiliary_losses.md` for detailed documentation
- See `configs/loss_config_example.yaml` for more examples
- See `IMPLEMENTATION_SUMMARY.md` for implementation details

## Validation

Run the validation script to verify installation:

```bash
python validate_implementation.py
```

All checks should pass (10/10).
