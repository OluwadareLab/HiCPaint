# HiCPaint

Blind inpainting for Hi-C contact maps with a Diffusion Transformer.

Defaults come from `src/configs/config.yaml`. CLI flags override the config.

## Train

# Single GPU  
python [runner.py](http://runner.py)  
  
# 2 GPUs (DDP)  
torchrun --nproc_per_node=2 runner.py --distributed

Optional overrides:

```bash
python runner.py --config src/configs/config.yaml --epochs 50
torchrun --nproc_per_node=2 runner.py --distributed --batch-size 4
```

## Inference

Inference is single-GPU only.

```bash
python scripts/infer.py
# or
python -m src.test_lib
```

Optional overrides:

```bash
python scripts/infer.py \
  --checkpoint ./output/checkpoints/best_model.pt \
  --split val \
  --device cuda:0
```
