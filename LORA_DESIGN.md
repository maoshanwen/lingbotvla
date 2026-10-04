# LingBot-VLA LoRA side branch

## Chosen adapter

The default experiment uses BF16 rank-stabilized LoRA (rsLoRA), rank 16,
alpha 32, and dropout 0.05. The pretrained model remains frozen. Adapter
weights stay in FP32 and AdamW updates only trainable parameters.

The first experiment does not use 4-bit QLoRA. The five A100 40 GB GPUs have
enough aggregate memory for the frozen 6B model with FSDP, while this codebase
uses custom VLA, vision, action-expert, and fused-MoE modules that have not been
validated with bitsandbytes. The current environment also has no bitsandbytes
installation. Quantization is therefore a later, isolated ablation rather than
the baseline.

## Placement

LoRA is injected into every action-expert attention and MLP projection:
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
`down_proj`. The Qwen3-VL language backbone and vision encoder remain frozen.
Small state/action projection heads are trained in full precision when their
names match `state_proj`, `action_in_proj`, or `action_out_proj`.

This placement is the best first tradeoff for robot control: it adapts the
policy path that directly produces actions, avoids spending adapter capacity on
the frozen visual representation, and covers all linear layers in the selected
transformer rather than only Q/V attention.

## Experiment ladder

1. Run `MAX_STEPS=10` as an integration and memory smoke test after evaluation
   releases the GPUs.
2. Run 100 steps to measure step time, peak memory, and loss stability.
3. Train for 2 epochs (about 219k optimizer steps for the current clean-data estimate), saving every 10,000 steps. Select the best checkpoint with a fixed validation/evaluation subset rather than assuming the final checkpoint is best.
4. Evaluate both checkpoints on a small fixed task subset before choosing a
   longer run.
5. Only then compare rank 8/16/32 or broaden the scope to
   `action_expert+vlm`.

The initial learning rate is 2e-4 with cosine decay to 2e-5, micro-batch 1 per
GPU, global batch 5, gradient checkpointing enabled, and compilation disabled
for easier debugging. The action chunk remains 50 to match evaluation.

## Launch

```bash
bash scripts/train_lora_robotwin.sh
```

The launcher intentionally fails until
`assets/training_data/robotwin_clean.txt` points to converted clean-only
LeRobot datasets. Randomized trajectories are rejected.
