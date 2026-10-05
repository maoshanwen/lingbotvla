# LingBot-VLA LoRA side branch

## Chosen adapter

The default experiment uses BF16 standard LoRA, rank 32, alpha 16, and no
adapter dropout. LoRA covers the VLM and action expert; the vision encoder and
small state/action projection heads are trained directly. AdamW uses 2e-4 for
LoRA/action parameters and 2e-5 for vision parameters.

The first experiment does not use 4-bit QLoRA. The five A100 40 GB GPUs have
enough aggregate memory for the frozen 6B model with FSDP, while this codebase
uses custom VLA, vision, action-expert, and fused-MoE modules that have not been
validated with bitsandbytes. The current environment also has no bitsandbytes
installation. Quantization is therefore a later, isolated ablation rather than
the baseline.

## Placement

LoRA is injected into every VLM and action-expert attention and MLP projection:
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
`down_proj`. The vision encoder is fully trainable at one tenth of the main
learning rate. Small state/action projection heads are trained directly when
their names match `state_proj`, `action_in_proj`, or `action_out_proj`.

This follows published VLA evidence that rank 32 saturates performance and that
adapting both the VLM and action expert, while allowing visual adaptation, is
more reliable than action-expert-only LoRA. Training-time multi-camera image
augmentation is enabled to reduce clean-scene overfitting.

## Experiment ladder

1. Run `MAX_STEPS=10` as an integration and memory smoke test after evaluation
   releases the GPUs.
2. Run 100 steps to measure step time, peak memory, and loss stability.
3. Use 110k steps only as a one-epoch upper bound. After the 100-step throughput profile, cap training so it finishes within 4 days and reserves the remaining GPU time for evaluation. Save every 10,000 steps. Select the best checkpoint using held-out clean validation loss, and retain only the best checkpoint plus the newest resumable checkpoint. The three-task simulator sample is run only once after each branch finishes training to compare completed branches; it must not select checkpoints. Run the full 50-task evaluation only for the strongest final branch candidates.
4. Run the fixed small simulator evaluation once after the branch finishes;
   use it to compare finished branches, not to select checkpoints.

The initial learning rate is 2e-4 with cosine decay to 2e-5, micro-batch 1 per
GPU, global batch 5, gradient checkpointing enabled, and compilation disabled
for easier debugging. The action chunk remains 50 to match evaluation.

## Launch

```bash
bash scripts/train_lora_robotwin.sh
```

Competition compliance: the launcher intentionally fails until
`assets/training_data/robotwin_clean.txt` points to converted clean-only
LeRobot datasets. It requires exactly 50 entries, requires every path to be a
clean dataset, and rejects randomized trajectories. Final official evaluation
still uses all 50 tasks with 100 clean and 100 randomized episodes per task.
