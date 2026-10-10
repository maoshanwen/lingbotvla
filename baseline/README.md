# LingBot-VLA 2.0 竞赛 baseline

这是针对天池蚂蚁灵波具身大模型挑战赛初赛的完整脚本流程：50 个 Aloha-AgileX clean ZIP → RoboTwin 官方处理 → LeRobot 数据集 → **只用 clean 数据**计算归一化统计并训练 LingBot-VLA 2.0 → clean/randomized 各 50 任务 × 100 回合仿真 → 官方 JSON 模板。输出的 `results.json` 只在全部 10000 回合完成且日志完整时才写出。

2026-09-29 已在 4×H100 上完成两步训练烟测：第 2 步耗时约 9.3 秒，每卡显存峰值约 56.6 GB，checkpoint 位于 `/tmp/lingbot-smoke-train-output/checkpoints/global_step_2/hf_ckpt`，训练日志见 `work/smoke/train4.log`。以该步速估算 2000 步的纯迭代时间约 5.2 小时；烟测只使用 `lift_pot` 的单条轨迹，完整数据的视频解码及保存开销尚未计入。仿真已能用该 checkpoint 完成 `lift_pot` 的单回合动作推理，但这不是正式比赛成绩：只训练了两步，也没有执行 clean/randomized 的 50×100 回合。烟测日志见 `work/smoke/eval_bounded/`。官方模型代码和 RoboTwin 仿真代码分别在相邻的 `lingbot-vla-v2/` 和 `RoboTwin/`，基模权重在 `lingbot-vla-v2-6b/`。

单回合烟测使用了 `work/smoke/run_eval_smoke.py`：临时关闭 cuRobo 的融合 LBFGS CUDA 内核，以绕开本机该内核的 `illegal instruction`；客户端写完结果后直接退出，以避开解释器退出时的段错误。脚本结束会恢复官方配置和客户端。官方启动脚本成功生成了 `work/smoke/eval_bounded/lingbot-smoke-train-output_0k_demo_smoke_20260929_051551/stats.txt`，显示 `lift_pot` 0/1、`NO(1/100)`。这个结果不能用于提交；完整比赛评测仍需先完成 50 个 clean 任务的数据处理与训练，并在 clean/randomized 各执行 50×100 回合。

## 新机器初始化

Git 仓库只需包含 `baseline/` 与 `灵波2026/` 中的小型说明、JSON 模板；根目录 [`.gitignore`](../.gitignore) 会排除虚拟环境、下载的第三方源码、模型、原始数据、仿真 assets 和训练输出。新机器需先有 Linux x86_64、NVIDIA 驱动、CUDA toolkit（含 `nvcc`）、`git`、`curl` 与足够的磁盘空间。初始化脚本会在缺少 uv 时安装它。

```bash
bash baseline/bootstrap.sh
```

[`bootstrap.sh`](bootstrap.sh) 会拉取固定提交的 LingBot/RoboTwin 源码，创建 `.venvs/lingbot`（Python 3.12、PyTorch 2.8/cu128）和 `.venvs/robotwin`（Python 3.11、PyTorch 2.4/cu124），按 [`requirements.txt`](requirements.txt) 与 [`requirements-sim.txt`](requirements-sim.txt) 安装依赖，并编译 cuRobo。随后 [`download_artifacts.py`](download_artifacts.py) 从官方 Hugging Face 仓库下载 LingBot 6B 基模、Qwen3-VL、MoGe、50 个 clean ZIP 和 RoboTwin 三个仿真资产包；下载使用当前机器已验证的 revision，可通过 `LINGBOT_MODEL_REV`、`ROBOTWIN_DATA_REV`、`QWEN_REV`、`MOGE_REV` 覆盖。最后解压 assets、更新机器人配置中的绝对路径、验证 50 个 ZIP。重复运行会利用已有环境和下载缓存；`--only-deps` 或 `--only-download` 可单独重试一个阶段。

当前机器的两个虚拟环境已重建；项目中的 `.venvs` 是指向 `/home/tiger/.local/share/lingbot-race/venvs` 的链接，避免依赖会被清理的 `/tmp`。可用 `source .venvs/lingbot/bin/activate` 进入训练环境；端到端脚本会自动激活它。切换到其他机器时，初始化脚本仍会在项目的 `.venvs` 位置创建环境。

初始化完成后运行：

```bash
TEAM_ID=真实队伍ID EVAL_COMPAT=1 bash baseline/run_all.sh
```

`run_all.sh` 会自动激活新建的训练环境，并用训练环境运行 LeRobot 转换，因此无需再安装 pi0 的整套 uv 项目依赖。`EVAL_COMPAT=1` 仅在遇到下述 H100 仿真故障时使用。

## 环境与额外权重

1. 推荐先运行上方的 `baseline/bootstrap.sh` 一次性创建 LingBot 训练/推理环境、RoboTwin 仿真环境并下载 assets。需要手动部署时，可参考 [`lingbot-vla-v2/README.md`](../lingbot-vla-v2/README.md) 和 [`RoboTwin/README.md`](../RoboTwin/README.md)。RoboTwin 源码固定在 LingBot 指南要求的 `13c3c47ff4312dd62484bcd51be034af55c062d1`；assets 不包含在 Git 源码里。
2. 训练还需要 Qwen3-VL-4B-Instruct 和 MoGe-2-vitb-normal；基模目录已有 LingBot-Depth 和 DINO-Video teacher。当前已下载到 `baseline/models/`；如需重下，可运行 `python baseline/download_prereqs.py`，也可通过 `QWEN3VL_PATH`、`MOGE_PATH` 指向已有文件。完整 Qwen 仓库约 9 GB。它们是官方基模所需的预训练组件，并非 randomized 任务数据。
3. `prepare_data.py process/all` 使用的 Python 需要 `h5py`、`cv2`、`PyYAML`；独立运行时 LeRobot 转换默认使用 RoboTwin `policy/pi0` 的 `uv run`，也可以用 `--convert-python` 指定装有 LeRobot、tyro 的训练环境 Python。`run_all.sh` 自动使用后一种方式。LingBot 训练环境固定 `datasets==3.6.0`；不要用系统里的 `datasets 4.0` 读取这个 LeRobot v2.1 数据集。设置足够的磁盘空间和缓存目录。

## 运行顺序

在当前 4×H100 机器上，一条命令可以从原始 clean ZIP 完成数据处理、训练、选取 `global_step_${MAX_STEPS}/hf_ckpt`、clean/randomized 仿真，以及提交 JSON：

```bash
TEAM_ID=真实队伍ID \
TRAIN_ACTIVATE=/tmp/lingbot-train-env/bin/activate \
EVAL_COMPAT=1 \
bash baseline/run_all.sh
```

入口是 [`run_all.sh`](run_all.sh)。它默认执行 `prepare_data.py all`（已有数据会跳过重复处理），训练 2000 步，确认目标 checkpoint 的全部权重分片存在后才启动 [`eval.sh`](eval.sh)。`eval.sh` 会执行 clean 和 randomized 各 50 任务 × 100 回合，并在成功后写出 `baseline/work/submission/01_评测结果/results.json`。`TEAM_ID` 必须是实际队伍 ID；本机默认使用 `work/smoke/conda_compat.sh` 切换训练与仿真虚拟环境。其他机器需设置 `CONDA_SH=/path/to/conda.sh`，并先激活 LingBot 训练环境，或指定 `TRAIN_ACTIVATE`。可用 `MAX_STEPS`、`SAVE_STEPS`、`MICRO_BATCH_SIZE`、`GLOBAL_BATCH_SIZE` 调整训练，用 `EVAL_NUM_GPUS` 和 `EVAL_NUM_PER_GPU` 调整评测并行度。若已准备好全部 50 个转换后数据，可设 `PREPARE_DATA=0`。

`EVAL_COMPAT=1` 只为当前 H100 机器的已复现故障准备：评测期间临时关闭 cuRobo 融合 LBFGS 内核，并在评测客户端写完结果后避开解释器退出时的段错误；结束后会恢复官方文件。其他环境可不设置，直接走原版评测代码。此兼容处理改了仿真规划器的数值执行路径，正式提交时应在复现说明中披露。完整数据转换、2000 步训练和 10000 回合评测耗时很长，上述完整命令尚未执行；当前只有前述两步训练和单回合烟测结果。

从工作区根目录执行：

```bash
python baseline/prepare_data.py verify
python baseline/prepare_data.py all --process-python /path/to/RoboTwin-env/bin/python
```

`all` 会逐任务读取当前 `灵波2026/robotwin_official/dataset/<task>/aloha-agilex_clean_50.zip`，只解压 `data/` 与 `instructions/`；调用 RoboTwin 自带 `process_data.py` 和 LeRobot 转换器，最终生成 `baseline/work/clean_datasets.txt`。重复运行会跳过已完成任务。也可先运行 `hdf5` 阶段完成解压与官方 HDF5 处理，再运行 `convert` 阶段。调试可用 `--tasks lift_pot` 只处理一个任务，但不会生成完整训练清单。`verify` 只检查 50 个压缩包是否有每任务 50 条数据与指令，不展开大型数据。

数据准备默认一次只处理一个任务。可在启动前设置 `PREPARE_WORKERS=4`，让 `run_all.sh` 同时准备四个不同任务；独立运行时可给 `prepare_data.py` 加 `--workers 4`。正在运行的进程不会读取新参数，需重启才生效；已完整完成的任务会跳过。原始解压数据位于 `RoboTwin/data/<task>/aloha-agilex_clean_50/`，中间 HDF5 位于 `RoboTwin/policy/pi0/processed_data/<task>-aloha-agilex_clean_50-50/`，最终 LeRobot 训练数据位于 `baseline/work/cache/huggingface/lerobot/robotwin_clean/<task>/`。

训练前，确保已下载额外权重并激活 LingBot 环境。官方 RoboTwin 示例清单包含 randomized 数据，**不要使用**。本脚本始终使用自己生成的 clean 清单与 clean 归一化统计。

```bash
export QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct
export MOGE_PATH=/path/to/moge-2-vitb-normal/model.pt
MAX_STEPS=2000 SAVE_STEPS=500 MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=4 \
  bash baseline/train.sh
```

`train.sh` 先生成独立的 clean robot config，再调用 LingBot 官方 `scripts/compute_norm_stats.py` 生成 `baseline/work/clean_norm_stats.json`，最后基于官方 RoboTwin 训练配置生成 `baseline/work/train_config.yaml` 并训练。这样训练数据加载器不会读取官方示例中混合数据的统计量。`MAX_STEPS` 默认 18000，`SAVE_STEPS` 默认 2000；`GLOBAL_BATCH_SIZE` 默认 32，`MICRO_BATCH_SIZE` 默认 2；脚本根据可见 GPU 数（多机时乘以 `NNODES`）自动计算梯度累积次数。4 卡、每卡 micro batch 为 2 时累积 4 次，全局 batch = 每卡 micro batch × 数据并行 GPU 数 × 梯度累积次数；全局 batch 必须是每卡 micro batch × GPU 数的整数倍。可设置 `MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=4` 恢复原来的 batch。6B 模型及深度/视频 teacher 需要多卡或大显存。训练 checkpoint 在 `baseline/work/train_output/checkpoints/global_step_*/hf_ckpt`。训练中的 `lingbotvla_cli.yaml` 和 clean norm 文件必须随 checkpoint 保留供推理读取。

续训时，学习率调度器使用本次生成的配置（包括 `MAX_STEPS`、学习率和 warmup 比例），在恢复模型与优化器后，立即按 checkpoint 的 `global_step` 重算各参数组学习率。`MAX_STEPS` 表示训练总目标步数，不是追加步数。例如已有第 2000 步 checkpoint，执行 `MAX_STEPS=50000 SAVE_STEPS=5000 bash baseline/train.sh` 会尝试恢复最新可用 checkpoint 并训练至总第 50000 步。当前默认余弦调度、基础 LR `1e-4`、最低 LR `5e-5`、warmup 比例 0 时，恢复至第 2000 步的基础参数组 LR 约为 `9.98e-5`，第 50000 步降至 `5e-5`；各组原有 LR 比例保留。延长训练预算后 LR 可能回升，已完成的前 2000 步不会被重新执行，因此它不等同于从头按 50000 步曲线训练。

从预训练权重重新开始微调，可执行 `ENABLE_RESUME=0 TRAIN_OUTPUT_DIR="$PWD/baseline/work/train_output_bs32_fresh" bash baseline/train.sh`。`ENABLE_RESUME=0` 关闭 checkpoint 恢复，模型仍从 `LINGBOT_BASE_WEIGHTS` 加载；新输出目录保留旧 checkpoint 和评测所需配置。默认 `ENABLE_RESUME=1`，输出目录仍为 `baseline/work/train_output`。

恢复日志中的 `Resumed LR schedule from current config` 会打印恢复步数、总步数和各参数组 LR。调度恢复的 CPU 测试可运行 `.venvs/lingbot/bin/python -m unittest discover -s baseline/tests -p 'test_resume_lr.py' -v`。


仿真时激活 LingBot 环境，并确保 RoboTwin 环境和 assets 已准备好：

```bash
export TEAM_ID=your_team_id
export QWEN3VL_PATH=/path/to/Qwen3-VL-4B-Instruct
export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
EVAL_NUM_GPUS=1 bash baseline/eval.sh \
  /absolute/path/to/baseline/work/train_output/checkpoints/global_step_2000/hf_ckpt
```

`eval.sh` 调用 LingBot 官方 `start_robotwin_infer_and_eval.sh` 两次，分别评测 `demo_clean` 和 `demo_randomized`；原始日志与 `stats.txt` 写到 `baseline/work/eval/`。成功完成全部 50×100×2 回合后，`make_results.py` 从每个任务最后的 `Success rate: 成功数/测试数` 日志生成 `baseline/work/submission/01_评测结果/results.json`，包含官方模板的 `schema_version`、`team_id`、`clean` 和 `randomized` 的各任务 `attempts/successes`。有缺失任务、少于 100 回合或官方统计标记未完成时，脚本会报错，不填虚假的零。

### 子集评测与前 K 回合

`eval.sh` 支持两个可选环境变量：

- `EVAL_TASKS`：默认 `click_bell,open_microwave,turn_switch,place_a2b_left` 四个任务。设为 `all` 评测全部 50 个任务。`recommended8` 选择 `click_bell,press_stapler,turn_switch,shake_bottle,open_microwave,place_a2b_left,pick_dual_bottles,stack_blocks_two`；也可传逗号分隔的官方任务名，例如 `EVAL_TASKS=press_stapler,turn_switch,shake_bottle`。
- `EVAL_EPISODES=100`：默认每任务、每 setting 100 回合，可设 1–100。`30` 表示原评测序列的前 30 个有效回合，不随机抽取回合。clean / randomized 都执行。

推荐 8 项 × 两组 × 前 30 回合，共 480 回合：

```bash
TEAM_ID=1382900 \
QWEN3VL_PATH=/tmp/lingbot-models/Qwen3-VL-4B-Instruct \
CONDA_SH=/mnt/bn/search-nlp-vagcp/qiaohaoyu/lingbot/baseline/venv_conda_adapter.sh \
EVAL_COMPAT=1 EVAL_NUM_GPUS=4 EVAL_NUM_PER_GPU=2 \
EVAL_USE_BF16=True EVAL_USE_FP32=False \
EVAL_TASKS=recommended8 EVAL_EPISODES=30 \
bash baseline/eval.sh \
  /mnt/bn/search-nlp-vagcp/qiaohaoyu/lingbot/baseline/work/train_output/checkpoints/global_step_2000/hf_ckpt
```

评测仍从 `seed=0` 对应的环境 seed `100000` 开始，按原顺序递增，并沿用专家规划失败时跳过该环境的逻辑。因此“前 K 个”指原 100 回合评测中的前 K 个有效回合，未必是连续 K 个环境 seed。候选语言指令的生成数量固定为原来的 100，以避免减小 K 后同一 seed 的指令变化；场景和指令内部的随机化仍由原 seed 控制。

每次结果写入 `baseline/work/eval/<本次目录>/results.json`。子集结果额外记录任务、回合数及选择方式，不覆盖 `work/submission/01_评测结果/results.json`。仅全部 50 任务且每组每任务 100 回合完整结束后，才更新全量提交 JSON。

底层启动脚本也支持 `--tasks <逗号分隔任务名>` 和 `--test_num K`；显式 `--tasks` 优先于原 `--num_tasks`。CPU 验证可运行 `python -m unittest discover -s baseline/tests -v`，这些测试不启动真实模型和仿真。

比赛提交包还需要 `02_代码材料/` 中的训练和评测代码及配置、`03_模型权重/` checkpoint、`04_复现与调优说明/`。参见 [`灵波2026/代码材料目录要求.md`](../灵波2026/代码材料目录要求.md)。若修改 RoboTwin 官方评测逻辑，必须在提交说明中披露；本 baseline 直接使用其任务与成功判定，并使用 LingBot 官方评测客户端。
