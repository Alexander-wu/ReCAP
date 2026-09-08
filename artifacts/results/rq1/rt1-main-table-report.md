# RT-1 主表实验：六种上下文调度策略对比

**日期**：2026-09-01
**模型**：RLVR-World 多步 checkpoint（`thuml/rt1-world-model-multi-step-rlvr`，12 层 Llama / hidden 768），全程冻结
**Tokenizer**：`thuml/rt1-compressive-tokenizer`（CompressiveVQModelFSQ）
**数据**：Open X-Embodiment `fractal20220817_data`（RT-1）shard 0
**硬件**：1× NVIDIA H20，PyTorch 2.9.1 / CUDA 12.9 / vLLM 0.13.0
**采样**：temperature 0.5、top-k 50、top-p 0.9、seed 42（所有策略完全一致）

## 复现的六种调度

帧对齐遵循参考实现：**frame 0 = 视觉上下文，frame 1 = 首个真实动态帧，生成从 frame 2 开始**。预测第 k 帧对齐真值第 k+2 帧。

主表的 5 个 baseline 全部实现 + ReCAP：

| 策略 | 机制 | 保留粒度 | prompt 上限 |
|---|---|---|---|
| `full_history` | Full Context。保留全部生成 block；超 `MAX_PROMPT_LEN=7900` 时截断为 `ctx + 末尾 6620 token` | token（非对齐） | 7900 |
| `sliding_window` | Sliding Window。只保留最近 W=6 个完整 block，**不保留 anchor** | block | 1838 |
| `uniform_sampling` | Uniform Sampling。等预算下，一半给最近 3 块，一半等距抽取更早的 4 块 | block | 1931 |
| `streaming_llm` | StreamingLLM。sink=4 token + 最近 647 token（= 6.957 block，**故意非对齐**） | token | 1931 |
| `block_kv` | Block-KV Selection。H2O/SnapKV 的 block 级改编，按「与当前帧 token 重合度 × 时近性衰减」选块 | block | 1931 |
| **`recap`** | **`ctx + 冻结 anchor A + 最近 W=6 完整 block`，全程不出 token 空间** | **block** | **1931（恒定）** |

六种策略在**同一个进程、同一次模型加载**内跑完，采样参数、动作序列、seed 完全一致。

> **Chunked Sliding Window 已移除**（2026-09-01 决定）。它每 6 帧要把上一帧经像素空间重编码，属于「换一种生成流程」而非「换一种上下文调度」，与主表其余方法不可比。相关代码、绘图和聚合逻辑均已清理。

### 两处刻意的公平性设计

**1. 等 block 预算。** `uniform_sampling`、`block_kv`、`recap` 都用 `1 + W = 7` 个 block（prompt 1931）。`sliding_window` 用 6 个（1838）—— 它就是 ReCAP 去掉 anchor，两者**恰好差一个 block**，所以它同时充当 anchor 组件的消融对照。

**2. StreamingLLM 用 token 预算，且故意不对齐。** 总预算 `93 × 7 = 651` token，减去 sink 4 个 → recent 647 token = **6.957 个 block**。prompt 上限 1931 与 ReCAP **完全相同**，但截断窗口起点几乎从不落在 block 边界上（从第 7 步起，30 步里 23 次切在 block 中间）。这是「同等预算下 block 级 vs token 级」最干净的对比。

注意：如果把 sink 设成一整块（93 token），recent 就刚好 6 块对齐，StreamingLLM 会退化成 ReCAP，测不出任何差异。原论文用 4 个 token，我们照此设置。

## ep0 结果（115 帧原长，生成 113 帧）

### 像素指标：几乎测不出差异

| 策略 | PSNR | MAE | Token 保留 | unique 首→末 | prompt 峰值 | s/frame |
|---|---:|---:|---:|---:|---:|---:|
| Full Context | 20.64 | 0.0526 | **0.040** | 77 → **3** | 7900 | 0.328 |
| Sliding Window | **20.89** | **0.0441** | 0.978 | 78 → 77 | 1838 | 0.318 |
| Uniform Sampling | 20.77 | 0.0447 | 0.990 | 78 → 78 | 1931 | 0.318 |
| StreamingLLM | 20.73 | 0.0455 | 0.980 | 78 → 76 | 1931 | 0.320 |
| Block-KV Selection | 20.86 | 0.0447 | 0.989 | 78 → 78 | 1931 | 0.321 |
| **ReCAP** | 20.62 | 0.0453 | 0.982 | 78 → 76 | 1931 | **0.317** |

五个有界策略的 PSNR 挤在 20.62–20.89 之间（0.27 dB 跨度），ReCAP 甚至垫底。**PSNR 不能作为主指标。**

### 运动/轨迹指标：ReCAP 全面领先

| 策略 | 运动相关性 | 轨迹相关性 | 末端任务推进 | 末段运动保持（t≥70） |
|---|---:|---:|---:|---:|
| Full Context | **−0.051** | +0.213 | 34.9% | 4.5% |
| Sliding Window | +0.376 | +0.653 | 51.0% | 14.8% |
| Uniform Sampling | +0.352 | +0.604 | 52.5% | 15.7% |
| StreamingLLM | +0.369 | +0.688 | 53.9% | 14.9% |
| Block-KV Selection | +0.265 | +0.512 | 46.2% | 13.1% |
| **ReCAP** | **+0.673** | **+0.814** | **73.9%** | **41.2%** |

- **运动相关性**：预测逐帧运动量与真值逐帧运动量的 Pearson r —— 直接回答「机器人该动的时候你动了吗」
- **轨迹相关性**：`mean|f_t − f_0|` 曲线与真值的相关性
- **末端任务推进**：末帧 drift / 真值末帧 drift，表示场景推进到了几成

**ReCAP 的运动相关性是次优（Sliding Window +0.376）的 1.79 倍，末段运动保持是 2.78 倍。** Full Context 的运动相关性为**负值**，说明它的输出与真实运动完全脱钩。

关键帧图七行（真值 + 6 策略）：Full Context 第 3 帧后机械臂彻底消失，其余五行都保住了机械臂，但只有 ReCAP 的姿态推进跟得上真值。

## 五条对论文重要的发现

### 1. 塌缩不是上下文溢出导致的

短 episode（ep2 23 帧、ep3 20 帧）的 `truncation_steps` 都是 **0** —— prompt 峰值 3419 / 3140，远低于 7900 上限，**根本没触发截断**。但 Full Context 的 token 保留率已经掉到 0.833 / 0.850。

说明失效来自**陈旧历史的累积本身**，而非截断这个工程细节。直接支撑 memory-role conflict，也堵住「你只是修了个截断 bug」这类质疑。

### 2. 主指标必须换成运动/轨迹类

ep0 上六策略的 PSNR 相差不到 0.3 dB，而运动相关性相差 **13 倍**（−0.051 到 +0.673）。原因很直接：机械臂在 256×320 画面里只占一小块面积，一个卡住不动但背景干净的 rollout，PSNR 不会太差。

主表建议的指标层次：
- **主指标**：motion correlation、task progress ratio、late motion preservation
- **辅助**：token diversity retention（对塌缩极敏感，Full Context 0.040 vs 其余 ≈0.98）
- **参考**：PSNR / MAE / LPIPS（须说明它们对该失效不敏感）

原计划里的 **nDTW 和 Arm Score 方向正确**，应尽快实现。

### 3. anchor 只值一个 block，但效果显著

ReCAP 与 Sliding Window 的唯一差异是那 93 token 的 anchor block（占 prompt 4.8%），却带来运动相关性 +0.376 → +0.673、任务推进 51.0% → 73.9%。这是极干净的组件消融。

注意一个必须在论文里说清的边界：**`ctx_prefix`（1280 token，frame 0 的场景 token）在所有六种策略中都不被驱逐。** 所以「保持 scene identity」不是 ReCAP 独占的能力；ReCAP 的 anchor 提供的是**第一个真实动态帧 + 其动作**，即一个真实的动力学参考点。论文措辞需要与此对齐，否则审稿人一查代码就会发现表述不准。

### 4. Baseline 的截断确实切在 block 中间

`MAX_PROMPT_LEN − CTX_PREFIX_LEN = 6620`，而 `6620 / 93 = 71.18` —— **不是 block 整数倍**。ep0 在第 71 步首次触发，共 42 次，每次都把某个 visual-action block 切成两半。

StreamingLLM 同理：recent 预算 647 token，窗口起点从第 7 步起几乎每步都非对齐（30 步里 23 次）。两者共同支撑 related work 对 token 级 eviction 的批判。

### 5. ReCAP 的成本优势是结构性的

| 策略 | prompt 峰值 | 是否随 horizon 增长 |
|---|---:|---|
| Full Context | 7900 | 是，直到撞上限后开始丢信息 |
| Sliding Window | 1838 | 否 |
| Uniform / StreamingLLM / Block-KV | 1931 | 否 |
| **ReCAP** | **1931** | **否，恒定 `1280 + 93×(1+6)`** |

所有有界策略的成本相当（s/frame 0.317–0.321），ReCAP 略快。**关键是在同等预算下 ReCAP 的质量明显更高**，不是靠多花预算换来的。此外 ReCAP 全程不离开 token 空间，无编解码累积损失。

## 待办

**50 样本系统评测**（此前用旧策略集启动过一次，已停止，需用六策略集重跑）：
- 前 50 个 episode，平均 45 帧，约 2141 生成帧 × 6 策略 ≈ 12800 步
- 聚合脚本 `summarize_comparison.py` 给出各策略均值 + 相对 Full Context 的**配对 bootstrap 95% CI** + 每指标胜率

**W 扫描**：W ∈ {1,2,3,4,6,8} × {Sliding Window, ReCAP}，验证 anchor 增益是否在小 W 下更明显。

## 文件清单

脚本：
- `infer_compare.py` — 单进程跑完六策略，逐帧指标 + 关键帧网格 + 对比 GIF
- `submit_compare.sh` — 单 episode 提交（`EMIT_PREVIEW=0` 关掉 GIF 回传，批量时用）
- `run_batch_compare.sh` — 批量驱动，逐 episode 收集 result.json
- `summarize_comparison.py` — 跨 episode 聚合 + 配对 bootstrap CI
- `plot_comparison.py` — 四面板逐帧曲线（PSNR / 运动保持 / drift / token 多样性）
- `plot_task_progress.py` — 三面板任务推进图（**主图素材**）

产出：
- `outputs/task_progress_ep0_6strategies.png` — 主图素材
- `outputs/compare_curves_ep0_6strategies.png` — 四面板曲线
- `outputs/compare-<episode>-<时间>/keyframe_grid.png` — 七行关键帧对比
- `outputs/compare-<episode>-<时间>/result.json` — 全部逐帧指标

## 复现命令

```bash
cd recap-multistep-inference

# 单个 episode，六策略 + 可视化
./submit_compare.sh /path/to/episode.npz 113

# 批量 50 样本（关掉 GIF 回传）
EMIT_PREVIEW=0 BATCH_TAG=rt1-50 ./run_batch_compare.sh /path/to/episodes 50 100

# 聚合统计
python summarize_comparison.py outputs/batch-rt1-50/*.json \
  --json-out outputs/batch-rt1-50-summary.json

# 主图
python plot_task_progress.py outputs/batch-rt1-50/*.json \
  --min-frames 40 --output outputs/task_progress_50.png
```
