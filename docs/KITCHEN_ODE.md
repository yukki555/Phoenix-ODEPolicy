# Phoenix 四任务动作 ODE 后训练

基线为本地 Phoenix-MWA 镜像中的 post-train 提交 `b91628dc3d55407ecb0fb4a19b26adf6d2c3ff32`。输入使用块首三路 RGB、任务文本和当前关节状态。每套新增参数在削黄瓜、鸡蛋翻面、切芹菜、擦盘子四任务间共享。训练与推理使用同一份冻结的 Joint96，动作输出为两臂关节和夹爪的 14 维绝对目标。

## 按架构和实验配置运行

| 架构 | 块首条件 | 积分变量 | Recipe |
|---|---|---|---|
| `action_plan_residual_ode` | 映射到厨房坐标的 Phoenix 动作块 | 相对原计划的残差 | [动作计划](../configs/ode_policy/recipes/kitchen_action_plan_residual.yaml) |
| `action_token_ode` | 末步投影前的整段 action tokens | 14 维绝对参考 | [动作序列](../configs/ode_policy/recipes/kitchen_action_tokens.yaml) |
| `action_query_ode` | 可学习 query 单次读取当前条件 | 14 维绝对参考 | [单次 query](../configs/ode_policy/recipes/kitchen_action_query.yaml) |

主实验在 `action_token_ode` 下选择关节历史反馈。每份 recipe 显式指定查询和注入位置，五组复用[同一训练协议](../configs/ode_policy/protocols/kitchen_action_token_posttrain.yaml)。

| 反馈进入的位置 | 动作特征查询 | 步号查询 |
|---|---|---|
| 速度场输入 | [主方案](../configs/ode_policy/recipes/kitchen_action_tokens_action_feature_field_condition.yaml) | [步号查询速度场](../configs/ode_policy/recipes/kitchen_action_tokens_block_step_field_condition.yaml) |
| 动作速度残差 | [动作特征查询残差](../configs/ode_policy/recipes/kitchen_action_tokens_action_feature_velocity_residual.yaml) | [步号查询残差](../configs/ode_policy/recipes/kitchen_action_tokens_block_step_velocity_residual.yaml) |

[直接输入当前关节的基线](../configs/ode_policy/recipes/kitchen_action_tokens_direct_feedback.yaml)使用相同的一步去噪和后训练预算。已有的小样本配置仍保留各自的去噪步数和更新数。

以下命令从仓库根目录运行。训练示例在单进程 RGB 入口做一次小批量功能检查。

```bash
recipe=configs/ode_policy/recipes/kitchen_action_tokens_action_feature_field_condition.yaml
python -m wam.ode_policy.runner resolve --recipe "$recipe"
common=(--model-config "$PHOENIX_MODEL_CONFIG" --phoenix-weights "$PHOENIX_WEIGHTS" --data-root "$KITCHEN_DATA_ROOT")
python -m wam.ode_policy.runner train --recipe "$recipe" "${common[@]}" --steps 1 --batch-size 2 --output runs/joint-history-smoke.pt
python -m wam.ode_policy.runner eval "${common[@]}" --head-weights runs/joint-history-smoke.pt --windows 3
```

最后一条命令读取 checkpoint 内的协议。对每个窗口只提取一次条件。历史反馈配方配对回放 `block_start`、`latest_joint_each_tick`、`one_tick_delayed`、`feedback_off` 和 `hold_latest`。另输出共同参考下的单步变化及相对专家的误差。默认采用该 head 关闭反馈后的参考。做跨模型比较时，给各组的 eval 命令传同一个 `--reference-head-weights <直接反馈基线.pt>`，统一使用该基线的块首轨迹及采样种子。单项评估可使用 `--mode latest_joint_each_tick`。部署从同一 checkpoint 恢复，然后调用 `full_chunk()` 或由控制器逐步调用 `next_target(k, measured, last_sent)`。新 checkpoint 保存架构全名、解析后的 recipe、实际训练步数和 batch、父模型指纹、输入状态统计，以及动作计划使用的尺度映射。旧 checkpoint 按保存的 family、映射和统计恢复。

已有块首 tokens 时，使用同一 factory 单独构造 head。下面四个张量依次为投影前 tokens 和按厨房坐标归一化的实测序列、上一参考、专家目标，形状分别为 `[B,32,1024]`、`[B,32,14]`、`[B,14]`、`[B,32,14]`。全量分片保存的是物理量，读取端应用训练分位数后再传入后三项。

```python
from wam.ode_policy.factory import build_head
from wam.ode_policy.recipe import resolve_recipe

spec = resolve_recipe("configs/ode_policy/recipes/kitchen_action_tokens_action_feature_field_condition.yaml")
head = build_head(spec, token_width=tokens.shape[-1], device=tokens.device)
loss = head.path_loss(tokens, measured, last_reference, expert_targets, feedback=True)
loss.backward()
```

复制动作序列 recipe，将 `conditioning.token_reduction` 从 `none` 改成 `temporal_mean_repeat`，用新输出路径训练。均值消融先沿 32 个时间位置取均值，再重复至原长度，随后执行相同投影、位置编码和 attention。单进程 RGB 入口的两组可共享 `--action-token-cache-dir` 中的逐窗 `.pt` tokens。训练步数、batch 和采样设置沿用同一份 recipe。

残差 recipe 默认用 `source_physical`。使用拟合映射时，改为 `action_prior.mapping: fitted_affine`，并在同一区段用 `audit`（相对 recipe 的路径）指定已经生成的训练集校准文件，或传入 `--prior-audit`。三组使用相同的父模型 state 编码。仅动作计划使用 action 解码和映射。

## 旧命令对应关系

| 训练 family | 评估 mode | 先验及执行方式 |
|---|---|---|
| `c1` | `fixed`、`feedback` | 块首完整去噪一次，保存末步输出投影前的整段 action tokens |
| `c2` | `fixed`、`feedback` | 块首用单个可学习 query 读取一次当前条件，复用同一条件向量 |
| 两种 C head | `paired` | 一次读取条件，在相同窗口用同一 checkpoint 评估固定、逐步反馈和保持最新姿态 |
| `ab` | `phoenix` | 原始 Phoenix 动作计划，对照组 |
| `ab` | `a` | 使用原计划和块首关节状态，沿整段时域积分残差 |
| `ab` | `b` | 使用同一份计划和同一套权重，每个控制周期读取最新实测关节，只生成下一个目标 |
| `ab` | `b_stale` | 同样逐步调用，持续使用块首关节状态 |

上述代号用于复现既有实验。当前主对照采用上表的查询方式与注入位置四种组合。旧的 `--mode c1` 和 `--mode c2` 仍对应各自的固定状态执行。

## 配置和数据

[四任务 YAML](../configs/ode_policy/kitchen.yaml) 保存切分、任务文本、网络宽度、训练超参和求解器配置。R1 Lite 启动配置给出了 512 宽度的关节速度场、32 的默认 batch 和 1e-4 学习率。C2 借鉴 ImageWAM 的单 query 和参考初值接口，梯度穿过冻结的 action expert 回到 query。本工程的 H=32、30 Hz、冻结骨干和上一轮预测目标作为训练边界属于 Phoenix 实验设置。首版将两夹爪与双臂一起积分，按维度归一化后统一优化路径 MSE。[R1 Lite 启动配置](https://github.com/NTUMARS/ODE-POLICY/blob/real-r1lite/scripts/train_r1lite.sh) · [R1 的参考接续约定](https://github.com/NTUMARS/ODE-POLICY/blob/real-r1lite/ode_policy/real/README.md) · [ImageWAM 实现](https://github.com/NTUMARS/ODE-POLICY/blob/wam-imagewam/wam/src/ode_imagewam/model.py) · [ImageWAM 官方训练条件](https://github.com/NTUMARS/ODE-POLICY/blob/wam-imagewam/docs/wam_libero/README.md)

数据根目录下需有 YAML 列出的六个来源，每个包含 `replay_buffer.zarr`。reader 仅读当前帧的原始三视角 RGB、`state_30hz` 和 `action_30hz` 的前 14 维。训练 496 个 episode，留出 22 个。窗口以第 k 帧为起点，监督 action[k ... k+31]，对应 state[k+1 ... k+32]；末行的重复终点排除在监督之外。非首帧用同一 episode 的 action[k−1] 作为上一目标的代理，首帧用实测关节。使用训练 episode 动作的 1%/99% 分位拟合 14 维坐标变换，正常数值范围线性映射到 [-1,1]，保留超出范围的动作。各 ODE head 共用厨房坐标；冻结 Phoenix 的输入状态另按所选来源的 state 统计变换后填入 canonical80。A/B 的动作先验按来源 action 统计逆变换到物理空间再转入厨房坐标，也可使用仅在训练 episode 上拟合的逐维仿射映射。

A/B 初始输出等于原计划。默认先训练 100 个整块更新，然后以 0.5 概率抽块内边界；其中等概率选取示范上一参考或模型预测前缀的末目标。前缀预测停止梯度，后缀从当前实测关节取得条件，只监督剩余有效步。C1/C2 从绝对参考递推 32 步。每个训练窗口以 `train.feedback_probability` 的概率逐步使用 q[k]，其余窗口复用 q[0]。默认 0.5，设为 0 或 1 可单独训练一种方式。两种方式均把模型预测接到下一步，使用相同全轨迹 MSE。checkpoint 中的 `training_windows` 记录实际混合数。C2 的单次读取使用固定的干净视觉时刻和零动作扩散时刻；ODE 每次调用使用块内实际进度。

## 父权重与先验尺度

上海已核实的公开格式权重位于 `/mnt/data/lcy/Phoenix-MWA/resources/robotwin/pretrain_256013.pt`，SHA-256 为 `5ed256c1d34650d689639b55560f243ffd90209640a201c76b13094087e2464f`。原始训练文件的 SHA-256 为 `f1a2749ff95253b098476b2b6bad9641b0ddea0183c4f366214a63d96bc29753`。两份文件实测 1,650 个 MoT 张量和 3 个 proprio 张量全部相等。本仓库加载器使用公开格式及其匹配的发布版编码器配置。来源统计取自同一训练 run 配置里的 `/mnt/data/lcy/InternW0-Delta/artifacts/local_robot37/full_20261003_v4/norm.json`，其 SHA-256 与 `_NORM_COMPLETE` 一致。[三个 ARX5/F15 统计组](../configs/ode_policy/robot37_256013_arx5.json)

F15 的双臂关节和夹爪监督是下一观测的位置。相同的 embodiment ID 可覆盖多个 stats_group；厨房示范没有唯一对应的源组。命令示例选择训练样本数最多的 F15/ARX5 作为候选输入约定。保持该 state 编码及同一噪声种子，对每个留出窗口只运行一次冻结的 Phoenix，同时比较保持当前姿态、将原始槽位直接解释为厨房坐标、按来源 action 统计解码、训练集仿射校准四条轨迹。拟合映射为每维独立的斜率和截距，所有任务及时间步共享。若省略源统计，诊断沿用厨房输入作为显式假设，只比较保持、直接取槽位、拟合后三种输出；输出校准不能反推正确的 state 编码。

每个窗口报告各控制时刻的关节 RMSE、夹爪 RMSE、相对当前姿态的运动方向命中数。方向仅统计专家位移大于该维训练分位跨度 1% 的项。夹爪事件以训练分位中点的首个正向和负向穿越为代理，分别统计两指的匹配时间差、漏检和多检；无匹配事件时不计算时机误差。该指标描述开度越阈时刻。报告与映射保存在 audit 文件。不同候选 state 统计需要分别生成诊断，避免用最终留出结果调参。

## 训练与回放

使用 Python 3.11 和 Phoenix 环境，安装本仓库的 `ode` extra。父模型需要匹配的 Joint96 配置、预训练权重、视觉编码器及 Wan 资产。模型配置可取父 run 的完整解析 YAML 或单独的 `model` 节点，路径须在运行机器上有效；加载器严格核对父权重的模型身份和张量。文本可使用厨房四任务原有 `text_cache.pt`（顶层 `embeddings`，按任务提供 `context` 和 `context_mask`）；省略该参数则由父模型 T5 对四条任务指令各编码一次。[父模型资产说明](RESOURCES.md)

从仓库根目录执行以下命令。先将环境变量指向实际父模型文件。第一次用一个更新验证加载、前反向和落盘；正式预算见 YAML。

```bash
pip install -e '.[ode]'
common=(--data-root /mnt/workspace/zyh/data --model-config "$PHOENIX_MODEL_CONFIG" --phoenix-weights "$PHOENIX_WEIGHTS")
profile=(--source-profile configs/ode_policy/robot37_256013_arx5.json --source-group raw36/arx456_xt_motion_dedup_20260916/arx5)
for family in c1 c2; do
    python -m wam.ode_policy.runner train "${common[@]}" "${profile[@]}" --family "$family" --output "runs/$family.pt"
    python -m wam.ode_policy.runner eval "${common[@]}" --head-weights "runs/$family.pt" --mode paired --windows 3
done

# 次级动作先验对照
python -m wam.ode_policy.runner audit-prior "${common[@]}" "${profile[@]}" --fit-windows 1 --windows 3 --output runs/ab-prior.pt
python -m wam.ode_policy.runner train "${common[@]}" --prior-audit runs/ab-prior.pt --prior-kind fitted --family ab --steps 1 --batch-size 1 --output runs/ab-smoke.pt
python -m wam.ode_policy.runner eval "${common[@]}" --head-weights runs/ab-smoke.pt --mode b --windows 1
```

A/B 可用 `--prior-kind native` 复用审计中的来源统计，或用 `--prior-kind fitted` 使用其训练集映射。C1/C2 可独立传入相同的 `--source-profile` 和 `--source-group`；两组只使用 state 编码。显式 `--raw-slot-control` 可复现旧的厨房尺度假设。部署时用 `KitchenPolicy.from_checkpoint` 恢复训练时的 state 和 action 映射。

重复 C1 训练时可以传入 `--c1-cache-dir runs/c1-tokens`。缓存保存冻结骨干输出投影前的原始 tokens，按父模型身份、时域、去噪设置、种子和实际条件内容区分。C2 每个窗口都重新计算 query 条件并反向传播；计时回放始终现场计算块首条件。

单进程 RGB 训练入口使用各自的 `--recipe` 和独立输出文件。使用旧式数据配置时也可显式传 `--architecture action_token_ode` 等架构全名。相同随机种子产生相同训练窗口顺序。每个样本独立做 Phoenix 前向，用梯度累积达到配置中的有效 batch。checkpoint 保存新增权重、实际更新数、归一化、配置和父配置/权重/文本缓存指纹。训练不覆盖已有输出。评估每个留出 episode 选相同的均匀分布窗口，逐任务报告 12 臂关节和 2 夹爪的原生单位 RMSE，各 episode 等权。默认每条轨迹三个窗口。

配对回放的 `hold_latest` 直接输出各步 q[k]。在该数据中增量误差 (预测−q[k])−(标签−q[k]) 等于位置误差，因此另报活动关节的增量方向、预测及专家的平均绝对步长。按关节、按窗口报告活动范围与主要转折。活动阈值取该关节训练分位跨度的 1%（下限 1e-6）。主要转折取相对两端均有最大突出幅度的内部峰或谷，超过阈值才计数，比较同类型转折的时刻。夹爪沿用每窗、每侧、每方向首次越过训练区间中点的计数。时机指标都附分母，零匹配显示 null。位置 RMSE 按 episode 等权，运动统计按有效样本或事件计权。

四列回放延迟分别为观测预处理与条件读取、head 至整块物理目标、两者合计、首个可用物理目标。配对模式复用块首耗时，反馈还报告单 tick 至 CPU 目标的 P50/P95 和超周期比例。离线回放使用示范状态。真机传感器读取、通信和控制器接收耗时由现场接口测量。

## 在线调用

`wam.ode_policy.policy.KitchenPolicy` 接收已经严格加载的 Phoenix、训练好的 head、checkpoint 中的归一化坐标和四任务文本条件。调用方持有传感器状态、控制队列及上一条已发参考。每个新块创建一个 block，重置或交接时丢弃旧 block。

```python
block = policy.begin_block(task, rgb, measured_q, last_sent)
chunk = block.full_chunk()  # A/C1/C2，形状 [32, 14]
```

B、C1 和 C2 的首帧没有既有目标时令 last_sent 等于实测状态。每次获得新关节状态后只计算一个目标，按实际成功下发的目标推进 k 和 last_sent。

```python
block = policy.begin_block(task, rgb, measured_q, last_sent)
target = block.next_target(k, latest_measured_q, last_sent)  # 形状 [14]
```

A/B 块内始终复用原计划 p。在第 k 个边界用 r = last_sent − p[k] 接续，块首 p[0] 定义为块首参考。32 次反馈各执行一个 Euler 步。该接口返回物理目标张量，机器人通信由部署侧持有。

### 真机配对记录

首批用 C1，相同任务、初始物体布局、工具、checkpoint 和块首种子作为一对，按对交替先运行 fixed 或 feedback。每次 episode 各自重置，沿用现场控制器的时钟、限位和接管。上线前先看留出集的活动范围、转折与夹爪事件，再在部署机影子执行检查 30 Hz 周期。四任务分别固定阶段判据，例如削黄瓜的有效削皮行程、翻蛋的翻面、切芹菜的切断、擦盘子的有效覆盖。现场评分按同一组布尔阶段标签和任务成功判据记录。

```python
from wam.ode_policy.trial import KitchenTrial, summarize_pairs

trial = KitchenTrial(policy, task, mode, pair_id, head_checkpoint_sha256, last_sent=last_sent)
trial.begin_block(rgb, measured_q)             # 每个块开始时调用
trial.tick(read_state, send_and_ack_target)    # 每个控制 tick 调用一次
# 安装在现场控制器内。read_state 返回最新 14 维张量，
# send_and_ack_target 返回实际已接受的 14 维物理目标（含控制器限幅）。
row = trial.result(success, stage_outcomes)    # 每个 episode 保存一条 JSON
report = summarize_pairs(completed_trial_rows)
```

`tick` 按成功下发推进参考。下发异常后应由控制器核对最后一条已提交的目标再建立新 trial。计时从 tick 入口到下发确认返回，逐步反馈包含读取关节状态的耗时，固定模式直接下发已规划目标，另记块首输入至首目标就绪的延迟。配对汇总按任务、权重和模式列出成功数、各阶段完成数、下发异常数、P50/P95 和 33.3 ms 超周期率。真机报告需保留每个配对的原始记录。

## 软件验证

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_ode*.py'
```

测试使用缩小维度的原生 Phoenix video/action expert、MoT 和 scheduler，替代视觉 codec 输入；覆盖共享 head、B 的速度场调用次数、C1 与原采样的一致性、C2 的单次读取与 query 梯度，以及小样本的训练、保存和回放。

2026-11-20，使用本仓库 reader 在上海只读打开全部六个来源，H=32 得到 623,427 个训练窗口和 31,621 个留出窗口。按来源与拆分别抽读一条 episode 的首个及最后一个合法窗口，共 18 个，验证三视角形状、14 维动作对齐和首尾边界；训练集分位统计为有限值。以上检查使用原始 Zarr。软件测试中的训练和回放使用小模型，完整父权重与真机执行尚未在本工程中验证。

### 256013 冻结先验的首次真实权重检查

2026-11-20 在上海加载上述发布权重，使用配套的三个 RGB 槽位、DINO/CS-VAE 和现场 T5 编码。选择 `raw36/arx456_xt_motion_dedup_20260916/arx5` 的 **state** 分位数输入本体状态。每任务取 4 条训练 episode 的中间窗口拟合一个共享的逐维仿射映射，每条留出 episode 取 1 个中间窗口。四种输出解释共用观测、原始槽位和随机种子。全程关闭 ODE 残差。

| 22 个窗口，按 episode 汇总 | 12 个臂关节 RMSE（rad） | 关节误差低于保持姿态的窗口 |
|---|---:|---:|
| 保持块首实测姿态 | 0.08093 | 基准 |
| 槽位数值直接当厨房坐标 | 0.3403 | 0/22 |
| 用来源 action 统计还原 | 0.08153 | 5/22 |
| 用 16 个训练窗口拟合仿射映射 | 0.08057 | 12/22 |

翻蛋的仿射校准误差为 0.08195 rad，保持姿态为 0.09255；黄瓜的对应结果为 0.05984 和 0.05074。跨任务的轨迹收益尚未确立。22 个窗口只有一个真实夹爪过阈事件，四种输出均未捕获，无法估计匹配事件的开合时间差。该检查只覆盖各示范的一个相位；可在固定定义下增加窗口，分开评估明显运动、接触与开合阶段。

## 2026-11-21 实验记录与 TODO：C1 参考恢复和逐拍反馈

C1 学习参考目标的速度场，以 30 Hz 递推 32 步，用整条 14 维参考轨迹的 MSE 监督。在线跨块从控制器确认接受的上一目标接续，同时单独传入实测状态。当前配方从示范参考起步，每个窗口以 0.5 概率使用逐拍实测，其余窗口复用块首状态。八窗口拟合则对每个窗口同时计算两路损失，各占一半。两路均使用原始专家轨迹，夹爪与臂关节共同积分。[积分与损失](../src/wam/ode_policy/head.py) · [C1 配方](../configs/ode_policy/recipes/kitchen_action_tokens.yaml) · [控制器确认逻辑](../src/wam/ode_policy/trial.py)

- [x] **建立八窗口 C1 的反馈基线。** 用拟合 1,200 次更新的同一份权重，回放拟合窗口、同 episode 的其余窗口及 22 条独立留出 episode。留出集各任务内，取块首模式的参考目标与示范实测状态偏差最高的四分之一时刻。固定 tokens、上一参考和块内步号后，最新状态相对块首状态仅降低下一目标的臂关节 RMSE 0.0043°。三路整块各自递推时，最新、滞后一拍、块首的 RMSE 分别为 20.1119°、20.1160°、20.1573°。这些状态来自专家示范。工作区实验记录保存在与仓库并列的 `artifacts/phoenix-c1-q-feedback-20261121/README.md` 和 `artifacts/phoenix-c1-fit-episode-feedback-20261121/README.md`。
- [ ] **检验初始参考偏移后的恢复能力。** 以当前无扰动 C1 为对照，按 ImageWAM 的设置试验：以 0.5 概率扰动块首参考，令 \(r'_0=r_0+\delta\)，第 \(j\) 步的监督改为 \(r_j^{*\prime}=r_j^*+0.7^j\delta\)，\(j=1,\ldots,32\)。实测序列和块首 tokens 沿用同一示范。实验前记录扰动维度、单位及按关节选取的幅度，两组匹配训练窗口、预算和反馈采样。在相同留出窗口上施加同一批偏移，报告各步对恢复目标的误差、回到原示范的误差曲线，以及零扰动时的轨迹质量。[ImageWAM 扰动配置](https://github.com/NTUMARS/ODE-POLICY/blob/wam-imagewam/wam/src/ode_imagewam/configs/libero_ode.yaml) · [恢复监督实现](https://github.com/NTUMARS/ODE-POLICY/blob/wam-imagewam/wam/src/ode_imagewam/model.py)
- [ ] **优先用正式四任务 C1 权重复测逐拍反馈。** 先在原留出窗口上固定 tokens、上一参考 \(r_k\) 和步号 \(k\)，仅替换最新 \(q_k\)、滞后一拍 \(q_{k-1}\) 或块首 \(q_0\)，记录下一目标的变化量和对专家动作的误差。随后从同一初值做三路整块递推，分别报告全部时刻和高偏差时刻的逐任务结果，保留保持最新姿态基线。若完成扰动训练，对新权重重复同一评估，分别记录参考恢复收益和最新实测带来的收益。真机闭环收益沿用上文的配对协议验证。

### 独立消融：用逐拍反馈修正 C1 latent

- [x] **实现独立配方与动作传递检查。** 从直接输入关节状态的 C1 checkpoint 开始，冻结 Phoenix、token reader 和速度场，只训练两层 MLP 适配器。每拍从同一份块首 tokens 读取当前条件 $h_k$，用 $h_k,r_k,q_k,r_k-q_k,k/32$ 预测 $\Delta h_k$。冻结的速度场读取 $h_k+\Delta h_k$、块首实测 $q_0$、上一已下发参考 $r_k$ 与块内进度，用一个 Euler 步输出下一物理目标。每拍独立修正 latent，预测目标沿时间接续。适配器末层零初始化，训练开始时可复现原 C1 固定块首状态的动作。[latent 配方](../configs/ode_policy/recipes/kitchen_action_tokens_latent_feedback.yaml) · [直接反馈配方](../configs/ode_policy/recipes/kitchen_action_tokens.yaml)
- [x] **八窗口拟合消融（2026-11-21）。** 从原拟合 C1 各追加 1,200 次更新，两个分支共用训练窗口、顺序和学习率。采用原 C1 共同上一参考，在每任务偏差最高的四分之一时刻（共 64 拍），latent 将块首反馈换为最新反馈时下一目标改变 0.01039° RMS，整块高偏差 RMSE 从 1.00046° 升至 1.02164°。等更新数的直接反馈 C1 单步改变 0.01177°，整块高偏差 RMSE 从 0.93081° 降至 0.92066°。两组全部 256 拍的最新反馈 RMSE 分别为 0.73047° 和 0.66797°。这八窗口中没有时刻达到独立留出集各任务的高偏差阈值。工作区原始轨迹、权重和报告在 `artifacts/phoenix-c1-latent-fit8-20261121/`。
- [x] **八窗口从头训练配对消融（2026-11-21）。** 两种反馈路径各训练完整 head，三个种子分别配对，共享参数初值和窗口顺序逐组一致。每组 2,400 次更新。直接反馈和 latent 的全时刻最新状态 RMSE 分别为 0.63920° 和 0.63282°。沿用相同 64 个高偏差时刻，最新状态相对各自块首状态的整块 RMSE 收益分别为 +0.01057° 和 −0.01781°，三个种子的收益符号均一致。详见工作区 `artifacts/phoenix-c1-scratch-fit8-20261121/README.md`。
- [ ] **用正式四任务权重完成配对消融。** 在相同窗口、初值、tokens 和种子下比较下表各路。新增适配器有额外的训练预算。报告起始 C1 的 checkpoint 指纹、适配器更新数及逐任务结果。若比较同预算微调的最终性能，再给直接反馈 C1 同等追加预算。
- [ ] **验证动作变化的方向和大小。** 配对回放逐窗输出 `diagnostic=shared_reference_feedback`。每拍固定原 C1 块首模式生成的上一参考、tokens 和步号，只切换适配器的 $q_0/q_k$。`target_change_rms` 衡量下一动作的实际变化。`fixed_rmse` 和 `latest_rmse` 分别衡量两种动作对下一拍示范的误差。臂用弧度、夹爪用原开度单位。结合整块各自递推的 RMSE、转折和夹爪事件评估收益。高偏差子集沿用上节的共同参考和分组定义，另行分层统计。软件因果检查从 $k=10$ 才替换实测，前十个动作应逐元素一致，已具备非零反馈响应的适配器应在第十一个动作出现变化。[回放实现](../src/wam/ode_policy/runner.py) · [回归测试](../tests/test_ode_recipes.py)

| 配对回放模式 | 适配器输入 | 速度场直接读取的实测 | 用途 |
|---|---|---|---|
| `adapter_off` | 关闭 | 块首 $q_0$ | 复现起始 C1 的固定状态轨迹 |
| `direct_c1` | 关闭 | 最新 $q_k$ | 复现同一起始 C1 原有的直接反馈轨迹 |
| `block_start` | 块首 $q_0$ | 块首 $q_0$ | 量化新增参数在没有新观测时的贡献 |
| `latest_joint_each_tick` | 最新 $q_k$ | 块首 $q_0$ | 与上一行相减，量化最新反馈经 latent 传递到动作的贡献 |
| `hold_latest` | — | — | 直接输出示范当前实测姿态 |

2026-11-21 本地兼容性检查使用已有八窗口 C1 权重（SHA-256 前缀 `d6158f11a957`）、一条保存的示范状态序列和固定随机 tokens。在相同输入下，`adapter_off`、`direct_c1` 分别与原 head 对应路径逐元素一致。零初始化的适配器也精确复现块首路径。更新一次适配器后，原 head 权重逐元素保持一致，固定参考时替换反馈可产生非零关节目标变化。此检查覆盖动作传递和旧权重兼容性，留出集动作质量另按上述 TODO 评估。

沿用“按架构和实验配置运行”中的 `common` 参数。训练需指定实际的起始 C1 权重。生成的 checkpoint 保存冻结 head、适配器和起始权重 SHA-256，可直接用于回放及原有在线接口。

```bash
recipe=configs/ode_policy/recipes/kitchen_action_tokens_latent_feedback.yaml
python -m wam.ode_policy.runner train --recipe "$recipe" "${common[@]}" \
  --base-c1-weights "$C1_CHECKPOINT" --output runs/c1-latent-feedback.pt
python -m wam.ode_policy.runner eval "${common[@]}" --head-weights runs/c1-latent-feedback.pt --mode paired
```

## 2026-11-22 实验记录：C1 去噪步数

在原四任务八窗口上，分别用 1、2、5、10 步原生联合去噪的末次前向提取 32 个 action tokens。每档三个配对种子，各从头训练直接反馈 C1 head 2,400 次。本轮同卡交错计时中，首目标 P50 依次为 403、678、1294、2272 ms。逐拍反馈下，1 步和 10 步的平均活动关节幅度比分别为 1.014 和 1.027，匹配主要转折分别为 12/12/11 与 11/11/11（每个种子共 13 个）。1 步的转折时机误差比 10 步多 0.20 拍。训练与评估使用相同窗口和采样噪声，独立留出集仍待验证。工作区的逐种子结果、额外转折统计、原始计时与权重在 `artifacts/phoenix-c1-denoise-fit8-20261122/`。

## 四拍关节反馈的主实验设计

每块用冻结的 Joint96 step 256013 做一步联合去噪，保留全部 32 个输出投影前的 action tokens。每拍从整段 tokens 读出当拍特征 $h_k$。四个实验都用 $h_k$、块首实测 $q_0$、当前参考 $r_k$ 和步号预测名义动作。每个实验从头训练新增 head，各任务共享。直接将 $q_k$ 输入速度场的 C1 单列为基线。

| 反馈进入动作的位置 | 用当拍动作特征查询历史 | 用块内步号查询历史 |
|---|---|---|
| 速度场条件 | 以 $h_k$ 为 query，读出的 $c_k$ 作为速度场的额外输入 | 以可学习的步号嵌入为 query，其余相同 |
| 动作空间速度残差 | 以 $h_k$ 为 query，将 $c_k$ 解码为 14 维速度，加到名义速度上 | 以可学习的步号嵌入为 query，其余相同 |

历史保留本块已观测的最近最多四条 $(q_i,r_i-q_i)$，每条带相对拍龄。$r_i$ 是取得 $q_i$ 时已经生效的参考。训练时使用该路当时递推的参考，在线时使用控制器确认接受的目标。第 $k$ 拍读取历史后仅推进 $r_{k+1}=r_k+v_k/30$。关闭反馈时，条件组使用 $c_k=0$，残差组显式跳过残差解码器，两组各自复现本模型的名义输出。固定块首实测是另一项实验，反馈读取仍开启。

对照约定如下。最新反馈使用截至 $k$ 的历史。滞后一拍只使用截至 $k-1$ 的至多四条同拍实测与参考，并按原始时间戳计算拍龄。每块首拍三路均可见 $q_0$。固定块首实测沿用八窗口原型的做法，在每个历史槽输入 $(q_0,r_i-q_0)$。此时的差值表示当前参考相对保持读数的偏移。关闭反馈单独检查名义路径的一致性。

在同一批留出窗口上做两种回放。单步诊断固定窗口 tokens、步号和整段共同参考历史，分别记录下一目标的改变幅度和相对专家的误差改善。整块回放从同一参考起步，三路各自递推 32 拍。留出集分析沿用每任务共同参考与真实示范状态的臂关节偏差最高四分之一作为共享高偏差子集，保留保持最新姿态基线。同行比较反馈查询方式，同列比较注入位置。四组已接入 factory 并在八窗口缓存上验证前反向。留出集收益和真机闭环收益待测。

### 2026-11-23 八窗口配对拟合

在乌兰察布 A100 上，用一步去噪的原八个窗口从头训练四种历史反馈配置和直接输入关节的基线。每组用种子 42、43、44 分别更新 2,400 次。同种子共享名义分支初值和窗口顺序，每窗训练 300 次。评估沿用预先固定的共同参考和 64 个高偏差时刻。

在相同参考下，最新实测让五组下一目标的臂关节 RMSE 平均降低 0.00067°–0.00122°。各自递推 32 拍后，直接反馈的高偏差 RMSE 比固定实测低 0.00170°，三个种子方向一致。动作特征和步号查询的速度场条件组分别升高 0.00541°、0.00102°，两种动作速度残差组均升高约 0.0214°。四种历史配置下，最新实测平均优于滞后一拍。八窗拟合尚未显示动作特征查询能稳定提高实时纠偏收益。[实验记录、逐种子指标与权重](../../artifacts/phoenix-joint-history-fit8-20261123/README.md)

## 全量缓存后训练的 batch 计划

四任务一步去噪缓存核验后，按 8 卡同步训练 C1 head。正式实验使用全局 batch **1,024 个窗口**，每卡 128 个窗口，一次同步更新，无梯度累积。逐拍反馈的四组消融及直接输入实测状态的 C1 基线使用相同 batch、采样顺序和优化预算。同一窗口分别用固定和逐拍实测展开时，样本预算只计一次。

正式训练前，在相同 batch 下从真实共享分片短跑。预热后记录稳定窗口吞吐、单次更新耗时和显存峰值，并检查 loss 有限、跨卡梯度同步和 checkpoint 续训。按这次短跑的端到端速度估算全量耗时。

每组 40,000 次 optimizer 更新，共处理 4,096 万个窗口样本，含重复采样。在第 1,250 和 5,000 次更新保存检查点，分别对齐先前 batch 32 和 128 各训练 40,000 次的窗口预算（128 万和 512 万）。

2026-11-23 在原八个拟合窗口上测了两路配对展开、FP32 全路径反传与 AdamW 更新。上海 L20Z 的单进程分配器限额为 24 GiB，使用当拍 token 查询的两种四拍历史反馈原型均通过每卡 4,864 窗，下一测试档 5,120 窗 OOM。乌兰察布 A100 八卡 DDP 在全局 4,096 窗通过，历史反馈原型每卡峰值保留显存约 2.50 GiB。输入常驻 GPU，实际共享分片读取吞吐要另测。原始结果位于工作区 `artifacts/phoenix-c1-batch-vram-20261123/README.md`。
