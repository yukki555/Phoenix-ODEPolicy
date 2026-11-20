# Phoenix 四任务动作 ODE 后训练

基线为本地 Phoenix-MWA 镜像中的 post-train 提交 `b91628dc3d55407ecb0fb4a19b26adf6d2c3ff32`。输入使用块首三路 RGB、任务文本和当前关节状态。每套新增参数在削黄瓜、鸡蛋翻面、切芹菜、擦盘子四任务间共享。训练与推理使用同一份冻结的 Joint96，动作输出为两臂关节和夹爪的 14 维绝对目标。

| 训练 family | 评估 mode | 先验及执行方式 |
|---|---|---|
| `c1` | `fixed`、`feedback` | 块首完整去噪一次，保存末步输出投影前的整段 action tokens |
| `c2` | `fixed`、`feedback` | 块首用单个可学习 query 读取一次当前条件，复用同一条件向量 |
| 两种 C head | `paired` | 一次读取条件，在相同窗口用同一 checkpoint 评估固定、逐步反馈和保持最新姿态 |
| `ab` | `phoenix` | 原始 Phoenix 动作计划，对照组 |
| `ab` | `a` | 使用原计划和块首关节状态，沿整段时域积分残差 |
| `ab` | `b` | 使用同一份计划和同一套权重，每个控制周期读取最新实测关节，只生成下一个目标 |
| `ab` | `b_stale` | 同样逐步调用，持续使用块首关节状态 |

C1 同一 checkpoint 的固定与反馈执行是首轮主对照。C2 使用同一协议，同时比较从块首观测到首个可执行目标的延迟。旧的 `--mode c1` 和 `--mode c2` 仍对应各自的固定状态执行。A/B 作为显式动作先验对照保留。

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

正式比较分别以 `--family ab`、`--family c1`、`--family c2` 训练，使用独立输出文件。相同随机种子产生相同训练窗口顺序。每个样本独立做 Phoenix 前向，用梯度累积达到配置中的有效 batch。checkpoint 保存新增权重、实际更新数、归一化、配置和父配置/权重/文本缓存指纹。训练不覆盖已有输出。评估每个留出 episode 选相同的均匀分布窗口，逐任务报告 12 臂关节和 2 夹爪的原生单位 RMSE，各 episode 等权。默认每条轨迹三个窗口。

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

`tick` 按成功下发推进参考。下发异常后应由控制器核对最后一条已提交的目标再建立新 trial。计时从读取关节状态前到下发确认返回，另记块首输入至首目标就绪的延迟。配对汇总按任务、权重和模式列出成功数、各阶段完成数、下发异常数、P50/P95 和 33.3 ms 超周期率。真机报告需保留每个配对的原始记录。

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
