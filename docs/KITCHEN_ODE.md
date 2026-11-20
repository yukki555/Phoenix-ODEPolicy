# Phoenix 四任务动作 ODE 后训练

基线为本地 Phoenix-MWA 镜像中的 post-train 提交 `b91628dc3d55407ecb0fb4a19b26adf6d2c3ff32`。输入使用块首三路 RGB、任务文本和当前关节状态。每套新增参数在削黄瓜、鸡蛋翻面、切芹菜、擦盘子四任务间共享。训练与推理使用同一份冻结的 Joint96，动作输出为两臂关节和夹爪的 14 维绝对目标。

| 训练 family | 评估 mode | 先验及执行方式 |
|---|---|---|
| `ab` | `phoenix` | 原始 Phoenix 动作计划，对照组 |
| `ab` | `a` | 使用原计划和块首关节状态，沿整段时域积分残差 |
| `ab` | `b` | 使用同一份计划和同一套权重，每个控制周期读取最新实测关节，只生成下一个目标 |
| `ab` | `b_stale` | 同样逐步调用，持续使用块首关节状态 |
| `c1` | `c1` | 完整去噪，读取最后一次输出投影前的整段 action token，再积分生成绝对参考 |
| `c2` | `c2` | 单个可学习 query 与当前帧走一次冻结的 MoT，从条件向量积分生成绝对参考 |

## 配置和数据

[四任务 YAML](../configs/ode_policy/kitchen.yaml) 保存切分、任务文本、网络宽度、训练超参和求解器配置。R1 Lite 启动配置给出了 512 宽度的关节速度场、32 的默认 batch 和 1e-4 学习率。C2 借鉴 ImageWAM 的单 query 和参考初值接口，梯度穿过冻结的 action expert 回到 query。本工程的 H=32、30 Hz、冻结骨干和上一轮预测目标作为训练边界属于 Phoenix 实验设置。首版将两夹爪与双臂一起积分，按维度归一化后统一优化路径 MSE。[R1 Lite 启动配置](https://github.com/NTUMARS/ODE-POLICY/blob/real-r1lite/scripts/train_r1lite.sh) · [R1 的参考接续约定](https://github.com/NTUMARS/ODE-POLICY/blob/real-r1lite/ode_policy/real/README.md) · [ImageWAM 实现](https://github.com/NTUMARS/ODE-POLICY/blob/wam-imagewam/wam/src/ode_imagewam/model.py) · [ImageWAM 官方训练条件](https://github.com/NTUMARS/ODE-POLICY/blob/wam-imagewam/docs/wam_libero/README.md)

数据根目录下需有 YAML 列出的六个来源，每个包含 `replay_buffer.zarr`。reader 仅读当前帧的原始三视角 RGB、`state_30hz` 和 `action_30hz` 的前 14 维。训练 496 个 episode，留出 22 个。窗口以第 k 帧为起点，监督 action[k ... k+31]，对应 state[k+1 ... k+32]；末行的重复终点排除在监督之外。非首帧用同一 episode 的 action[k−1] 作为上一目标的代理，首帧用实测关节。使用训练 episode 动作的 1%/99% 分位拟合 14 维坐标变换，正常数值范围线性映射到 [-1,1]，保留超出范围的动作。三组共用此坐标变换，向冻结 Phoenix 提供的 proprio 则按其 canonical80 有效槽填充。此处使用四任务坐标系解释预训练动作先验。

A/B 初始输出等于原计划。默认先训练 100 个整块更新，然后以 0.5 概率抽块内边界；其中等概率选取示范上一参考或模型预测前缀的末目标。前缀预测停止梯度，后缀从当前实测关节取得条件，只监督剩余有效步。C1 与 C2 均从绝对参考直接积分。C2 的单次读取使用固定的干净视觉时刻和零动作扩散时刻；ODE 每次调用使用块内实际进度。

## 训练与回放

使用 Python 3.11 和 Phoenix 环境，安装本仓库的 `ode` extra。父模型需要匹配的 Joint96 配置、预训练权重、视觉编码器及 Wan 资产。模型配置可取父 run 的完整解析 YAML 或单独的 `model` 节点，路径须在运行机器上有效；加载器严格核对父权重的模型身份和张量。文本可使用厨房四任务原有 `text_cache.pt`（顶层 `embeddings`，按任务提供 `context` 和 `context_mask`）；省略该参数则由父模型 T5 对四条任务指令各编码一次。[父模型资产说明](RESOURCES.md)

从仓库根目录执行以下命令。先将环境变量指向实际父模型文件。第一次用一个更新验证加载、前反向和落盘；正式预算见 YAML。

```bash
pip install -e '.[ode]'
common=(--data-root /mnt/workspace/zyh/data --model-config "$PHOENIX_MODEL_CONFIG" --phoenix-weights "$PHOENIX_WEIGHTS")
python -m wam.ode_policy.runner train "${common[@]}" --family ab --steps 1 --batch-size 1 --output runs/ab-smoke.pt
python -m wam.ode_policy.runner eval "${common[@]}" --head-weights runs/ab-smoke.pt --mode b --windows 1
```

正式比较分别以 `--family ab`、`--family c1`、`--family c2` 训练，使用独立输出文件。相同随机种子产生相同训练窗口顺序。每个样本独立做 Phoenix 前向，用梯度累积达到配置中的有效 batch。checkpoint 保存新增权重、实际更新数、归一化、配置和父配置/权重/文本缓存指纹。训练不覆盖已有输出。评估每个留出 episode 选相同的均匀分布窗口，逐任务报告 12 臂关节和 2 夹爪的原生单位 RMSE，各 episode 等权。默认每条轨迹三个窗口。

回放计时从已读入内存的观测预处理开始，输出条件读取、head 到动作输出及总耗时 P50/P95；包括首次调用。B 单列逐 tick 的 head 耗时。B 的离线反馈来自示范实测状态，离线延迟不含相机采集、网络和控制器开销。真机成功率、实时控制周期及后缀纠偏能力需要单独评估。

## 在线调用

`wam.ode_policy.policy.KitchenPolicy` 接收已经严格加载的 Phoenix、训练好的 head、checkpoint 中的归一化坐标和四任务文本条件。调用方持有传感器状态、控制队列及上一条已发参考。每个新块创建一个 block，重置或交接时丢弃旧 block。

```python
block = policy.begin_block(task, rgb, measured_q, last_sent)
chunk = block.full_chunk()  # A/C1/C2，形状 [32, 14]
```

B 的首帧没有既有目标时令 last_sent 等于实测状态。每次获得新关节状态后只计算一个目标，按实际成功下发的目标推进 k 和 last_sent。

```python
block = policy.begin_block(task, rgb, measured_q, last_sent)
target = block.next_target(k, latest_measured_q, last_sent)  # 形状 [14]
```

块内始终复用原计划 p。在第 k 个边界用 r = last_sent − p[k] 接续，块首 p[0] 定义为块首参考。32 次反馈各执行一个 Euler 步。该接口返回物理目标张量，机器人通信由部署侧持有。

## 软件验证

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_ode_policy.py'
```

测试使用缩小维度的原生 Phoenix video/action expert、MoT 和 scheduler，替代视觉 codec 输入；覆盖共享 head、B 的速度场调用次数、C1 与原采样的一致性、C2 的单次读取与 query 梯度，以及小样本的训练、保存和回放。

2026-11-20，使用本仓库 reader 在上海只读打开全部六个来源，H=32 得到 623,427 个训练窗口和 31,621 个留出窗口。按来源与拆分别抽读一条 episode 的首个及最后一个合法窗口，共 18 个，验证三视角形状、14 维动作对齐和首尾边界；训练集分位统计为有限值。以上检查使用原始 Zarr。软件测试中的训练和回放使用小模型，完整父权重与真机执行尚未在本工程中验证。
