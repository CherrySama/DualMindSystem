# 接触输入精简第一轮

2026-10-08。用户确认优先减少冗余重复信息，本轮只修改模型输入投影，不调整任务判定或执行控制。独立协作 agent 完成了字段语义、代码及冻结请求的交叉核对。

## 问题与修改

旧输入在四个接触侧面下，各为七个动作附带完整预测几何。侧面定义、校准来源、最近点坐标、校正向量和预测边界被反复展开；原两份请求重放均返回 `max_tokens_exceeded`。

生产代码仅修改 `src/jev4mujoco/policies/jev_context.py`。每侧的当前完整 `geometry` 保留，每个动作的 `projected_contact_geometry` 改为 `projected_contact_summary`：

- 保留 `observable`；不可观测时保留 `reason` 和空 `parts`。
- 每个工具部件保留 `part_id`、有符号 `normal_gap_m`、`tangent_deficit_m`、`height_deficit_m`、`finite_patch_overlap`、`local_error_m` 和 `approach_side_status`。
- 外层动作、预期位移、预测总误差、预测进展、来源、`is_observed=false` 和固定朝向/忽略动力学与路径碰撞的假设保持原样。
- 四侧候选、当前最近点和校正向量、当前几何的非接触确认/未验证路径声明，以及阶段条件、约束证据、状态绑定保持原样。单层与两层动作输入共享这份摘要。

摘要省略的是未来最近点/校正坐标和重复侧面元数据，并非完整几何的无损编码。未来部件最近点方向的解释细节减少了；逐部件间隙、缺口、正反侧状态和动作比较依据仍保留。几何计算及 `projected_progress_m = current_error - projected_error` 公式未变，负间隙、没有重叠时的 `null`、距离计算失败时的 `null` 不补零。

## 冻结输入重放

使用上次批次保存的两份失败意图请求，仅替换逐动作预测字段；校验其余所有字段相等，原事件文件内容哈希未变。

| 请求 | 原紧凑 JSON 字节 | 精简后字节 | 服务端输入 tokens | HTTP | 合法侧面选择 |
| --- | ---: | ---: | ---: | ---: | --- |
| 推入区域 | 74,214 | 46,612 | 21,758 | 200 | contact_edge_3 |
| 推离区域 | 74,329 | 46,749 | 21,783 | 200 | contact_edge_1 |

请求字节约减少 37%。字节数不是 token 数；token 数来自本次成功响应的 `usage`。服务端预算上限仍未知，两个固定输入被接受不能证明所有未来输入均在预算内。两次重放只产生两次合法模型选择，没有机器人动作或任务回合。

保存的请求、响应、来源哈希和非预测字段一致性记录：

- `runs/contact_input_compaction_20261008/push_to_region_summary_replay.json`
- `runs/contact_input_compaction_20261008/push_aside_summary_replay.json`

## 完整执行验证

随后用原统一入口、两层 JEV、零任务恢复，各运行一次推动任务。两回生产源码一致，场景、控制与 verifier 配置保持原样。

| 任务 | 在线请求 | 执行动作 | 实际接触阶段 | Runtime / 隐藏验收 | 停止位置 |
| --- | ---: | ---: | --- | --- | --- |
| 推入区域 | 36，全部 HTTP 200 | 26，全部 completed | 通过 | 均未通过 | push_toward_goal |
| 推离区域 | 36，全部 HTTP 200 | 26，全部 completed | 通过 | 均未通过 | push_toward_goal |

两回均在 state 1283 / action epoch 25 观察到工具与物体接触、物体有支撑，Runtime 据此通过 `establish_contact`。随后 JEV 选择 `forward`，执行器完成该动作；state 1336 / epoch 26 中工具接触为 `clear`，物体仍有支撑。此后没有新物理动作，连续三次局部重规划请求耗尽既有额度而停止。实际任务恢复次数均为 0。

这是新的可定位失败，不能把本轮描述为推动任务修复成功。动作前的接触通过与动作后的接触丢失来自不同时刻，目前没有证据表明这两个 Runtime 判断互相矛盾。为何推动动作未保持接触，需要继续核查动作选择、连续接触表达和执行后的变化，不能仅凭最终失败指定根因。

独立交叉核对还确认：推动候选回到 `phase_contract`，当前 `selected_local_intent.contact_side=null`、`contact_geometry=null`、`action_evaluations=[]`。历史 `local_intent.last_termination` 仍保留先前选侧，阶段条件和约束也仍表达 `supported_push_contact`，所以不能说接触信息完全消失；缺少的是本动作的当前侧面几何与接触变化预测。原接触 verifier 验证实际主体接触、支撑、未持有和高度，不验证接触发生在候选指定侧面；本次接触通过也不能证明实际接触侧与选择一致。这些是接口及验收范围事实，尚不足以锁定接触丢失的唯一原因。

本轮共 74 次在线请求（两次冻结输入重放加两回各 36 次），52 个机器人动作、2 个新增任务回合，新增任务成功 0。两段失败视频均完整解码，各 57 帧。之前十二任务批次的结果不改写，也不将本轮局部复测合并成新的十二任务成功率。

完整请求计数、动作状态、阶段通过证据、首次推动前后接触、视频和文件哈希见 `runs/contact_input_compaction_20261008/validation.json`。视频及事件分别位于该目录的 `push_to_region_episode`、`push_aside_episode`。

## 回归与版本

新增四项有区分力的检查：四侧/动作比较与当前几何保持一致；两部件身份和负间隙；有限面不重叠及高度/切向缺口；不可观测与距离计算失败保留未知值。完整回归 **201 项通过**。

修改前源码复用已有版本快照；最终版本只保存一份源码归档，不按任务复制源码。版本文件 `runs/contact_input_compaction_20261008/version.json` 记录生产与测试指纹、前后源码归档及文件哈希。生产仅输入投影一处变化，测试仅 `tests/test_push_contact_round2.py` 一处变化。

本轮优先完成输入精简。客户端非 200 响应的错误正文记录仍是后续独立修复项；连续推动接触与动作选择需要先做证据分析，再确认具体修改。
