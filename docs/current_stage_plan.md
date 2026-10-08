# 当前阶段计划：统一任务契约与通用能力闭环

## 阶段目标

在理想的固定 RGB-D、平行夹爪、刚性桌面物体场景中，用同一套契约覆盖既定的十二类单任务，并在 MuJoCo 中逐类完成可执行闭环验证。挑战场景和现实部署在基础闭环之后推进。

本阶段不采用“一种任务一个动作 JSON”或固定动作序列。每次任务由指令和首帧视觉动态实例化，JEV 根据实时状态逐步选择原子动作，Runtime 独立验证能力阶段与最终完成条件。

## 固定架构边界

2026-10-08 共同修复后的完整复测：同模式、同场景及阈值、零任务恢复，十二类各一次，抓取与堆叠双验收成功（2/12）；七项执行未完成、分类网络中止、清理模型返回不一致、排列视觉退化异常。分类单独补跑请求正常，仍在搬运持有估计降级后失败，未替换主统计。本轮没有HTTP 400，也没有新增生产修改。959份实际motor输入的条件/约束/历史来源核对无投影不一致；剩余失败不能统一归为Runtime矛盾或信息丢失。下一轮优先分析搬运目标进展与持有估计，具体修改待证据和确认。见 [完整复测报告](common_information_retest_20261008.md)。

2026-10-08 共同转换修复第一轮：用户确认后，由代码 agent 实施、主 agent 独立复核。保留抓取探测与释放机制说明，补齐原观测身份/时间/有效性并恢复动作层一帧适用历史；局部权限与 Runtime 验收边界保持原实现。213 项回归通过，1,021 份保存动作输入离线对照无差异错误，四次在线冻结请求均 HTTP 200；堆叠断点选择打开夹爪，相对放置当前几何未知仍重规划。本轮执行动作及新增任务回合均为零，没有新成功率。细节及待处理边界见 [共同信息转换修复](common_information_repair_round1_20261008.md)。

2026-10-08 五能力审查：按用户对跨类别信息偏向的质疑，暂停推动专用修改，完成20个注册阶段的投影核对及两组保存事件对照。当前条件和约束在动作投影中保留，但阶段机制说明被统一规则覆盖，Runtime前后测量中的前值被统一裁掉；缺少连续动作效果实现与实际信息丢失应分别处理。建议先修共同转换层的机制语义保留及带完整来源的前后测量，具体实施待确认；本次没有生产改动或新回合。证据与范围见 [五能力信息链路审查](cross_capability_information_audit_20261008.md)。

2026-10-08 输入精简第一轮：按用户确认，保留四侧候选、当前完整几何和阶段证据，将逐动作完整预测几何改为逐部件摘要，两份原失败输入约减少37%，均返回 HTTP 200。201 项回归通过；两类推动各复测一次，全程请求成功且建立实际接触，但第一次推动后接触丢失，局部重规划停止，均未通过最终验收。本轮只调整模型输入投影，下一处连续推动问题需继续分析；客户端错误正文记录待后续处理。细节见 [输入精简与执行验证](contact_input_compaction_round1_20261008.md)。

2026-10-08 当前实施状态：已按架构评审完成第一轮共享证据契约，新增阶段持续约束的作用域、五类状态与证据时间，并恢复点意图动作输入的完整阶段条件。Runtime 既有阶段判定、动作审查与执行控制未改变；197 项回归通过，三份保存状态重放保持原判定。独立高层 LLM 尚未接入，本轮没有在线 JEV 请求或新增任务成功率。细节见 [第一轮实施报告](phase_condition_contract_round1_20261008.md)；下一轮的受限高层入口设计尚待确认。

2026-10-08 优先级更新与复测：用户明确先修复给定 TaskPlan 下的十二类执行闭环，高层 LLM 接入暂缓。当前冻结版本以两层 JEV、零任务恢复逐类运行一次，结果为 1/12 双重验收成功（单方块抓取），八项执行未完成、两项 HTTP 400 请求中止、一项视觉几何运行异常。按压执行超时复现；搬运往返与持有降级、释放阶段重选、推动请求及视觉退化路径需要分别核查。本次模式和恢复额度与旧 2/12 批次不同，不作孤立改动的成功率对比。完整记录见 [首次执行复测](first_execution_retest_20261008.md)。

1. `VisualSceneV1` 是一份统一视觉 JSON schema。每次 RGB-D 观测产生一个遵守同一结构的新实例；所有任务共用，不按任务拆分视觉格式。
2. `TaskPlanV1` 是一份统一任务 JSON schema。慢脑根据指令和首帧 `VisualSceneV1` 生成每个 episode 的任务实例；它描述实体绑定、目标关系和完成契约，不描述推动方向、距离、动作次数或轨迹。
3. `CapabilityProgramV1` 是一份统一能力程序 schema。编译器把 TaskPlan 展开为带当前实体和目标参数的能力实例。
4. 能力模板是通用积木，不是具体任务脚本。当前核心集合为：
   - `pick`
   - `transport`
   - `place`
   - `surface_push`
   - `press`
5. JEV 只在 Runtime 当前激活的能力和阶段内选择下一次原子动作。Runtime 保留完整 `StateSnapshotV2`；JEV 读取由它裁剪出的 `JEVDecisionContextV1`，其中只包含当前 capability、相关主体/目标、交互误差、接触与夹爪状态、最近动作以及固定步长手柄。JEV 不直接推进阶段、不宣布任务成功，也不自由改写能力拓扑。
6. Runtime 拥有能力激活、阶段转换、具名 verifier 配置、完成判定、超时和阻塞权限。
7. MuJoCo 隐藏真值只用于仿真最终验收，不进入 JEV、TaskPlan 或 runtime verifier 的公开状态。
8. 距离、容差、连续观测数和稳定时间来自具名 deployment/runtime 配置，不由慢脑临时生成。

2026-10-03 核对发现，第 7 项的完整信息边界尚未实现：隐藏物体位姿和最终验收不回流控制，但状态构造仍直接使用 MuJoCo 接触反馈参与抓持与支撑判断。当前已显式标注这一模拟接触来源；真实硬件只有 RGB-D 和机器人自身反馈，必须完成对应的状态估计替换后才能声称满足该边界。详见 [子任务接口修复](jev_subtask_interface_update_20261003.md)。

2026-10-03 用户确认后，第 5 项增加两层选择的试验模式：JEV 提议局部意图并在该意图内选择一个原子动作，Runtime 保存和校验意图、检查局部有效性与进展，阶段和最终完成权限仍属于 Runtime。允许显式请求局部重规划，不允许自由改写任务或能力拓扑。默认仍为单层，给定 TaskPlan 条件下先进行同版本对照；慢脑尚未接入。详见 [局部意图闭环](local_intent_closed_loop_20261003.md)。

当前代码边界：旧的单碗 Stage 1 Runtime、旧状态估计器、旧规则/JEV 策略和对应探针已从活动代码与测试集中移除。MuJoCo 执行器通过 `configs/executor_v1.yaml` 加载控制参数，并由当前任务的 RGB-D 配置绑定场景；不再依赖旧 Stage 1 配置和场景。场景复位与原子执行器回归使用当前方块场景。旧 `JEVStateV2` 投影已移除，JEV 策略统一使用 `JEVDecisionContextV1`。

当前代码阅读入口见 [README](../README.md)。2026-10-02 的结构整理按契约、视觉、规划、运行时、仿真、策略和实验七组职责组织；迁移表与核验记录见 [结构方案](code_structure_plan.md) 和 [整理报告](code_structure_report.md)。本轮整理不新增任务、异步视觉或在线 JEV 可靠性证据。

## 十二类任务到通用能力的映射

| TaskKind | 能力组合 |
| --- | --- |
| `pick_object` | `pick` |
| `pick_and_place` | `pick -> transport -> place(goal)` |
| `sort_objects` | 对初始绑定对象重复 `pick -> transport -> place(goal)` |
| `clear_region` | 对初始区域对象重复 `pick -> transport -> place(goal)` |
| `place_relative` | `pick -> transport -> place(relative_goal)` |
| `place_inside` | `pick -> transport -> place(inside_goal)` |
| `push_to_region` | `surface_push(IN_REGION == true)` |
| `push_aside` | 对初始绑定对象重复 `surface_push(CLEAR_OF_REGION == true)` |
| `arrange_objects` | 生成目标布局后，重复 `pick -> transport -> place(layout_goal)` |
| `stack_objects` | 重复 `pick -> transport -> place(ON_SURFACE == true)` |
| `press_button` | `press(triggered == true)` |
| `ordered_sequence` | 按 stage 顺序组合以上任务程序 |

## 通用目标表达

能力实例的目标至少包含：

- `predicate`
- `reference_ref`
- `desired_value`

例如：

- 推入区域：`IN_REGION(target_region) == true`
- 推开区域：`CLEAR_OF_REGION(protected_region) == true`
- 放入容器：`INSIDE(container_interior) == true`
- 堆叠：`ON_SURFACE(top_surface) == true`

方向、接触边、预接触位姿和动作步数不是 TaskPlan 或 capability goal 的字段，由 JEV 根据当前状态决定。

## 分层实施顺序

1. 统一契约和通用 capability goal 投影。
2. 将现有区域放置收敛为参数化 `place`，保持已验证 Pick-and-Place 回归通过。
3. 完成 `surface_push` 的公开状态、entry/phase/completion verifier、确定性规则基线和单方块物理闭环。
4. 用同一 `surface_push` 验证推入区域和推出区域，不新增任务专用动作实现。
5. 实现通用 `place` 的相对放置、容器放置、排列和堆叠目标。
6. 实现 `press` 能力和按钮状态验证。
7. 实现多对象展开、逐项验证和 ordered sequence 的组合执行。
8. 基础能力的确定性规则闭环验证已在此前阶段完成。按 2026-10-02 用户确认，Stage 2.3 移除活动规则策略，后续仅运行 JEV，十四个实验首批各一次。
9. 基础理想场景全覆盖后，再增加轻度遮挡、位置扰动和视觉噪声，最后进入现实单任务测试。

## 每项任务的验收证据

每个任务至少需要：

1. TaskPlan 与初始 VisualScene 绑定验证通过。
2. CapabilityProgram 只使用通用能力模板。
3. 所有动作满足 state/action epoch 单步绑定和 single-flight 执行。
4. Runtime verifier 通过具名 entry、phase 和 completion contract。
5. 独立 MuJoCo 隐藏真值通过，不参与策略决策。
6. 成功实例视频只在 runtime verifier 与隐藏真值同时通过后保存为 `success_episode.mp4`。
7. 全量回归测试保持通过。

## 当前证据状态

- 基础抓取、区域搬运、颜色分类、桌面清理、相对位置放置、表面推动、三物体成行、两方块堆叠、放入开口容器、两物体杂乱场景目标抓取、大按钮按压和顺序组合，均已在理想 MuJoCo 场景通过 Runtime verifier 与独立隐藏真值。
- `pick_object -> pick`：`runs/v1_block_pick/attempt_001/success_episode.mp4`。
- `pick -> transport -> place(IN_REGION == true)`：`runs/v1_block_rule/attempt_002/success_episode.mp4`。
- `sort_objects`：`runs/v1_two_block_sort/attempt_001/success_episode.mp4`。
- `clear_region`：`runs/v1_two_block_clear/attempt_001/success_episode.mp4`。
- `place_relative`：`runs/v1_block_relative_place/attempt_001/success_episode.mp4`。
- `surface_push(IN_REGION == true)`：`runs/v1_block_surface_push/attempt_001/success_episode.mp4`。
- `surface_push(CLEAR_OF_REGION == true)`：`runs/v1_block_surface_push_aside/attempt_002/success_episode.mp4`。完成条件要求 footprint 完全无重叠并满足具名安全间距，不以 `IN_REGION == false` 代替。
- `arrange_objects`：三物体测试为 `runs/v1_three_block_arrange/attempt_001/success_episode.mp4`；隐藏真值横轴跨度为 0.001754 m，最小两两净间距为 0.028238 m。此前两物体视频只证明重复放置和间距，不再作为完整“成行”证据。
- `stack_objects`：`runs/v1_two_block_stack/attempt_001/success_episode.mp4`。
- `place_inside`：`runs/v1_block_inside_open_box/attempt_001/success_episode.mp4`。
- 杂乱场景抓取复用 `pick_object`，通过两物体目标选择且未扰动干扰物：`runs/v1_two_block_clutter_pick/attempt_001/success_episode.mp4`。当前只覆盖轻度杂乱，不代表严重遮挡。
- `press_button`：`runs/v1_large_button_press/attempt_001/success_episode.mp4`；隐藏真值实际按压行程为 0.011333 m，超过 0.008 m 门槛。
- `ordered_sequence`：`runs/v1_ordered_place_then_push/attempt_002/success_episode.mp4`；TaskPlan 的两个 stage 被顺序编译成 `pick -> transport -> place -> surface_push`，未新增任务专用动作模板。
- 2026-10-02 清理后的全量自动化回归为 96/96 通过；清理前本地复核为 97/97，减少的一项是已移除的旧 `JEVStateV2` 投影专用测试。场景复位和原子执行器测试已迁移到当前 V1 方块场景。
- JEV 已通过统一入口接入新框架闭环。裁剪后的 `JEVDecisionContextV1` 在 `pick_object` 实测中完成 28 次原子动作决策，Runtime verifier 与独立隐藏真值同时通过；证据为 `runs/v1_block_pick_jev/attempt_006/success_episode.mp4`。该结果属于此前在线记录。Stage 2.3 已按用户要求删除活动规则策略，后续在线评估仅使用 JEV。
- 轻度位置扰动、视觉噪声和遮挡挑战：尚未开始；当前结果只代表理想场景覆盖。
- 现实：尚未验证，不得标记为现实可用。现实部署前仍需重新标定相机、工作区、接触/稳定阈值和安全限制，不能直接照搬仿真数值。

成功视频默认在双重验收通过后额外记录 2 秒无动作静止画面；该收尾只改善证据可读性，不参与任务完成判定。

## Stage 2.3 当前覆盖安排

统一入口已支持 `--tasks all --episodes 1`；每实验一次 JEV 回合，内部恢复计入同一视频。现有恢复范围不扩展，最终失败仍保存并统计。保留的 84 项回归通过，13 项规则整任务检查已归档。批量运行与逐视频统计详见 [批量评估方案](batch_evaluation_plan.md)。此前规则视频只作为历史实现证据，不进入本次 JEV 批次成功率。

2026-10-02 首批在线 JEV 已运行完十四项，覆盖十二类任务，双重验收成功 5/14；首次成功 4，经恢复成功 1。失败仍保留视频，完整证据见 [评估报告](batch_evaluation_report.md)。覆盖表示已运行，不表示全部通过。

## 2026-10-02 场景配置更新

按用户确认，本轮将活动实验收敛为十二类任务各一个 XML、一份同名固定 RGB-D 配置，共用 `scenes/common.xml`。`place_inside` 选用实际碗模型，`pick_object` 使用单方块；入盒和杂乱抓取保留在修改前备份及历史运行记录中，不再进入当前 `--tasks all`。

区域标记只承担显示与区域表达；任务实体保留碰撞。各 Runtime 直接构建对应任务计划，去除强制 `target_area` 依赖；隐藏验收尺寸来自实际碰撞几何，目标边界和支撑高度与本任务配置同步。较远的对象和目标已向内侧调整，以通过当前固定夹爪姿态的可达性检查。

此前十四项 JEV 批次及历史成功视频对应旧场景，不能作为新场景的在线成功证据。本轮场景检查、物理探针与回归结果见 [场景配置报告](scene_configuration_report.md)。

## 2026-10-03 方块尺寸统一

按用户确认，所有活动场景的方块统一为 3×3×3 厘米；初始支撑高度和 keyframe 同步更新。十二场景共有 18 个方块实例，84 项回归及 52 个选定位姿检查通过。此前 10 月 2 日场景核验对应混合尺寸版，新的尺寸证据见 [尺寸更新记录](block_size_update_20261003.md)。该尺寸核验完成时尚未运行完整在线 JEV 回合，后续结果见下方闭环复测。

## 2026-10-03 给定计划下的闭环复测

按用户确认，在给定 TaskPlan 的条件下测试 JEV 快脑、通用能力和 Runtime 能否完成仿真执行闭环；本轮不加入 LLM 慢脑。十二任务各运行一次，每任务最多五次实际重启（首次加重试最多六次），多物体任务共用额度。失败保留物理现场，恢复抓取链或当前推动/按压，保留其他对象已完成的步骤；仍持有物体时保留抓取证据重启搬运。实际重启后重新计算一次尝试的决策预算。

这次授权取代此前“现有恢复范围不扩展”的本轮限制。JEV 仍选择原子动作及恢复选项，Runtime 验证并执行逻辑重启。仅观察、被拒绝的重启请求不计入实际次数；连续五次没有实际重启的恢复观察/无效请求、JEV 主动停止、无法恢复及接口错误会提前结束并记录原因。隐藏真值仅用于结束后的独立验收，验收分歧不会回流到控制。每任务一段完整视频，分别记录直接成功、恢复后成功、失败和验收分歧；一次固定场景回合不能证明重复可靠性。

完整修正版批次已完成十二项，双重验收通过2项（抓取、抓取后放置），其余10项失败；累计实际重启27次，均未超过每任务5次。分类、清理、堆叠有3项验收分歧，没有恢复后成功。88项检查及全部计划/视频/事件一致性核验通过，源码、配置与场景指纹前后不变。当前闭环仍不可靠，阶段误差表达、全局终检、跨对象接近与候选夹持恢复等缺口详见 [闭环复测报告](closed_loop_retest_20261003.md)。

## 2026-10-04 第一轮正确性修复

按用户“一轮一轮来”的安排，第一轮完成分类/清理全局终检、接触与抓持估计一致性、阶段条件核对；143项检查通过。原批次2,766份公开状态中重评2,683份阶段状态，覆盖15/20个阶段，未发现条件列表与验收通过的分歧；分类、清理原最终状态会被新增终检拒绝。没有新增完整任务回合，原2/12成功率不变。

本轮停在正确性修复，不进入接触几何、避障、控制调参或恢复扩展。现有恢复还不能自动重做早先失效物体；后续接近/接触几何检查及执行器核查见 [第一轮报告](runtime_correctness_round1_20261004.md)。

## 2026-10-04 第二轮接触接口实施

按随后确认的方案，已补齐推动夹指标定几何、独立右指反馈、模型选定有限侧面及最近点距离/动作预测，并在明显轮廓缩小时声明 PARTIAL、停止沿未知几何推动。161 项检查通过；同一历史前缀后的五步诊断接近可建立实际接触。

两回有界两层 JEV 仍失败。新证据确认：预接触阶段只检查到整个外框的距离，尚未核对已选侧面；距离预测也未表达动作后的原接触高度约束。下一处阶段衔接及约束审查修改待用户确认，不先放宽阈值或改控制。原十二任务结果不变，完整记录见 [第二轮实施与验证](push_contact_round2_20261004.md)。
