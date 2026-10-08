# 给定任务计划下的 JEV 仿真闭环复测（2026-10-03）

十二项均已结束，双重验收通过 **2/12**：单方块抓取、抓取后放置；其余十项失败。本轮两项成功均为直接成功，没有恢复后成功。**当前闭环尚不能认为可靠。**

运行时间：2026-10-03 00:57:32 至 2026-10-03 01:19:56（北京时间），约22.4分钟。策略为 `jev-1.13.0`，使用最新十二场景及3×3×3厘米方块。每任务一个给定 TaskPlan、一个回合、一段完整视频，最多五次实际重启；共计27次实际重启，10次恢复请求被拒。

## 逐任务结果

| 任务 | Runtime | 隐藏物理验收 | 实际重试 | 决策计数 | 结果与证据 |
| --- | --- | --- | ---: | ---: | --- |
| 抓取后放置 | 通过 | 通过 | 0/5 | 79 | 直接通过；[视频](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/pick_and_place/episode_001/success_episode.mp4) · [结果](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/pick_and_place/episode_001/result.json) · [恢复统计](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/pick_and_place/episode_001/statistics.json) |
| 单方块抓取 | 通过 | 通过 | 0/5 | 30 | 直接通过；[视频](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/pick/episode_001/success_episode.mp4) · [结果](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/pick/episode_001/result.json) · [恢复统计](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/pick/episode_001/statistics.json) |
| 分类 | 通过 | 失败 | 1/5 | 156 | 第一块后续偏移，最终目标未重新核对；[视频](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/sort/episode_001/failed_episode.mp4) · [结果](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/sort/episode_001/result.json) · [恢复统计](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/sort/episode_001/statistics.json) |
| 清理工作区 | 通过 | 失败 | 0/5 | 139 | 第一块后续偏移，最终目标未重新核对；[视频](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/clear/episode_001/failed_episode.mp4) · [结果](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/clear/episode_001/result.json) · [恢复统计](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/clear/episode_001/statistics.json) |
| 相对位置放置 | 失败 | 失败 | 1/5 | 153 | 搬运中抓取证据丢失，5次恢复请求被拒；[视频](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/relative/episode_001/failed_episode.mp4) · [结果](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/relative/episode_001/result.json) · [恢复统计](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/relative/episode_001/statistics.json) |
| 推入区域 | 失败 | 失败 | 5/5 | 46 | 接触前对齐方向不匹配，动作超时；[视频](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/push_to_region/episode_001/failed_episode.mp4) · [结果](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/push_to_region/episode_001/result.json) · [恢复统计](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/push_to_region/episode_001/statistics.json) |
| 推离区域 | 失败 | 失败 | 5/5 | 758 | 接触前对齐上下往返，预算和动作超时；[视频](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/push_aside/episode_001/failed_episode.mp4) · [结果](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/push_aside/episode_001/result.json) · [恢复统计](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/push_aside/episode_001/statistics.json) |
| 排列 | 失败 | 失败 | 5/5 | 108 | 第二块低位对齐时发生单指接触并移位，动作超时；[视频](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/arrange/episode_001/failed_episode.mp4) · [结果](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/arrange/episode_001/result.json) · [恢复统计](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/arrange/episode_001/statistics.json) |
| 堆叠 | 通过 | 失败 | 0/5 | 87 | 视觉与真实支撑重叠率分别为51.16%和48.03%；[视频](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/stack/episode_001/failed_episode.mp4) · [结果](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/stack/episode_001/result.json) · [恢复统计](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/stack/episode_001/statistics.json) |
| 入碗 | 失败 | 失败 | 0/5 | 149 | 搬运中为候选夹持，5次恢复请求被拒；[视频](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/bowl/episode_001/failed_episode.mp4) · [结果](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/bowl/episode_001/result.json) · [恢复统计](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/bowl/episode_001/statistics.json) |
| 按按钮 | 失败 | 失败 | 5/5 | 600 | 按压点误差未构造，持续保持；[视频](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/press/episode_001/failed_episode.mp4) · [结果](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/press/episode_001/result.json) · [恢复统计](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/press/episode_001/statistics.json) |
| 先放置后推动 | 失败 | 失败 | 5/5 | 107 | 红块放置通过，蓝块推动阶段超时；[视频](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/ordered/episode_001/failed_episode.mp4) · [结果](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/ordered/episode_001/result.json) · [恢复统计](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/ordered/episode_001/statistics.json) |

没有未处理的运行错误和 JEV 接口错误。Runtime 与隐藏物理验收有3项分歧：分类、清理工作区、堆叠。这三项均归为失败。

## 已核实的失败位置

### 分类、清理：能力完成后没有重新检查所有最终目标

分类红块放置完成后，在后续蓝块执行期间视觉中心偏移了34.0毫米；清理任务第一块偏移了22.7毫米。两项的最终公开 `in_region` 都已经为 false，隐藏物理验收也为 false，但 Runtime 仍判定完成。两类场景继承共享的终检方法，该方法直接接受能力程序完成，没有重查所有绑定主体的终局区域条件。事件证明了后续位移；本轮没有把具体碰撞部位认定为唯一原因。

对应代码：[共享终检](/Users/mikeho/Desktop/projects/JEV4Mujoco/src/jev4mujoco/runtime/loop.py:545)、[分类场景](/Users/mikeho/Desktop/projects/JEV4Mujoco/src/jev4mujoco/experiments/scenarios.py:150)、[清理场景](/Users/mikeho/Desktop/projects/JEV4Mujoco/src/jev4mujoco/experiments/scenarios.py:192)；数值证据：[最终目标复核](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/runtime_findings.json)。

### 推动、按压：阶段目标误差与验证目标不一致

推入区域的 Runtime 在 `align_precontact` 检查 TCP 到接触前位置的水平误差；发送给 JEV 的向量却是目标区域中心减物体中心。物体未动时，该向量保持不变，夹爪越过物体后仍向同一方向移动。修正版共有45次向左动作和6次超时，实际重启5次。推离区域有382次向下、375次向上，最终仍未通过接触前对齐。顺序任务完成了红块放置，随后在同一推动阶段失败。

按压点 `[0.42,-0.12,0.055]` 在公开目标字段中存在，但 `_interaction` 对点目标没有计算分支，把没有区域边界的点判为不可观测，并输出零向量。JEV 在600次决策中主要选择保持，Runtime 的按压点对齐未通过，最终没有接触按钮。

对应代码：[交互误差构造](/Users/mikeho/Desktop/projects/JEV4Mujoco/src/jev4mujoco/runtime/state_builder.py:444)、[JEV 阶段上下文](/Users/mikeho/Desktop/projects/JEV4Mujoco/src/jev4mujoco/policies/jev_context.py:140)；推入数值证据：[阶段误差记录](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/push_context_finding.json)。上述数值和代码分支已核实；把阶段误差不一致认定为主要执行原因，有动作选择和阶段停滞记录支持。

### 候选夹持：当前恢复入口无法继续

相对放置和入碗结束时，两个手指均有主体接触，但视觉跟随/抓取探测失败，状态为 `candidate_held`，没有已确认持有物体，也没有支撑面。当前恢复只允许已确认夹持或可见且有支撑的主体重启；后续各五次重抓请求都被拒。相对放置在这之前有一次有效的搬运重启，入碗没有有效重启。这里只能确认夹持证据失效，不能把它直接等同于物体已经掉落。

对应代码：[恢复前置条件](/Users/mikeho/Desktop/projects/JEV4Mujoco/src/jev4mujoco/runtime/capability_runtime.py:108)；证据为两任务的最终 `physical` 与 `statistics.json` 恢复事件。

### 排列：第二物体低位对齐时被推动

排列第一块已放置，第二块在抓取 XY 对齐阶段出现单指接触，真实蓝块从初始 `(0.40,-0.08)` 移至约 `(0.5174,-0.1638)`，移动约144毫米。夹爪末态高度约25毫米；重新进入同一抓取链后仍处在低位对齐阶段，六次动作超时导致五次重启额度耗尽。事件和末帧支持低位对齐与物体位移同时发生；撤离高度/接近动作是需要进一步核对的能力边界。

### 堆叠：视觉边界与真实几何跨过了同一验收门槛

视觉支撑重叠率为51.16%，真实几何为48.03%，配置门槛为50%。上层块确实接触底块且稳定，但真实重叠比例不足，因此双重验收失败。此项与分类/清理的漏终检不同：Runtime 重新检查了关系，但关系本身受视觉几何误差影响。

数值证据：[支撑重叠比较](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/stack_overlap_finding.json)。

## 本轮恢复机制改动与核验

恢复保留当前物理现场和其他主体已完成的步骤；抓取链、推动、按压共享每任务五次实际重启额度。保持已确认夹持时，只重启搬运/放置并保留抓取证据。每次有效重启后重新计算单次决策预算。动作拒绝、失败和超时均进入恢复；JEV 主动停止、接口错误或五次无效恢复请求会提前结束任务。仅观察与被拒绝的请求不计为实际重试。

首轮诊断发现动作超时入口遗漏，已修正并重新启动本报告的完整十二项批次。第一次部分批次保存为诊断证据，不进入本报告的12项统计。

诊断记录：[诊断说明](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T164155859509Z/diagnostic_note.md)。源码修改前备份：`/Users/mikeho/Desktop/projects/jev-retry-before-20261003-jmoma5uz`。

88项检查全部通过；新增检查覆盖五次重启后停止、保留物理状态、保留前一主体完成步骤、更新决策预算，以及原子动作超时立即恢复。批次辅助验证也确认失败后继续全部十二任务，推动/按压重启不会误计为重抓。

十二段视频均完成帧读取与末帧提取。保存的 TaskPlan 与重新按同一场景构建的初始计划逐项相等；恢复后的能力对象、目标、顺序和目标谓词保持不变。全部结果的实际重启计数与事件一致，且不超过5。所有结果的总体判定均等于 Runtime 与隐藏验收的合取。

证据：[88项检查](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/regression_tests.log)、[计划与视频核验](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/artifact_validation.json)、[版本清单](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/manifest.json)、[逐项统计](/Users/mikeho/Desktop/projects/JEV4Mujoco/runs/batches/jev_20261002T165732265439Z/summary.md)。源码/配置/场景总指纹为 `6cf1ffe2a2ba9f48050af838208cb16d2240cc9945b54661546adecd9b477696`，完整批次前后保持一致；启动版本已另存 `version_snapshot/`。

## 结果边界

本轮验证的是给定计划下的 JEV 快脑、通用能力、状态估计和 Runtime 闭环；未加入 LLM 慢脑。一次固定初始场景的结果不能证明重复可靠性。成功视频带2秒静置尾段，数值验收发生在闭环结束时；视频核验包含帧读取和选定末帧检查，不声称逐帧人工审阅全部视频。

本机渲染器持续报告深度精度受限；它是当前测量条件之一，未证明是所有失败的原因。隐藏物理真值只在回合结束后用于验收，不用于恢复选择。因此 Runtime 已误判成功的三项没有触发后续重试，这也是本轮暴露出的判断缺口。

下一步应优先核对阶段误差表达、全部主体的终局复核、低位跨对象接近、候选夹持恢复及视觉验收余量，再进行同条件复测；这些问题修正后才能判断加入慢脑带来的改进。
