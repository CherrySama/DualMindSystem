# DualMindSystem

这是当前的 JEV4MuJoCo 研究仓库：预设任务计划生成 `TaskPlanV1`，JEV 根据当前阶段和新鲜观测选择一个原子动作，Runtime 负责阶段状态、条件与完成判定，MuJoCo 执行器负责实际运动。当前还没有接入负责高层语义规划的 LLM；这个分工仍处于架构设计阶段。

## 当前状态

2026-10-08 的十二类任务首次执行复测为 **1/12**。唯一通过的是单方块抓取，不能据此宣称抓取能力已经可靠。其余结果被区分为：

- 搬运对齐和持有证据在部分任务中反复失效；
- 两个推动任务在接触意图请求处收到 HTTP 400，服务端具体原因尚未由客户端正文确认；
- 按压仍复现执行器位置未收敛问题；
- 清理任务出现视觉几何运行异常。

因此，当前成功率不能简单归因于 JEV，也不能把 HTTP 400 或未充分执行的回合当成完整物理能力失败。最新证据见 [首次执行复测](docs/first_execution_retest_20261008.md)。接口边界和架构分工见 [System2/System1/Runtime 评审](docs/system2_system1_runtime_review_20261006.md)；输入冗余与接触表达修复见 [公共信息复测](docs/common_information_retest_20261008.md) 和 [接触输入压缩](docs/contact_input_compaction_round1_20261008.md)。

## 安装与运行

```bash
conda activate jev-mujoco
python -m pip install -r requirements-assets.txt
python assets/download_objects.py
python scripts/run_v1.py --help
python scripts/run_v1.py --task pick_and_place
python scripts/run_v1.py --tasks all --episodes 1
```

大型 RoboTwin 对象库不会提交到 GitHub。下载脚本从 `TianxingChen/RoboTwin2.0` 获取 `objects.zip`，解压到 `assets/objects/`，完成后删除压缩包并写入完成标记。`assets/mujoco/` 中的 OBJ 是当前 MuJoCo 场景直接引用的转换网格；两者格式和职责不同，不能直接互换。

每次运行的事件、请求、结果和视频保存在 `runs/`，该目录被 Git 忽略。完整任务映射见 [当前阶段计划](docs/current_stage_plan.md)，代码入口见 [scripts/run_v1.py](scripts/run_v1.py)。

## 当前边界

- Runtime 的验证与隐藏物理验收保持分离，隐藏真值不回流给 JEV；
- 当前接触、抓持和支撑证据仍包含 MuJoCo 接触读数，尚未证明为纯 RGB-D/机器人反馈的真机闭环；
- 重试、高层 LLM、自由任务规划和完整接触力估计暂不作为已实现能力；
- 所有新增成功率都必须来自完整任务回合，冻结输入重放和局部探针不计入任务成功率。
