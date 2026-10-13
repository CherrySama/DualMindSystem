# JEV4Mujoco

Panthera 桌面操作的 MuJoCo 仿真框架。当前链路为：预设指令与首帧视觉 → TaskPlan → 五类通用能力 → JEV 局部意图与动作 → Runtime 审查、执行与独立核验。

最新同版、两层 JEV、零任务恢复复测：**10/12 完整自主成功**；按压效果成功但执行超时，效果双确认合计 **11/12**。堆叠正式回合接口中止，同版独立补测执行正常但未形成堆叠。记录对应清理前冻结版本，不能视为十二类稳定全通过。见[复测报告](docs/full_twelve_retest_20261013.md)。

高层 LLM、真实视觉识别、统一 retry 与现实时间闭环尚未完成。在线输入限定为 RGB-D、关节位置/速度、估计力矩、控制指令、夹爪位置和校准参数；隐藏仿真物体/接触真值仅供独立裁判。当前实例掩码与类别/颜色语义仍使用理想仿真适配器。见[当前架构](docs/current_stage_plan.md)与[真实反馈契约](docs/real_robot_feedback_contract.md)。

## 运行

使用现有 `jev-mujoco` 环境，JEV 凭证通过 `.env` 或环境变量加载，格式见 [.env.example](.env.example)。

```bash
conda activate jev-mujoco
python scripts/run_v1.py --help
python scripts/run_v1.py --task pick --maximum-retries 0 --jev-stages 2
python scripts/run_v1.py --tasks all --episodes 1 --maximum-retries 0 --jev-stages 2
```

支持 `pick`、`pick_and_place`、`sort`、`clear`、`relative`、`push_to_region`、`push_aside`、`arrange`、`stack`、`bowl`、`press`、`ordered`。使用 `--tasks pick stack` 选择子集，或 `--task stack --episodes N` 重复单个实验。每类场景与 RGB-D 配置分别位于 `scenes/` 和 `configs/`；公共机器人配置在 `scenes/common.xml`。

示例显式关闭任务恢复并选择两层 JEV；程序默认仍是 `--maximum-retries 5`、`--jev-stages 1`。已有恢复入口保留现场并重启当前能力/抓放链，尚未形成统一恢复策略。客户端请求重试与任务恢复分别统计。

每回合保存事件、计划、结果、统计和完整视频；批次位于 `runs/batches/jev_*/`，包含冻结版本清单、逐回合统计与报告。只有 Runtime 和独立裁判同时通过时才使用成功视频名称，执行异常仍单列。`runs/` 被 Git 忽略，本机证据不会随代码上传。

## 离线回归

```bash
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

428 个用例按 15 个功能模块组织，另有一个共享几何夹具文件与两份数据夹具。覆盖契约、规划、感知、仿真执行、Runtime 阶段与恢复、JEV、抓取、搬运、放置、推动、按压及客户端。离线回归不发送在线模型请求，通过也不等于在线可靠性或真实机器人验证。

## 代码阅读入口

| 职责 | 入口 |
| --- | --- |
| 实验运行与装配 | [run_v1.py](scripts/run_v1.py)、[registry.py](src/jev4mujoco/experiments/registry.py)、[scenarios.py](src/jev4mujoco/experiments/scenarios.py) |
| 视觉、任务、状态契约 | [contracts](src/jev4mujoco/contracts) |
| 预设计划与能力编译 | [planning](src/jev4mujoco/planning) |
| JEV 决策与模型客户端 | [policies](src/jev4mujoco/policies) |
| 状态融合、阶段与动作验证 | [runtime](src/jev4mujoco/runtime) |
| RGB-D 重建及几何 | [perception](src/jev4mujoco/perception) |
| MuJoCo 后端与执行器 | [simulation](src/jev4mujoco/simulation) |
| 批次统计与独立裁判 | [experiments](src/jev4mujoco/experiments) |

`panthera_python/` 保留完整 Panthera Python SDK。对象资源位于被忽略的 `assets/objects/`；需要恢复资源时，先安装 `requirements-assets.txt`，再运行 `python assets/download_objects.py`。下载来源与解压规则见[资源脚本](assets/download_objects.py)，本轮整理不触发下载。

项目当前已关联现有 GitHub 仓库；本轮仅本地整理，没有提交或推送。文档收敛为六份，历史修复文档与旧测试原文件移出项目归档，最新完整批次及堆叠补测保留。见[文档索引](docs/README.md)、[清理记录](docs/cleanup_20261013.md)和[后续交接](docs/next_conversation_handoff_20261013.md)。
