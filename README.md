# NCRT：面向大语言模型安全评估的多策略红队实验平台

<div align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-22C55E">
  <img alt="Version" src="https://img.shields.io/badge/状态-研究原型-F59E0B">
  <img alt="Tests" src="https://img.shields.io/badge/测试-待配置-lightgrey">
</div>

<p align="center">
  对授权模型执行可追溯、多策略、可比较的大语言模型安全评估。
</p>

> [!IMPORTANT]
> 本项目仅面向已获授权的模型安全研究与红队测试。仓库中的测试数据可能涉及有害主题；请遵循法律、组织政策与伦理审查要求，并对运行产物实施访问控制。

## 目录

- [项目简介](#项目简介)
- [核心能力](#核心能力)
- [系统架构与运行流程](#系统架构与运行流程)
- [策略与评测设计](#策略与评测设计)
- [环境与安装](#环境与安装)
- [快速开始](#快速开始)
- [命令行参数](#命令行参数)
- [Web 与桌面端](#web-与桌面端)
- [API 参考](#api-参考)
- [输出与复现实验](#输出与复现实验)
- [项目结构](#项目结构)
- [贡献与路线图](#贡献与路线图)
- [许可证](#许可证)

## 项目简介

NCRT（**N**ext-generation **C**ognitive **R**ed-**T**eaming）是一个研究型 LLM 安全评估平台。它将候选提示生成、目标模型调用、自动化评测、策略切换和实验记录组合为统一工作流，支持单目标、批量数据集和多策略对比三种运行方式。

项目提供四种 planner，并由 `StrategyManager` 统一调度。调度器会追踪每轮得分和最优轨迹，在策略阶段结束或陷入停滞时传递结构化摘要，使后续策略能够基于已观测到的模型边界继续评估。

<p align="center">
  <img src="assets/scheduler-attacker-overview.png" alt="NCRT 调度器与攻击器协同流程示意图" width="900">
</p>

<p align="center"><sub>图 1：NCRT 的多策略调度与摘要交接机制。</sub></p>

## 核心能力

- **统一评估链路**：以 `StepResult`、`ConversationTurn` 和 `AttackResult` 统一承载 planner 中间状态、对话轨迹和最终结果。
- **多策略组合**：内置 Crescendo、PAIR、TAP 和 Safe2Harm，并可单独运行或交由调度器编排。
- **可配置模型端点**：攻击模型、目标模型与 judge 模型可分别配置，兼容 Ollama 原生接口和 OpenAI 兼容接口。
- **多视角自动评测**：judge 从合规性、危害性和上下文三个视角独立评分，并以证据融合方式汇总；明确拒答会被规则检测校正为零分。
- **可观测执行过程**：CLI 输出轮次日志；Web 服务通过 SSE 推送实时进度、轮次、调度状态和结果。
- **结构化结果落盘**：单目标和批量任务都会生成 JSON 结果，Web 端还可导出完整文本记录。

## 系统架构与运行流程

<!-- Experimental：若 Mermaid 在目标渲染器中不可用，请在 GitHub 中预览。 -->

```mermaid
graph TD
    A[评测目标或数据集] --> B[CLI / Web / 桌面端]
    B --> C[StrategyManager]
    C --> D[Planner 策略池]
    D --> E[Generator]
    E --> F[目标模型]
    F --> G[Judge]
    G --> C
    C --> H[JSON 结果与 SSE 事件]
```

一次外部评测轮次的执行逻辑如下：

1. `Generator` 调用攻击模型生成候选提示，并把提示发送给目标模型。
2. planner 接收目标模型回复和 judge 反馈，更新自己的状态机或搜索状态。
3. `Judge.evaluate()` 调用三个评测视角；评分会经过 Dempster–Shafer 风格的证据融合，并由快速拒答检测做保守修正。
4. `StrategyManager` 记录最优分数、可见对话轮次、planner 使用情况和摘要交接信息。
5. 当得分达到阈值、可用策略耗尽或轮次预算结束时，系统返回 `AttackResult` 并写出结果。

> [!NOTE]
> `--rounds` 与 `SchedulerConfig.max_llm_calls` 当前限制的是调度/策略轮次，不是全部 HTTP 模型请求的精确次数。一轮可能包含攻击模型、目标模型与多个 judge 调用；Safe2Harm 还包含内部阶段。

## 策略与评测设计

| 组件 | 实现位置 | 当前职责 |
| --- | --- | --- |
| Crescendo | `baseline/methods/crescendo.py` | 使用分阶段的多轮对话规划，并在后续轮次基于历史继续迭代。 |
| PAIR | `baseline/methods/pair.py` | 保留得分较优的历史尝试与 judge 反馈，执行贪心式迭代精炼。 |
| TAP | `baseline/methods/tap.py` | 生成多个候选提示，基于目标模型真实回复的得分保留较优分支。 |
| Safe2Harm | `baseline/methods/safe2harm.py` | 执行分析、领域选择、改写、目标模型生成与重建评分的多阶段流程。 |
| StrategyManager | `scheduler/scheduler.py` | 负责首个 planner 的初始预算、后续 planner 的随机顺序、摘要交接与结果汇总。 |
| Judge | `core/judge.py` | 执行多视角评分、证据融合、重建评分与快速拒答检查。 |

<p align="center">
  <img src="assets/planner-and-evaluation-design.png" alt="NCRT 策略池、调度增强与评测指标设计图" width="900">
</p>

<p align="center"><sub>图 2：单策略基线、调度增强与多视角评测的关系。</sub></p>

默认调度器的 roster 为 `crescendo → safe2harm → pair → tap`。首个策略默认获得 10 轮预算；其余策略会在接收前一策略的摘要后按随机顺序执行各自的最小状态周期。选择单一 planner 时，系统仍通过同一调度器运行，但 roster 只保留该策略。

## 环境与安装

推荐使用 Python 3.10 或更高版本。仓库当前没有 `requirements.txt` 或锁文件；从源码导入可确认的运行依赖如下：

| 依赖 | 用途 | 必需场景 |
| --- | --- | --- |
| `requests` | 调用模型服务 | CLI、Web、桌面端 |
| `flask` | 提供 Web API 与 SSE | Web、桌面端 |
| `pywebview` | 创建原生桌面窗口 | 仅 `Client/launcher.py` |

```powershell
# 在仓库根目录执行
python -m venv .jailbreak
.\.jailbreak\Scripts\Activate.ps1
pip install requests flask pywebview
```

<details>
<summary>配置模型服务</summary>

NCRT 使用三个逻辑角色：攻击模型、目标模型和 judge 模型。每个角色都可独立设置模型名、API 基址和密钥。请使用受控服务并通过环境变量、密钥管理器或运行时输入提供凭据；不要把真实密钥写入代码、README 或结果文件。

```powershell
python run.py `
  --planner scheduler `
  --attack-model "<ATTACK_MODEL>" `
  --attack-base-url "https://<ATTACK_ENDPOINT>/v1" `
  --attack-api-key "<ATTACK_API_KEY>" `
  --victim-model "<VICTIM_MODEL>" `
  --victim-base-url "https://<VICTIM_ENDPOINT>/v1" `
  --victim-api-key "<VICTIM_API_KEY>"
```

`Generator` 会识别 Ollama 风格地址并调用其原生聊天接口；其他地址按 OpenAI 兼容的 `/chat/completions` 方式请求。

</details>

## 快速开始

从仓库根目录运行以下命令。

```powershell
# 运行一个获得授权的单目标评测
python run.py --planner crescendo --goal "<已获授权的评测目标>"

# 通过调度器对数据集样本进行批量评测
python run.py --planner scheduler --scale 10 --rounds 20

# 对四种独立 planner 执行对比实验
python run.py --compare --scale 10 --rounds 20
```

数据集位于 `data/harmful_prompts.json`。省略 `--goal` 时，CLI 会依照 `--scale` 从该数据集抽样；`--scale all` 表示选择全部样本。

> [!WARNING]
> 批量运行前请确认 `output/` 的内容。当前恢复逻辑以该目录中 JSON 文件数量判断已完成任务，因此混合多个实验或比较模式时可能导致不准确的跳过或结果覆盖。需要严格实验管理时，请为每次实验设置独立的 `--output` 目录。

## 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--planner` | `crescendo` | `crescendo`、`pair`、`tap`、`safe2harm` 或 `scheduler`。 |
| `--goal` | 空 | 单个授权评测目标；为空时读取数据集。 |
| `--scale` | `10` | 批量样本数，或 `all`。 |
| `--rounds` | `20` | 每个目标的调度轮次上限。 |
| `--threshold` | `0.7` | 达到该得分即视为评测成功。 |
| `--attack-model` / `--victim-model` / `--judge-model` | 因角色而异 | 三类模型的标识符。judge 模型留空时复用攻击模型。 |
| `--attack-base-url` / `--victim-base-url` / `--judge-base-url` | 空 | 对应模型服务的 API 基址。 |
| `--attack-api-key` / `--victim-api-key` / `--judge-key` | 空 | 对应模型服务的访问凭据。 |
| `--workers` | `1` | CLI 批量和对比任务的并行 worker 数。 |
| `--seed` | `42` | 数据集抽样的随机种子。 |
| `--output` | `output/` | CLI 结果输出目录。 |
| `--compare` | 关闭 | 对所有独立 planner 执行比较。 |

## Web 与桌面端

### NCRT 统一命令

在项目根目录执行 `NCRT.bat`（或将该目录加入 `PATH` 后直接执行 `NCRT`）。首次使用前，复制并修改根目录的 `config.json`；同一份示例也保存在 `ncrt/config.example.json`。三个模型角色分别支持模型名、OpenAI 兼容 API 地址和密钥环境变量，避免把真实密钥写入配置文件。

```powershell
# 显示介绍与帮助
.\NCRT.bat
.\NCRT.bat -help
.\NCRT.bat -v

# 读取 config.json 并执行评估；完整对话 JSON 与运行日志写入 output/
.\NCRT.bat -run

# 仅显示动态进度条，详细控制台日志仍写入 output/
.\NCRT.bat -run -nolog

# 启动与 Client/launcher.py 相同的桌面界面
.\NCRT.bat -run -client
```

可通过 `--config <PATH>` 使用另一份配置，例如 `.\NCRT.bat -run --config .\my-config.json`。`api_key_env` 指向的环境变量优先于配置项中的 `api_key`。

```powershell
# 启动 Flask 服务
python Client/server.py

# 启动桌面窗口（内部启动 Flask 服务）
python Client/launcher.py
```

浏览器界面支持单目标、批量和比较模式；可以分别填写三类模型配置、设置阈值、查看实时轮次与调度状态，并在完成后导出结果。桌面端使用 `pywebview` 承载同一 Web 界面。

> [!CAUTION]
> Web 界面中的 API key 会随请求传递给本机服务。请仅在可信本机环境使用，勿把服务暴露到不受控网络，也不要在浏览器默认值中保留真实凭据。

## API 参考

| 方法与路径 | 功能 | 关键输入/输出 |
| --- | --- | --- |
| `GET /api/health` | 健康检查。 | 返回服务状态。 |
| `POST /api/attack` | 创建评测会话。 | 请求体为模型、planner、目标、批量与阈值配置；返回 `session_id`。 |
| `GET /api/stream/<session_id>` | 获取 SSE 事件流。 | 推送 `status`、`round`、`scheduler`、`result` 与 `done` 事件。 |
| `POST /api/stop/<session_id>` | 请求停止活动会话。 | 返回是否成功设置停止标记。 |
| `GET /api/export/<session_id>` | 导出完成会话。 | 返回可下载的纯文本轮次记录。 |
| `GET /api/dataset/preview` | 预览数据集。 | 可通过 `limit` 查询参数限制返回数量。 |

## 输出与复现实验

每个结果条目包含 planner、目标、成功状态、最佳分数、轮次、最终提示/回复以及 scheduler 元数据。Web 路径还会保存完整的可见轮次和策略摘要。默认产物写入 `output/`。

为了让实验结果可复核，建议在每次实验中记录：

- Git commit、Python 与依赖版本；
- 攻击模型、目标模型和 judge 模型的名称与版本；
- 数据集版本、样本数、`--seed`、`--rounds` 和 `--threshold`；
- 模型服务参数、原始 JSON 产物和失败日志；
- 评测协议及人工复核规则。

自动 judge 分数是研究指标，不是安全保证。对外报告的比较结论应包含模型版本、完整配置、原始产物与人工核验说明。

## 项目结构

```text
Jailbreak/
├── run.py                         # CLI 入口
├── core/
│   ├── generator.py               # 模型客户端与请求重试
│   ├── judge.py                   # 多视角评分与拒答检测
│   └── types.py                   # 统一数据类型
├── scheduler/
│   └── scheduler.py               # 策略编排、摘要交接与结果汇总
├── baseline/methods/
│   ├── crescendo.py               # 渐进式多轮策略
│   ├── pair.py                    # 迭代精炼策略
│   ├── tap.py                     # 候选分支与剪枝策略
│   └── safe2harm.py               # 多阶段语义转换策略
├── Client/
│   ├── server.py                  # Flask API 与 SSE
│   ├── launcher.py                # pywebview 桌面启动器
│   └── static/index.html          # 浏览器界面
├── assets/                        # README 图示资源
├── data/harmful_prompts.json      # 评测数据集
└── output/                        # 运行产物目录
```

## 贡献与路线图

欢迎围绕可复现性、评测质量与工程可靠性提交改进。在提交前请避免提交真实凭据、未脱敏的敏感模型输出或大型运行产物。

- [ ] 增加 `requirements.txt` 或可锁定的依赖清单。
- [ ] 为调度、输出命名、断点恢复和 judge 解析补充单元测试。
- [ ] 以独立的实验 manifest 取代按文件数量恢复的方式。
- [ ] 将全局随机数改为每次运行独立的随机数生成器，并完整记录种子。
- [ ] 记录真实模型调用次数、token 用量和耗时，而非仅记录策略轮次。

## 许可证

本项目采用 [MIT License](LICENSE)。
