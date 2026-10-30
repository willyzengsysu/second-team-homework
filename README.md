# 高贝塔股票是否获得更高收益？

投资学第二次团队作业。2015-2019 年估计历史贝塔、固定分五组，2020-2024 年比较个股与组合。2026-10-28 修订数据口径、审计、检验解释和报告生成。

## 先看这些

当前为**组内审阅版**，尚未补齐成员信息、人工复核和全部数据证据。

| 想了解什么 | 入口 |
|---|---|
| 当前报告 | [PDF（9 页）](output/pdf/CAPM报告_修订版.pdf) · [网页可读正文](report/报告.md) |
| 老师的要求 | [作业任务书](第二次团队作业：高贝塔股票是否获得更高收益.md) |
| 上轮改了什么 | [项目修订说明](docs/修订说明.md) · [报告审查与润色意见](docs/报告审查与润色意见.md) |
| 我该复核什么 | [组内复核记录与待办](docs/组内复核记录.md) |
| 数字和方法从哪里来 | [自动结果表](output/tables/report_tables.md) · [方法说明](docs/方法说明.md) · [数据字典](docs/数据字典.md) |
| 如何上传与反馈 | [GitHub 上传及协作说明](docs/GitHub上传说明.md) |

当前主结果：最高减最低历史贝塔组月均收益差 **-0.202%**，HAC 95% 区间 **[-1.136%, 0.732%]**；H-L alpha 检验 p=0.581。贝塔和收益均未保持严格递增，不能据此验证或直接否定 CAPM。原始数据口径和退出现金流限制见正文。

## 给组员的使用方式

**只看报告不需要安装 Python。** 点击上述正文，或下载 PDF 即可。旧稿放在 report/versions/，供回溯，不作为当前结论。

### 1. 检查已提交结果（无需抓行情）

使用 Python 3.12，在仓库根目录执行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts/verify_outputs.py --offline
```

`--offline` 核对提交的面板、回归、组合收益、报告及哈希，不访问未入库的原始缓存；依赖包安装仍需要网络。requirements-lock.txt 保存原运行环境的完整版本，需使用原环境版本时可改为安装该文件。

### 2. 从已提交面板复现分析

```powershell
.venv\Scripts\python.exe scripts/run_all.py --from s4
```

这会重建贝塔、组合、敏感性分析、图表与 Markdown 报告，覆盖相应派生产物；PDF 须另行导出。无需先抓行情。中文图表字体在 Windows 上使用系统字体，其他系统可能需要安装中文字体。

### 3. 从来源重新抓数和清洗

```powershell
.venv\Scripts\python.exe scripts/run_all.py
.venv\Scripts\python.exe scripts/verify_outputs.py
```

S1 使用已保存的名单和历史行业快照；S2 将 baostock 行情缓存在 data/raw/，中断后可续抓。来源接口可能限流或变化，重抓不保证与旧快照逐字节相同。基准缓存完整时会复用，需要主动更新才运行 `python -m src.s2_fetch --refresh-benchmarks`。

原始缓存不入库，因此**刚克隆后不要直接从 S3 开始**。完成本机 S2 抓取后，才可运行 `python scripts/run_all.py --from s3`。完整流程是 S1 股票池 → S2 行情 → S3 清洗 → S4 贝塔 → S5 组合 → S5r 敏感性 → S6 图表 → S7 表格、数字锁定与正文。

## 目录导航

```text
config/                 样本、收益、回归与输出配置
src/                    S1-S7 分析及报告生成模板
scripts/                一键运行、结果核验、PDF 导出
tests/                  计算及数据处理规则测试
data/snapshots/         固定名单、历史行业与事件表
data/interim/           候选、市场/利率、来源及核查清单
data/processed/         面板、最终样本与缺失/退出审计
output/tables/          回归、组合、敏感性与数字锁定
output/figures/         图表
output/pdf/             当前报告 PDF
report/                 当前正文、写作上下文与旧稿
docs/                   方法、数据字典、审查和组员复核
ai_log/                 AI 使用摘要；完整对话待组员归档
```

本地 .venv/、data/raw/、logs/ 和 tmp/ 由 Git 忽略，不上传。证据文件通过 .gitattributes 保持原始字节，避免不同系统的换行转换破坏哈希。

## 数据与关键规则

| 项目 | 当前口径 |
|---|---|
| 股票池 | 保存的交易所上市/退出日期重建 2019-12-31 池；显式过滤 A 股证券代码，排除 B 股 |
| 行业 | baostock 请求 date=2019-12-31，保留 updateDate，拒绝未来/无日期分类 |
| 个股价格 | 全部候选统一使用 baostock 后复权月线 adjustflag=1；退出末月用同源日线补齐，保留成交量 |
| 市场 | 中证全指全收益 H00985，统一市场基准 |
| 无风险代理 | 同月末 1 年期国债到期收益率，按复利折月；不是实际 1 个月债券总收益 |
| 未知缺月 | 不填零；不跨未知缺月把多月涨跌当作单月收益 |
| 来源零成交量月 | 不当作有效交易报价，也不自动标为已核实整月停牌 |
| 停牌核实 | 只有附来源的事件才标记 suspended；当前事件表为空 |
| 退出 | 正式退出月以后不使用行情；退出前缺口、终止清偿/换股不足单列审计 |
| 组合 | 历史贝塔实际数值排序、固定五组；可得成员均值是统计口径，不等同于已验证可交易业绩 |

股票池不按未来存续筛选，但名单来源覆盖、后复权现金流算法和退出经济结果仍有限制。不得把“保留退市股票”等同于“完整计入所有退市损失”或“完全消除选择偏差”。

腾讯旧缓存仍保留在 data/raw/prices/，本次探测新增缓存可能位于同目录。腾讯访问受限后已停止使用其抓取，主分析只读取 data/raw/prices_baostock/，不会混用主来源。跨来源核对仅作为诊断，不改变样本。

## 结果与报告

- report/报告.md：S7 从结果表自动生成的正文，含九个逻辑分页；直接编辑会被重跑覆盖，正文模板位于 src/report_content.py。
- output/tables/：完整回归、固定分组、月度成员数、收益差、H-L alpha 检验、CAPM 基准偏差、五个敏感性情景。
- data/processed/missing_audit.csv、exit_audit.csv：逐月缺失与逐股退出核查。
- data/snapshots/：历史来源快照与有来源的事件覆盖表。
- output/tables/numbers_lock.json：结果文件完整 SHA256、正文哈希、代码哈希及运行版本；这是程序溯源，不是人工签认。

导出 PDF（Windows 默认使用系统宋体/黑体；其他系统见脚本字体说明）：

```powershell
pip install -r requirements-report.txt
python scripts/build_report_pdf.py
```

输出 output/pdf/CAPM报告_修订版.pdf。修改或重跑后须重新导出并检查页数、表格和图。PDF 导出单独运行，不属于需要联网的分析流水线。

## 解读边界

H-L 收益差为零与 H-L alpha 为零是不同检验。年化收益差报告 12 倍月均差，不称为实际复合多空收益。经验 SML 的残差标准差与相对 CAPM 指定线的 MAE/RMSE 分开报告；五点拟合和 HC1 标准误只作描述。

敏感性对照均保持主样本和组别：严格缺失、内部末价延续、延续至退出且无额外损失、退出全损压力情景、利率简单折月。情景假设不视为已核实数据，全损情景不是实际损失或统计置信界。

## 提交前待组员完成

姓名/学号、真实分工、独立人工复算、原始 AI 对话导出、退出公告和清偿/换股核实。报告保留这些待办，不宣称人工复核或完整对话归档已经完成。
