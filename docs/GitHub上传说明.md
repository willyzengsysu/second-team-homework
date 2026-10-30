# 上传与组员协作

当前目录已是 Git 仓库。先在 GitHub 创建空仓库，取得实际仓库地址；推荐用 Git 或 GitHub Desktop 上传整个项目。

## 用 Git 上传

在项目根目录执行，先确认 git status 中的文件确实要分享：

```powershell
git status --short
git add .
git diff --cached --stat
git commit -m "整理团队审阅版：报告、复现说明与复核入口"
```

首次设置远程并推送（把占位地址换成自己创建的仓库地址，不要原样运行）：

```powershell
git remote add origin <你的仓库地址>
git push -u origin HEAD
```

如果已有 origin，请先 `git remote -v` 检查，沿用正确地址，无需再次 add。仓库可见范围和协作者权限由仓库所有者设置。

## 什么会入库

- 代码、配置、依赖清单、任务书与说明。
- 当前 Markdown/PDF 报告、图表、CSV 结果和审计表。
- 固定名单/历史行业快照、来源清单和哈希。
- AI 使用摘要、润色前版本及待填写的组内复核记录。

虚拟环境、运行日志、临时渲染图、原始行情缓存、环境密钥及编辑器本地设置由 .gitignore 排除，不需要删除本地文件。

## 组员如何开始

1. 直接在网页看 README 的“先看这些”，或下载 PDF。
2. 阅读已知限制和组内复核记录，认领实际要核对的项目。
3. 需要核算时安装依赖并运行 `python scripts/verify_outputs.py --offline`；该模式不依赖未入库的行情缓存。
4. 发现问题通过 Issues 或 Pull Request 反馈，避免多人同时覆盖报告生成模板。

提交实验结果时一并说明输入、口径和命令；修改数据后不要只手改报告中的数字。
