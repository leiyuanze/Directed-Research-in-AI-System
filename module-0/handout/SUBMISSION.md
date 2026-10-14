# 提交说明 / How to submit (Gradescope)

作业的十个步骤见 `handin_requirements.pdf`。评分依据为提交的 git 历史。

## 提交内容

完成十个步骤后，在仓库根目录执行以下命令，并将生成的两个文件放入一个 ZIP 提交至
Gradescope。该窗口只接受一个提交包，例如 `module0_full_submission.zip`：

```bash
git bundle create submission.bundle --all
git reflog --date=iso > git_reflog.txt
zip module0_full_submission.zip submission.bundle git_reflog.txt
```

| 文件 | 内容 |
| --- | --- |
| `submission.bundle` | 仓库的完整历史，含全部 commit、分支与 remote 引用。必须使用 `--all`，否则 `for_fun` 分支不在包内，第 5 步与第 8 步无法评分。 |
| `git_reflog.txt` | 本地 reflog。第 9 步与第 10 步只改变 HEAD 指向，不产生 commit，仅可通过 reflog 核实。 |

不要只提交 `submission.bundle`，也不要在 ZIP 中放入数据集或模型权重。ZIP 中多一层以
学号命名的目录不会影响评分器。

reflog 仅存于本地。若更换过设备或重新 clone 过仓库，reflog 将不完整，请在提交时注明，由助教人工核验。

## 评分方式

评分器 clone 提交的 bundle，按十个步骤逐条检查，每步 10 分：

| 步骤 | 检查方式 |
| --- | --- |
| 1. clone 仓库 | bundle 能被 clone，且有真实的提交历史 |
| 2. 新建 readme.md | 历史里出现过 readme.md |
| 3. 写入 hello world 并 commit | 某个 commit 的 readme.md 里有 `hello world` |
| 4. push 到远端 | bundle 里有 `refs/remotes/origin/*` |
| 5. 新建 for_fun 分支 | 存在名为 `for_fun` 的分支 |
| 6. main 上改成 aloha | main 上某个 commit 的 readme.md 里有 `aloha` |
| 7. 提交推理代码，commit message 里写准确率 | 代码里有 ResNet / MNIST / resize；某条 commit message 里有准确率数字 |
| 8. 把 for_fun 合并进 main | `for_fun` 的 tip 能从 main 到达，且合并后 readme 仍是 `aloha` |
| 9. 回到第 3 步的 commit | reflog 里有 `checkout: moving from main to <hash>` |
| 10. 切回最新，在 readme 里总结所学 | main 最新的 readme.md 内容 |

## 说明

- `readme.md` 的文件名不区分大小写。
- 默认分支名为 `main` 或 `master` 均可。
- 第 8 步采用 merge commit 或 fast-forward 均可。
- 第 7 步的准确率必须写在 commit message 中。
- 不得将模型权重或数据集提交进仓库。

本作业可多次提交，以最后一次提交为准。
