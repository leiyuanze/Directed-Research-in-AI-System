aloha

这次作业让我完整走了一遍 Git、GitHub 和 Hugging Face 的基础流程。

Git 部分，我学会了 clone 仓库，使用 add、commit 和 push 管理提交，理解了
commit 历史的作用；还练习了 switch 创建和切换分支、checkout 回到旧 commit、
reflog 查看 HEAD 的变化，以及 merge 合并分支。最重要的是，我明白了分支只是
指向 commit 的可移动引用，只有保留完整的提交历史，Git 才能正确追踪每一步。

GitHub 部分，我了解了本地仓库和远程仓库的区别，理解了 origin 的含义，并把
main 分支推送到远程仓库。通过 push 和 remote-tracking ref，本地提交可以同步到
GitHub；同时分支 for_fun 也说明了 GitHub 和 Git 如何配合进行分支管理与协作。

Hugging Face 部分，我使用 transformers 加载了预训练的 ResNet-50，结合
AutoImageProcessor 对 MNIST 图像进行处理并 resize 到模型输入尺寸。原来模型
输出的是 1000 个 ImageNet 类别，不能直接和 MNIST 的 0~9 标签比较；所以我把
ResNet-50 当作冻结的特征提取器，用 0~9 的数字特征中心做原型分类，再通过余弦
相似度预测数字，最终得到 82.78% 的十类准确率。这个过程也让我理解了推理模式和
显存管理的重要性。

