## 关于 VSCode SSH 断开后任务是否继续

**直接结论：关闭 VSCode 后，训练进程会被杀死。必须用 `tmux`。**

### 为什么？

VSCode SSH 连接本质是一个 SSH session。关闭 VSCode 时，SSH session 结束，
session 内启动的所有进程（包括 Python 训练脚本）会收到 SIGHUP 信号并终止。

### 解决方案：tmux

`tmux` 是一个终端复用器，能让进程在**与 SSH session 完全独立的后台**运行。
即使关闭 VSCode、断网，tmux 中的进程仍然在服务器上继续运行。

```bash
# 创建一个名为 "lam-train" 的 tmux 会话
tmux new-session -s lam-train

# 在 tmux 中启动训练（启动后可以随时断开）
conda activate dreamdojo
cd /home/xuan/embodied-ai/code/lam
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes 3 --mixed_precision bf16 train_lam.py --config config/lam_agibot.yaml

# 【关键操作】脱离（detach）tmux 但保持进程运行：
# 按 Ctrl+B，然后按 D
# 此时可以安全关闭 VSCode

# 下次重新连接服务器后，重新进入 tmux 查看训练状态：
tmux attach -t lam-train

# 查看所有正在运行的 tmux 会话：
tmux ls
```

### tmux 常用快捷键

| 操作 | 快捷键 |
|------|--------|
| 脱离（保持后台运行）| `Ctrl+B` 然后 `D` |
| 重新进入 | `tmux attach -t <名字>` |
| 在 tmux 内翻页查看日志 | `Ctrl+B` 然后 `[`，用方向键翻页，按 `Q` 退出 |
| 新建窗口 | `Ctrl+B` 然后 `C` |
| 切换窗口 | `Ctrl+B` 然后数字 |
| 强制关闭会话 | `tmux kill-session -t <名字>` |

**推荐工作流**：
1. SSH 连接服务器
2. 进入 tmux
3. 启动训练
4. Detach（`Ctrl+B D`）
5. 关闭 VSCode
6. 训练在服务器后台持续运行
7. 第二天重新 SSH 进来，`tmux attach -t lam-train` 查看进度