# 训练进程退出诊断日志

Linux 上原来的训练命令无需添加参数。`train.py` 和 `tune.py` 的训练/验证子进程由独立守护进程采样，启动时打印：

```text
[Guard] Diagnostics: /.../runs/diagnostics/20261022T103000Z-abc123
```

每次调用都有独立目录，恢复训练也会创建新的日志目录，旧日志保留。时间戳统一 UTC（北京时间加 8 小时）。可以用环境变量 `TRAIN_YOLO_LOG_ROOT=/你的日志目录` 指定存放位置。

| 文件 | 内容 |
| --- | --- |
| `DIAGNOSIS.md` | 退出后的中文诊断：原始返回码、信号判断、本次 OOM/内存压力计数增量、采样到的 RAM 峰值及限额 |
| `console.log` | 训练子进程的完整标准输出和标准错误，包含 JPEG 警告、异常堆栈、训练及验证进度；同时回显到终端 |
| `resources.jsonl` | 启动信息、约每 5 秒资源快照、退出前状态、清理后残留进程 |
| `phases.jsonl` | 模型训练开始、epoch 开始/结束、验证开始/结束、权重保存及训练结束；记录运行目录和部分配置 |

资源快照包括：

- 当前可见 cgroup v2 和可访问上级的 `memory.current/peak/high/max`、`memory.events/events.local`、内存分类、swap、内存与 I/O 压力、CPU 配额消耗统计。
- 同进程组的进程 PID/PPID、状态、RSS、CPU ticks、读写计数、等待点 wchan；RSS 含共享页，不能直接相加当作容器实际内存。
- `/dev/shm` 的总量、使用量和剩余量。
- 宿主机 meminfo 和内存/I/O pressure（宿主机内存不等于容器额度）。
- `nvidia-smi` 各 GPU 的显存使用、总显存和利用率；查询设有 2 秒超时，失败会记录原因。它是整张卡的数据，可能包含其他进程。
- 阶段日志在 CUDA 已初始化时记录训练进程的 allocated/reserved/peak allocated，不主动建立新 CUDA 上下文。

监控不使用 torch 初始化 GPU，不扫描数据集，不记录整个环境变量。资源日志边采样边刷新，不依赖训练进程正常退出。日志增量取本次开始和结束的差值，避免把以前累积的 OOM 事件误认成这次问题。旧内核或受权限限制的字段标记 unavailable；cgroup v1 不提供上述完整 v2 计数，不能因此推断没有 OOM。

## 如何判断

1. 先看 `DIAGNOSIS.md`。SIGKILL 且本次 `oom_kill` 增加，支持有 OOM 杀进程；若 cgroup 共享，仍不能仅凭计数确定被杀对象。
2. `high/max` 增加只说明内存压力或触及上限，不等于发生 OOM 杀进程。`memory.peak` 可能是容器累计峰值；报告单独保存本次采样到的最大 current 值。
3. SIGKILL 且没看到 `oom_kill` 增量，结论是原因未确定，需要平台的宿主机/上级 cgroup/操作记录；不能自动认定人为终止。
4. 若守护进程收到 Ctrl+C/SIGTERM，会单独记录收到的信号，以及是否在宽限期后主动发送 SIGKILL。
5. 对照 `phases.jsonl` 和资源时间线判断是否在验证开始后内存增长；`on_model_save` 表示发生过权重保存回调，恢复前仍要确认实际 checkpoint 文件。
6. 清理后的 `remaining_processes` 如果包含 D 状态，说明仍有不可中断等待，结合 wchan 和 I/O pressure 进一步分析。

整个容器或守护进程也被 SIGKILL 时，可能没有最终诊断文件；此时保留的资源/阶段/控制台日志仍可分析，但不能保证知道是谁发出的信号。5 秒采样也可能错过瞬时峰值。日志目录需要足够可写空间；控制台输出量决定日志大小。

Windows 直接训练暂不提供 Linux 守护进程资源监控；当前功能面向 AutoDL Linux。测试包含信号模拟、真实子进程输出捕获、cgroup 增量判断和模型阶段回调；Linux 真实进程组测试仅在 Linux 环境运行。
