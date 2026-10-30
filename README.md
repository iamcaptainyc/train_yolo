# train_yolo：YOLO11 / YOLO26 动物目标检测

使用 Ultralytics 做**多类别目标检测**：输出动物类别与边界框。提供通用 Kaggle、LILA 下载工具，以及 Animals Detection 数据集专用的标注转换和训练流程。LILA 默认只选有效框图片，并加入约占最终总数 5% 的明确空图负样本，支持网址列表及断点续传，见 [下载说明](docs/lila.md)。仓库不包含数据集、模型权重或训练结果；下载操作需要手动运行脚本。

## 项目结构

```text
train_yolo/
  download_kaggle.py       # 通用 Kaggle Datasets 下载
  download_lila.py          # LILA AWS / Azure / GCP 下载
  download_huggingface.py   # HF 数据集直接落盘、重试及 SnakeAid 本地 YAML
  lila_selection.py        # 有效框筛选、明确空图抽样与图片续传
  requirements-lila.txt    # 仅下载 LILA 所需 Python 依赖
  kaggle_resume.py          # 整包 Range 续传与校验
  download_weights.py      # 官方 YOLO11 / YOLO26 检测权重下载
  prepare_dataset.py       # Animals Detection 原始标注 → YOLO
  animal_data.py            # 通用原始标注索引、分组预览与逐层均衡采样
  split_yolo.py             # 批量重分 YOLO，默认 8:1:1，--in-place 直接更新原目录
  prepare_lila.py          # 五个 LILA 数据集合并类别 → YOLO，相机/序列分组
  train.py                 # 训练 / 恢复 / 评估 / 预测
  training_guard.py        # Linux 独立进程组守护与残留 DataLoader 回收
  tune.py                  # 短命令入口，自动命名实验目录
  search_yolo.py           # Optuna 轮转搜索 YOLOv8n/s、YOLO11n/s，导出最佳 n/s
  search_worker.py         # 独立 GPU 训练/评估阶段与逐候选结果
  requirements.txt         # Python 依赖
  tests/
    test_prepare_dataset.py
    test_download_kaggle.py
    test_kaggle_resume.py
    test_download_lila.py
    test_lila_selection.py
    test_prepare_lila.py
  docs/lila.md             # LILA 多源下载说明
  docs/lila_annotation_audit.md # 五数据集实际标注审计
  docs/lila_yolo.md         # LILA 类别转换及防泄漏划分
  docs/snake_data.md        # SnakeAid 下载与 Open Images 有框蛇筛选
  docs/model_search.md      # 四模型自动搜索、每轮测试及恢复说明
  docs/animal_data.md       # 通用索引/体型标签/19类方案与30K抽取命令
  docs/split_yolo.md        # 平铺/缺少划分数据集的批量重新划分用法
  datasets/                # 本地原始和转换后数据，Git 忽略
  runs/                    # 本地训练结果，Git 忽略
```

## 获取代码

```powershell
git clone https://github.com/iamcaptainyc/train_yolo.git
cd train_yolo
```

安装环境后，依次执行“下载 → 转换并检查预览 → 训练 → 测试集评估”。下载工具可用于其他 Kaggle 数据集；转换器只适用于下文描述的 Animals Detection 原始格式。

LILA 相机陷阱数据集使用独立下载器，默认 AWS、32 并发：

```bash
python -m pip install awscli
python download_lila.py https://lila.science/datasets/ena24detection
python download_lila.py https://lila.science/datasets/wcscameratraps --provider aws --workers 32

# 从网址列表并行下载，最多同时 2 个数据集，每个数据集 16 并发
python download_lila.py --urls-file examples/lila_urls.txt --mode parallel --dataset-workers 2 --workers 16
```

直接传入 LILA 数据集页面网址即可自动识别，无需查询数据集名字；也保留名称和关键词输入。支持下载图片/元数据/bbox、切换三云源和按子目录下载。默认显示原生传输进度及任务进度，并持久保存恢复状态；中断后重跑原命令即可增量补下载图片，已完成且校验通过的标注自动跳过。`--retries 10` 可增加失败重试次数。完整用法见 [LILA 下载说明](docs/lila.md)。LILA 标注格式与本仓库的 Animals Detection 转换器不同，下载后不能直接套用 `prepare_dataset.py`。

## 1. 安装环境

建议 Python 3.11 或 3.12。在本项目目录执行（以下为 PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch, ultralytics; print('torch:', torch.__version__, 'ultralytics:', ultralytics.__version__, 'CUDA:', torch.cuda.is_available())"
```

需要 NVIDIA GPU 训练时，先按 [PyTorch 安装页](https://pytorch.org/get-started/locally/)选择匹配驱动的 CUDA 安装命令，再安装本项目依赖。`CUDA: False` 时请先解决 PyTorch 环境，或使用 `--device cpu`。首次使用官方 `.pt` 模型会自动下载预训练权重；离线时将权重路径传给 `--model`。

YOLO11 和 YOLO26 的官方权重名分别为 `yolo11s.pt`、`yolo26s.pt`，可把 `s` 换成 `n/m/l/x`。官方接口参考：[YOLO11](https://docs.ultralytics.com/models/yolo11/)、[YOLO26](https://docs.ultralytics.com/models/yolo26/)、[训练参数](https://docs.ultralytics.com/modes/train/)。依赖采用兼容范围，实验环境确认可用后可执行 `pip freeze > requirements-lock.txt` 保存精确版本。

## 2. 下载并准备数据

下载 [Kaggle Animals Detection Images Dataset](https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset)，解压到例如 `datasets/raw/`。

也可以使用通用下载脚本（接口参考 [kagglehub 官方文档](https://github.com/Kaggle/kagglehub#download-dataset)）：

```powershell
python -m pip install --upgrade kagglehub
python download_kaggle.py antoreepjana/animals-detection-images-dataset --output datasets/raw
```

支持任意 `OWNER/DATASET`、完整 Kaggle 数据集网址和 `OWNER/DATASET/versions/N`，可一次传入多个数据集。省略 `--output` 时自动创建 `datasets/OWNER/DATASET/`；显式指定版本时使用 `datasets/OWNER/DATASET/version-N/`，按发布者区分同名数据集。目录相对于当前工作目录。单数据集的 `--output` 指定最终目录；多数据集的 `--output` 指定公共根目录，各自追加 `OWNER/DATASET[/version-N]`，避免覆盖。也可用 `--cache` 沿用 kagglehub 默认缓存（不能与 `--output` 同时使用）。示例：

```powershell
python download_kaggle.py "https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset" --version 1 --output datasets/animals_v1
python download_kaggle.py OWNER/DATASET --file data.csv --output datasets/single_file
python download_kaggle.py OWNER/DATASET --login
python download_kaggle.py antoreepjana/animals-detection-images-dataset
python download_kaggle.py OWNER/DATASET --cache
python download_kaggle.py OWNER/DATASET --output datasets/my_dataset --retries 5 --retry-delay 10 --max-retry-delay 60
```

多数据集并行下载，例如：

```powershell
python download_kaggle.py antoreepjana/animals-detection-images-dataset uciml/iris --workers 2
python download_kaggle.py OWNER_A/DATASET_A OWNER_B/DATASET_B --output datasets/batch --workers 2 --retries 5
```

默认最多同时下载 3 个数据集，`--workers 1` 改为串行。每个数据集独立重试、独立输出目录，日志带数据集标识；同一标识或网址重复输入时只下载一次。登录在开始任务前只执行一次。某个下载失败不会阻止其他数据集，最后汇总成功/失败数，并在有失败时以退出码 1 结束。所有输出目录会在开始前检查；目录相同或互为父子时拒绝执行，避免并行覆盖。批量模式不接受 `--file` / `--version`，版本应逐个写为 `OWNER/DATASET/versions/N`。

这里的并行是“多个数据集同时下载”，不对单个数据集压缩包做多连接分片。它可能改善总吞吐，但不能保证提高单数据集速度；若每个任务只有 20～30 KB/s，可先比较直连与代理线路的效果。提高重试次数主要改善失败后的恢复，不能提高带宽。

下载默认在超时、连接中断、HTTP 408/429/5xx 后额外重试 3 次（最多 4 次调用），等待 5、10、20 秒；`--retry-delay` 设置初始等待，`--max-retry-delay` 限制等待上限。`--retries 0` 关闭重试。认证失败、资源不存在、证书错误或本地文件错误结束对应任务。Ctrl+C 会取消排队任务及后续重试，正在进行的网络请求需等待返回或超时后退出。KaggleHub 内部也可能重试，因此总耗时和底层请求数可能更多。

普通模式使用自动命名或 `--output` 目录时，本轮超时留下的部分文件会在重试时由 KaggleHub 覆盖并重新下载；使用 `--cache` 时沿用其缓存机制。重试不会重新触发 `--login`。非空输出目录需要显式选择 `--resume` 续传，或 `--force` 重新下载。

### 中断后继续下载

先更新代码和下载依赖，再在**原来的工作目录**运行相同数据集并加上 `--resume`：

```bash
git pull --ff-only
python -m pip install --upgrade 'kagglehub>=1.0.2,<2'
python download_kaggle.py banuprasadb/wildlife-dataset --resume --retries 5
```

也可从首次下载就加 `--resume`，中断后原命令重跑即可。支持多数据集同时续传：

```bash
python download_kaggle.py OWNER_A/DATASET_A OWNER_B/DATASET_B --resume --workers 2
```

续传模式复用目标目录中的 `版本号.archive`，包括旧脚本留下的文件；使用 HTTP Range 只请求缺失字节。通过 `.kaggle-resume.json` 固定数据集和版本，避免“最新版本”变化后混用内容。下载失败保留部分文件，后续重试不会自动开启强制重下。完成后先校验远端 MD5（若提供）与 ZIP CRC，再解压并标记完成；已有完成标记则直接跳过。校验、解压需要额外时间，压缩包和解压文件同时存在时需要足够空间。

如果服务器忽略 Range、版本不一致或校验失败，会保留现有文件并明确报错，不会静默删除进度。`--resume` 面向整个 ZIP 数据集，不可与 `--force`、`--cache` 或 `--file` 同时使用；只有解压文件、没有压缩包或可识别完成标记的旧目录不能恢复压缩包下载。它也不支持从其他下载器的任意 `.part` 文件自动恢复。首次开始后应避免由另一个脚本进程同时操作同一输出目录。

默认下载整个数据集，`--file` 可限定到数据集内的单个文件或目录。`--force` 表示重新下载并可能清理已有内容，不能用于保留断点。使用专属目录，避免混入其他数据集；切换版本应选择新目录。访问私有或需同意条款的数据时，可用 `--login` 交互认证，并先在 Kaggle 页面接受相应条款。脚本不保存或打印凭据。此脚本面向 Kaggle Datasets，竞赛下载请使用 kagglehub 的 `competition_download`。

脚本支持原始目录结构（外面允许有一层或多层解压目录）：

```text
datasets/raw/animals/
  train/
    Cat/
      image001.jpg
      Label/image001.txt
    Polar bear/
      image002.jpg
      Label/image002.txt
  test/
    Cat/...
```

原始标注每行格式为 `类别名 xmin ymin xmax ymax`，坐标单位是**像素**。类别名允许空格；每张图片允许多个类别和多个框。脚本从标注内容建立全局类别映射，不把整张图强制标成目录名，也不将所有动物合并为一个类别。类名统一小写并合并连续空白，类别按名称排序。

**Kaggle 页面未能直接读取标注样本**：当前格式依据该数据集已有的[转换实现](https://github.com/devidlatkin/yolov8_animal_detection/blob/main/convert_dataset.py)核对。请下载后先运行准备脚本并检查预览。如果实际版本是 VOC XML、COCO JSON 或已转换的 YOLO 数字标签，不能直接使用这个转换器；已是 YOLO 时可直接将其 YAML 传给 `train.py --data`。

```powershell
python prepare_dataset.py --source datasets/raw --output datasets/animals_yolo
```

如果下载时采用自动命名目录，改用：

```powershell
python prepare_dataset.py --source datasets/antoreepjana/animals-detection-images-dataset --output datasets/animals_yolo
```

处理规则：

- 保留原始 `test/`；默认从 `train/` 按类别近似分层划分 15% 为 `val/`，随机种子为 42。生成验证集时保留每个训练类别至少一张训练图片；已有 `val/` 时原样保留。两个模型共用同一输出目录以获得相同划分。
- 按图片内容 SHA-256 去重；同一 split 的相同文件合并标注。跨原始 split 的重复图片会报错，需先整理，避免评估泄漏。不会识别重新压缩、裁剪后的近似重复图。
- 校验图片、缺失/空标签、非有限数、越界/退化框与 EXIF 方向。默认发现错误立即停止；可显式加 `--skip-invalid` 跳过有问题的整张图，错误记录保存在报告中。
- 输出已存在时拒绝覆盖，请指定新目录。默认 `--copy-mode auto` 优先硬链接图片（同一文件系统不重复占图片空间），不支持时回退复制。空间紧张时用 `--copy-mode hardlink`，无法硬链接则报错；需要独立副本用 `--copy-mode copy`。硬链接共享文件内容，原图及输出图片均应保持只读。实际方式记录在 `report.json` 的 `image_storage_modes`。

输出：

```text
datasets/animals_yolo/
  data.yaml
  images/{train,val,test}/
  labels/{train,val,test}/
  preview/       # 每个 split 默认 8 张带框预览
  report.json    # 类别、各 split 图片数/框数、缺失类别、跳过原因、划分清单
```

**开始训练前打开 `preview/` 检查框与类别，查看 `report.json` 的类别分布。** 小样本类别可能缺席验证集/测试集，见 `missing_classes`；多类别图的划分只是近似分层。

### AutoDL 首轮训练

已下载且使用默认自动命名目录时，直接执行：

```bash
git pull --ff-only
python -m pip install -r requirements.txt
python prepare_dataset.py --source datasets/antoreepjana/animals-detection-images-dataset --output datasets/animals_yolo --copy-mode hardlink
```

`--source` 应指向实际解压目录，脚本会自动查找内部的 train/test 层级；如果数据放在别处，仅替换该路径。上面的 hardlink 模式适合原始数据和输出均位于同一 AutoDL 数据盘的情况，不会为全部图片再占用一份空间。跨文件系统时改成 auto 或 copy，并确保足够空间。

转换成功后查看 `preview/` 和 `report.json`，然后先跑 1 epoch 验证流程，再训练首版：

```bash
python train.py --data datasets/animals_yolo/data.yaml --model yolo11s.pt --epochs 1 --batch 8 --device 0 --workers 4 --name animals_smoke
python train.py --data datasets/animals_yolo/data.yaml --model yolo11s.pt --epochs 50 --batch 8 --device 0 --workers 4 --name animals_v1
```

`yolo11s.pt` 若不在本地由 Ultralytics 获取；已有权重可传 `--model weights/yolo11s.pt`。batch=8 是起始配置，显存不足时减小。转换默认遇到坏图或异常标注会停止；确认要跳过时加 `--skip-invalid`，它会跳过整图并记入报告，不生成假负样本。转换期间显示图片校验/哈希和输出进度，尚未开始训练。

## 3. 训练 YOLO11 和 YOLO26

可先独立下载官方预训练检测权重（不会启动训练），默认下载 `yolo11s.pt` 和 `yolo26s.pt` 到 `weights/`：

```powershell
python download_weights.py
python download_weights.py --models yolo11n.pt yolo26n.pt --output weights --retries 5
python download_weights.py yolov8n.pt yolov8s.pt --output weights
```

支持直接传模型名或使用 `--models`，可混用并去重；不传仍下载 YOLO11s 和 YOLO26s。YOLOv8 文件名为 `yolov8n.pt`、`yolov8s.pt`（注意字母 `v`）。支持 YOLOv8、YOLO11、YOLO26 各自的 `n/s/m/l/x` 型号。使用 Ultralytics 自带下载器从官方 assets 仓库下载；已有权重跳过，文件大小检查不等于完整性或可加载性验证。下载较慢时可在 PowerShell 配置实际代理地址，例如 `$env:HTTPS_PROXY = "http://127.0.0.1:7890"`，端口应替换为实际值。

训练时指定本地文件路径：

```powershell
python train.py --model weights/yolo11s.pt --epochs 100 --batch 8 --device 0 --name yolo11s
python train.py --model weights/yolo26s.pt --epochs 100 --batch 8 --device 0 --name yolo26s
```

先做 1 个 epoch 的流程验证：

```powershell
python train.py --model yolo11n.pt --epochs 1 --batch 2 --device 0 --name smoke_yolo11
```

正式训练（以下 batch 是可调整的初始配置，不保证适配所有显卡）：

```powershell
python train.py --model yolo11s.pt --epochs 100 --imgsz 640 --batch 8 --device 0 --name yolo11s
python train.py --model yolo26s.pt --epochs 100 --imgsz 640 --batch 8 --device 0 --name yolo26s
```

训练从预训练权重微调；默认 seed=42、早停 patience=30、AMP 开启、不缓存图片。显存不足先降低 `--batch`，再考虑 `n` 模型或降低 `--imgsz`。Windows 默认 `--workers 0`，确认稳定后可调到 2 或 4。需要关闭混合精度可用 `--no-amp`。重复运行时 Ultralytics 会自动创建带序号的新目录，以下命令中的权重路径应按实际输出修改。

在相同划分、图像尺寸、epoch、batch 和硬件条件下比较两个模型。相同 seed 不保证不同硬件/库版本的结果完全一致，早停也可能使实际 epoch 不同。

断点恢复（仅用于中断、尚未完成的训练）：

```powershell
python train.py --resume runs/animals/yolo11s/weights/last.pt --device 0
```

恢复时沿用检查点记录的训练配置。如果训练已结束、想追加一轮微调，用 `--model .../last.pt --epochs 50 --name finetune` 新开训练。

### Linux 训练异常退出保护

需要自动比较 YOLOv8n/s、YOLO11n/s 时，运行 `python tune.py --data 数据集/data.yaml`，无需指定名称。原有 `--mode standard` 默认先跑 auto 基线，再比较 batch 并搜索参数，每模型 8 次。新增 `--mode progressive`（小尺寸筛选→正式尺寸训练）和 `--mode prune`（始终正式尺寸，30/80 轮剪枝→完整训练），均使用 Optuna Pruner、默认固定 batch=64。三种模式默认每个 COMPLETE 候选立即测试并保存指标，渐进模式包含筛选阶段；PRUNED 不测试。`--batch 128` 可指定固定 batch；`--dry-run` 查看计划。三种模式始终按验证集选优，详见 [完整命令与恢复说明](docs/model_search.md)。

在 Linux/AutoDL 中，`python train.py ...` 默认由一个不加载 PyTorch/CUDA 的轻量守护进程启动真正的训练进程，参数保持原样。每次运行创建独立进程组。训练正常完成、异常退出或收到 SIGKILL 后，只要守护仍存活，就会终止本组尚存的 DataLoader 等子进程；先 SIGTERM，等待后对未退出进程使用 SIGKILL。Ctrl+C/SIGTERM 转发给本次任务，超时后强制停止。Linux subreaper 用于回收被训练主进程遗留的后代进程，避免交给容器 PID 1 后长期残留。

这不会按名称批量杀其他 Python，也不会删除数据集或已保存权重。Windows 仍使用原直接运行方式；通过其他 Python 程序导入并直接调用 `train.main()` 时也不经过命令行守护。自行脱离本次进程组的外部程序不在清理范围。

`Killed` 不是足以判定显存不足的错误信息，可能是主机/容器 RAM 限制或外部 SIGKILL。96GB GPU 显存、宿主机 free 显示的总内存、容器可用内存是不同限制；worker 数量需要结合容器内存额度，不能只根据 CPU 核数增大。排查时降低 batch/workers，并检查 cgroup memory.events 和平台内存统计。

**若守护也被强杀、容器整体被杀或进程卡在不可中断内核 I/O，脚本无法保证清理。** 需要平台/宿主机层面的进程或 cgroup 管理。已有旧残留不会由新版本按名称自动清除，应先确认归属再清理，然后更新代码重新启动训练。新训练会打印 `[Guard] Worker PID/process group: ...`，被 SIGKILL 时返回退出码 137，并打印 RAM/外部终止排查提示，不将原因直接判为 CUDA OOM。

Linux 上可单独验证守护行为（不需要 CUDA 或模型）：

```bash
python -m unittest discover -s tests -p 'test_training_guard.py' -v
```

## 4. 测试集评估与预测

训练过程用 `val` 选择最优权重。模型和参数确定后在独立 `test` 上评估，避免根据测试集反复调参。

```powershell
python train.py --mode val --model runs/animals/yolo11s/weights/best.pt --split test --device 0 --name test_yolo11s
python train.py --mode val --model runs/animals/yolo26s/weights/best.pt --split test --device 0 --name test_yolo26s
python train.py --mode predict --model runs/animals/yolo11s/weights/best.pt --source path/to/images --device 0
```

关注 Precision、Recall、mAP50、mAP50-95 与各类别表现；训练目录含 `weights/best.pt`、`weights/last.pt`、`results.csv` 和训练曲线。评估目录包含混淆矩阵等图表，预测目录包含带框图片和标签。实际产物取决于 Ultralytics 版本和训练是否完整结束。

## 5. 代码验证

```powershell
python -m unittest discover -s tests -v
```

测试用临时合成图片检查坐标转换、多词类名、重复图合并、划分复现、跨 split 泄漏拦截和缺失标签行为，不需要下载数据或模型。真实数据训练和 GPU 性能需在下载数据、安装 Ultralytics 与对应 PyTorch 后验证。

### LILA 五数据集合并与审计

LILA 常用类别处理完成后，用 [YOLO 合并脚本](docs/merge_yolo.md) 与已转换的 Animals Detection 合并，统一英文名/ID；检查所有类别，从原标注恢复细类和中间类别，达到 100 张独立，少于 100 张逐级归入上级，再按相机/序列重新划分（目标比例 90%/5%/5%）；不按数量抽样，单一物种超过 10,000 张也保留全部有效图片，图片只用硬链接：

```bash
python merge_yolo_datasets.py --lila yolo_datasets/lila_camping --animals yolo_datasets/animals-detection-images-dataset --lila-source datasets/lila --output yolo_datasets/camping_all_classes --min-class-images 100 --max-class-images 10000 --cap-class-images 0 --workers 16
```

露营识别任务使用新的[常用英文类别策略](docs/lila_camping.md)：保留熟悉细类与大类、排除未确定动物类别，野生猪类统一为 `wild_boar`，与 `pig` 分开。已经通过审计的硬链接输出可用 `--remap-from` 从原始标注恢复细类、保留划分，避免再次全量解码和计算图片哈希：

```bash
python prepare_lila_audited.py --remap-from yolo_datasets/lila --output yolo_datasets/lila_camping --workers 16
```

下载完成后使用 `prepare_lila_audited.py`：按官方分类树处理父子类别，默认按相机/序列连通组划分 train/val/test = 90%/5%/5%，支持 `--workers 16` 并发校验/哈希/写出/输出审计，核对原始标注与下载清单，独立反算所有输出框，并生成类别数量、合并规则和框图的 `AUDIT_REPORT.md`。运行与审计口径见 [五数据集审计合并说明](docs/lila_audited_merge.md)。

### LILA 自动清理与断点恢复

`prepare_lila_audited.py` 默认排除损坏图片和无法安全修复的整图标注，自动修正至多 1 像素的越界框，完成后生成 `AUDIT_REPORT.md`、完整剔除清单和可训练的 `data.yaml`。原始图片保留。默认不保存正常图片的逐图校验、哈希或逐框明细，只保存每 128 张的批次断点与异常/修正清单。中断后重跑相同命令与输出目录，跳过未变化且已完成批次的解码检查，未完成批次最多重检 128 张；内容哈希仍在内存中重算用于去重分组。已有旧逐图缓存会归并成批次断点，再自动删除旧大文件。详见 [自动处理与恢复说明](docs/lila_audited_merge.md#默认自动处理与恢复)。

```bash
python -m pip install -r requirements-lila.txt
python prepare_lila_audited.py --source datasets/lila --output datasets/lila_yolo_auto --copy-mode hardlink --workers 16
```

若 `Write YOLO` 已到 100%，后面的输出审计才失败：更新后在原命令末尾加 `--resume-output`，保留原输出目录和其他参数。此模式直接恢复既有划分和标签审计，禁止进入全量输入图片解码/哈希阶段。见 [已写出结果直接恢复](docs/lila_audited_merge.md#write-yolo-已完成输出审计失败直接恢复输出)。

## 定位与修复 JPEG 解码警告

先暂停训练，然后运行 `python repair_images.py yolo_datasets/camping_animals/images --workers 16`。递归检查图片，只修复有问题且可解码的 JPEG，原位同步修复所有硬链接；无法解码则删除图片及标签，并重算数据集报告数量；支持进度和 Ctrl+C 续跑，只记录问题图片。结果见 `runs/image_repair/REPORT.md`，详见 [图片修复说明](docs/image_repair.md)。

## 训练被终止的诊断日志

Linux 训练与 Optuna 子任务默认生成 `runs/diagnostics/<自动名称>/`，记录完整控制台输出、每 5 秒资源快照、训练/验证阶段和退出诊断。原命令无需改动，下次异常优先查看 `DIAGNOSIS.md`。详见 [诊断日志说明](docs/training_diagnostics.md)。

## 提取不超过 30000 张的 56 类小数据集

运行 `python extract_camping.py --ap10k /路径/ap-10k --workers 16`，先统一 AP-10K、Animals Detection 与已处理的 LILA，再全局均衡抽样，最后按分组重新划分。优先 AP-10K→Animals→LILA；LILA 至少一个框的面积占比严格大于 1/16 才保留整图和全部框。硬链接，总数含负样本不超过 30000，详见 [小数据集提取说明](docs/camping_subset.md)。

只训练指定模型：`python tune.py --data 数据集/data.yaml --models yolo11s.pt`；两个模型用 `--models yolo11n.pt yolo11s.pt`。不指定则训练原四个模型。模型选择参与自动命名；切换模型组合时不复用旧的显式输出目录。`--trials` 仍是每个模型的试验总次数。

补齐已有搜索实验的测试集指标（不重新训练）：`python tune.py --test-only --output runs/animals_search_batch`。默认新实验逐候选测试；详见 [搜索与补测说明](docs/model_search.md)。

保留全局类别 ID 合并多个 YOLO 数据集：`python merge_yolo_ids.py --datasets 数据集A 数据集B --output 新输出目录`。默认保留所有类别名称；占位 ID 必须显式指定，例如 `--reserved-ids 2:0-26`。默认硬链接并保留原划分，详见 [通用 ID 合并说明](docs/merge_yolo_ids.md)。
