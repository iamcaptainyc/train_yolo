# train_yolo：YOLO11 / YOLO26 动物目标检测

使用 Ultralytics 做**多类别目标检测**：输出动物类别与边界框。提供通用 Kaggle、LILA 下载工具，以及 Animals Detection 数据集专用的标注转换和训练流程。仓库不包含数据集、模型权重或训练结果；下载操作需要手动运行脚本。

## 项目结构

```text
train_yolo/
  download_kaggle.py       # 通用 Kaggle Datasets 下载
  download_lila.py          # LILA AWS / Azure / GCP 下载
  kaggle_resume.py          # 整包 Range 续传与校验
  download_weights.py      # 官方 YOLO11 / YOLO26 检测权重下载
  prepare_dataset.py       # Animals Detection 原始标注 → YOLO
  train.py                 # 训练 / 恢复 / 评估 / 预测
  requirements.txt         # Python 依赖
  tests/
    test_prepare_dataset.py
    test_download_kaggle.py
    test_kaggle_resume.py
    test_download_lila.py
  docs/lila.md             # LILA 多源下载说明
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
```

直接传入 LILA 数据集页面网址即可自动识别，无需查询数据集名字；也保留名称和关键词输入。支持下载图片/元数据/bbox、切换三云源和按子目录下载。图片下载中断后可重跑原命令，按所选云工具规则比较并跳过已有图片。完整用法见 [LILA 下载说明](docs/lila.md)。LILA 标注格式与本仓库的 Animals Detection 转换器不同，下载后不能直接套用 `prepare_dataset.py`。

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

- 保留原始 `test/`；默认从 `train/` 按类别近似分层划分 15% 为 `val/`，随机种子为 42。已有 `val/` 时原样保留。两个模型共用同一输出目录以获得相同划分。
- 按图片内容 SHA-256 去重；同一 split 的相同文件合并标注。跨原始 split 的重复图片会报错，需先整理，避免评估泄漏。不会识别重新压缩、裁剪后的近似重复图。
- 校验图片、缺失/空标签、非有限数、越界/退化框与 EXIF 方向。默认发现错误立即停止；可显式加 `--skip-invalid` 跳过有问题的整张图，错误记录保存在报告中。
- 输出已存在时拒绝覆盖，请指定新目录。转换复制图片，需要额外磁盘空间；不会修改原始数据。

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

## 3. 训练 YOLO11 和 YOLO26

可先独立下载官方预训练检测权重（不会启动训练），默认下载 `yolo11s.pt` 和 `yolo26s.pt` 到 `weights/`：

```powershell
python download_weights.py
python download_weights.py --models yolo11n.pt yolo26n.pt --output weights --retries 5
```

`--models` 支持 YOLO11、YOLO26 各自的 `n/s/m/l/x` 型号，可同时指定多个。使用 Ultralytics 自带下载器从官方 assets 仓库下载；已有权重跳过，文件大小检查不等于完整性或可加载性验证。下载较慢时可在 PowerShell 配置实际代理地址，例如 `$env:HTTPS_PROXY = "http://127.0.0.1:7890"`，端口应替换为实际值。

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
