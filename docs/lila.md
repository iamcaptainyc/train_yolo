# LILA 相机陷阱数据集下载

**直接把数据集页面网址传给脚本即可，无需查询数据集名称：**

```bash
python -m pip install awscli
python download_lila.py https://lila.science/datasets/ena24detection
python download_lila.py https://lila.science/datasets/wcscameratraps --provider aws --workers 32
```

默认 AWS、32 并发，自动下载图片和可识别的标注。网址中的页面名不必与目录名称一致，例如 `ena24detection` 会通过实际云存储路径识别为 ENA24。也可以一次传入多个页面网址。

`download_lila.py` 读取 [LILA 官方相机陷阱目录](https://lila.science/wp-content/uploads/2023/06/lila_camera_trap_datasets.csv)，支持其收录的数据集，不将路径写死为 WCS。目前查询到 32 个条目，例如 WCS、NACTI、Caltech、Snapshot Serengeti、Snapshot Safari 等。官方增加条目后用 `--refresh` 更新目录即可。

输入页面网址时，脚本还会读取页面列出的云目录和标注链接，通过实际文件路径匹配目录；没有目录条目时可从页面公开目录直接生成配置。仅提供大压缩包、需要申请访问或无法识别图片目录的页面会明确报错，不会下载其他数据集。官方目录获取失败时，网址输入仍会尝试直接解析页面。

三个云源由同一个脚本调度，使用 [LILA 推荐的原生工具](https://lila.science/image-access)：

| `--provider` | 需安装的命令 | 图片传输方式 |
| --- | --- | --- |
| `aws`（默认） | `aws`，可用 `python -m pip install awscli` 安装 | 匿名 `aws s3 sync` |
| `azure` | [AzCopy v10](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10) 的 `azcopy` | `azcopy copy`，比较源文件更新时间 |
| `gcp` | [Google Cloud SDK](https://cloud.google.com/storage/docs/gsutil_install) 中的 `gsutil` | `gsutil -m rsync -r` |

仅使用哪个源，就安装哪个工具，并确保其位于 PATH 中。脚本本身仅依赖 Python 标准库；查看目录和 `--dry-run` 不需要安装云工具。云工具负责图片级并发、网络传输与错误输出。公开数据无需设置 AWS 密钥；脚本不修改现有 AWS 配置，使用临时配置控制并发。

## AutoDL 上使用

```bash
git pull --ff-only
python -m pip install awscli

# 查看所有数据集，或按关键字搜索
python download_lila.py --list
python download_lila.py --search snapshot

# 查看选定数据集所有源及标注地址，不下载数据
python download_lila.py wcs --info

# 查看将执行的命令，不运行云工具
python download_lila.py wcs --provider aws --dry-run

# 下载 WCS 图片、元数据和目录中列出的 bbox 文件
python download_lila.py wcs --provider aws --workers 32

# 也可使用全名或目录中的 short_name
python download_lila.py "WCS Camera Traps" --kind images
python download_lila.py caltech-camera-traps --kind metadata
```

名称使用大小写不敏感的精确匹配，或唯一的关键字匹配。模糊匹配到多个数据集时会列出候选，不会任意选一个。

默认输出结构：

```text
datasets/lila/
  .catalog.csv                         # 首次访问时下载的官方目录缓存
  lila_download_report.json            # 最近一次实际下载结果与退出码
  wcs-camera-traps/
    images/animals/...
    images/empty/...
    annotations/...
  nacti/
    images/...
    annotations/...
```

`--output` 始终指定公共根目录，其下自动创建数据集 short_name 子目录。不同源保持相同的本地目录布局。

## 切换下载源、控制并发

```bash
python download_lila.py wcs --provider aws --workers 32
python download_lila.py wcs --provider azure --workers 32
python download_lila.py wcs --provider gcp --workers 32
```

`--workers` 是**单个数据集内的文件传输并发**，不是同时下载的数据集数。可以从 16 或 32 开始，根据总吞吐、超时情况和磁盘性能调整；并发提高不保证提速。

可以一次提供多个数据集，按输入顺序逐个执行；每个数据集内部并发下载：

```bash
python download_lila.py wcs nacti caltech-camera-traps --provider aws --workers 32
```

失败命令默认额外重试 3 次（等待 5、10、20 秒），`--retries 0` 关闭脚本层重试。可用 `--retry-delay` / `--max-retry-delay` 控制初始等待和上限。单个命令最终失败后继续其他条目，最终汇总报告；有任何失败时脚本退出码为 1。云工具也可能自行重试。

```bash
python download_lila.py https://lila.science/datasets/ena24detection \
  --retries 10 --retry-delay 5 --max-retry-delay 60
```

进度默认直接显示在终端，包括当前任务编号（如 `[1/2][ena24/images]`）、当前尝试次数、结束耗时、已处理任务数与百分比。云工具的 stdout/stderr 直接连接终端，保留其文件/字节进度和传输速度。任务百分比按图片目录/标注等传输任务计数，**不代表总数据字节的百分比**；字节进度以云工具显示为准，尚在列举远端文件时可能无法立即显示总量。

## 重跑与恢复

图片目录通过云工具比较已有文件并补下载缺失/需要更新的文件；中断后直接重跑同一命令。不会删除本地多余文件。AWS/GCP 的比较与 Azure `ifSourceNewer` 的比较规则不同，不能将其视为统一的逐字节校验或压缩包断点续传。

默认自动恢复，无需额外参数，也可以显式加 `--resume`。每个任务的源地址、目标路径、运行/中断/失败/完成状态会原子保存到输出根目录下的 `.transfer-state/`，每次尝试及时更新；Ctrl+C 时先终止传输进程，再记录中断状态。即使异常退出未能记录，重跑仍会调用云工具比较图片目录，不会因状态文件显示“完成”而跳过整个图片检查。

标注文件使用云工具的单文件复制；完成后保存文件大小和 SHA-256。重跑时若源地址未变且本地内容校验通过，则跳过已完成标注；文件缺失、内容变化或上次未完成时重新传输。`--no-resume` 会重新执行标注传输，`--refresh` 同时刷新目录并重新检查标注，适用于远端在同一 URL 更新文件的情况。这两个选项都不会清空图片目录，图片仍然按云工具规则增量比较。

元数据及 bbox 压缩包仅下载、不自动解压。**恢复以已完成图片/任务为单位，不保证半张图片或半个标注压缩包跨运行按字节继续。** 图片若损坏而大小/时间等恰好满足工具跳过规则，仍可能需要手动检查处理。校验标注 SHA-256 验证的是本地内容与本次下载完成时一致，不代表每次都会检查远端版本变化。

## 只下载一部分或先下载标注

`--kind` 支持 `images`、`metadata`、`bbox`、`all`，默认 `all`。`all` 模式在目录未提供 metadata/bbox 地址时提示并跳过；显式选择缺失的标注类型则报错。metadata 可能只有图片/序列级分类，**有 metadata 不代表有检测框**。

```bash
# 训练检测模型前，可先检查框标注
python download_lila.py wcs --kind bbox
python download_lila.py wcs --kind metadata

# 只下载指定图片子目录；本地保留相对路径
python download_lila.py wcs --kind images --prefix animals/0011
```

`--prefix` 作用于图片目录，不会自动按物种或 bbox 筛选，不接收通配符或 `..`。选择物种或仅下载有框图片需要另行根据标注生成文件清单；当前脚本不提供这种筛选。

## 更新目录与标注版本

```bash
python download_lila.py --list --refresh
```

目录缓存位于指定输出根目录下的 `.catalog.csv`。`--refresh` 从官方重新获取；也支持 `--catalog my_lila_catalog.csv` 使用本地清单，离线运行或添加尚未列入官方表的条目。

本地 CSV 必须含 `name`、`short_name`，源地址列为 `image_base_url_aws/azure/gcp`、`metadata_url_aws/azure/gcp`、`bbox_url_aws/azure/gcp`，与官方列名一致。所选操作需要对应列。AWS 接受官方 S3 HTTP 地址或 `s3://`；GCP 接受 `storage.googleapis.com` 地址或 `gs://`；Azure 接受公开 Blob HTTPS 地址。

已知三个 LILA 公共镜像根目录的对象会按相同相对路径转换到所选云源，包括部分表格单元格填入另一云地址的情况。新版对象可能尚未同步到所有镜像，远端返回不存在时可以切换源或指定准确地址。

**官方目录中的部分 bbox 链接是历史版本，刷新目录不意味着标注一定是该数据集页面上的最新版。** 直接输入页面网址时会优先采用页面上可识别的标注链接；同时有 `with_classes` 和 `no_classes` 框标注时优先前者。仅输入名称时仍使用目录中的标注。也可以通过 `--bbox-url` 覆盖单个数据集的框标注地址，这个选项不影响图片目录。

ENA24 的检测框位于其 metadata JSON 中，没有单独的 bbox 下载项；默认 `all` 会下载 metadata，即使提示目录中无单独 bbox 地址也不表示缺少检测框。

例如，[WCS 页面](https://lila.science/datasets/wcscameratraps)提供带物种类别的 2022 年版框标注，而当前目录仍指向 2020 年版。使用 AWS 下载页面列出的带类别版本：

```bash
python download_lila.py wcs --provider aws --kind bbox \
  --bbox-url s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/wcs/wcs_20220205_bboxes_with_classes.zip
```

这个工具负责下载，不自动转换 COCO Camera Traps 为 YOLO，也不判断各类别是否具备完整框标注。后续训练前应检查标注类型、类别、空图及推荐的数据划分。
