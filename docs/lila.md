# LILA 相机陷阱数据集下载

**直接把数据集页面网址传给脚本即可，无需查询数据集名称：**

```bash
python -m pip install -r requirements-lila.txt awscli
python download_lila.py https://lila.science/datasets/ena24detection
python download_lila.py https://lila.science/datasets/wcscameratraps --provider aws --workers 32
```

默认 AWS、32 并发，先下载标注，筛选有效动物框图片，再加入约占最终图片总数 5% 的明确空图负样本。网址中的页面名不必与目录名称一致，例如 `ena24detection` 会通过实际云存储路径识别为 ENA24。也可以一次传入多个页面网址。

`download_lila.py` 读取 [LILA 官方相机陷阱目录](https://lila.science/wp-content/uploads/2023/06/lila_camera_trap_datasets.csv)，支持其收录的数据集，不将路径写死为 WCS。目前查询到 32 个条目，例如 WCS、NACTI、Caltech、Snapshot Serengeti、Snapshot Safari 等。官方增加条目后用 `--refresh` 更新目录即可。

输入页面网址时，脚本还会读取页面列出的云目录和标注链接，通过实际文件路径匹配目录；没有目录条目时可从页面公开目录直接生成配置。仅提供大压缩包、需要申请访问或无法识别图片目录的页面会明确报错，不会下载其他数据集。官方目录获取失败时，网址输入仍会尝试直接解析页面。

三个云源由同一个脚本调度。默认筛选模式使用公开 HTTPS 地址按清单并发下载图片，复用连接；标注下载和显式选择 `--image-policy all` 时使用 [LILA 推荐的原生工具](https://lila.science/image-access)：

| `--provider` | 需安装的命令 | 全目录模式图片传输方式 |
| --- | --- | --- |
| `aws`（默认） | `aws`，可用 `python -m pip install -r requirements-lila.txt awscli` 安装 | 匿名 `aws s3 sync` |
| `azure` | [AzCopy v10](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10) 的 `azcopy` | `azcopy copy`，比较源文件更新时间 |
| `gcp` | [Google Cloud SDK](https://cloud.google.com/storage/docs/gsutil_install) 中的 `gsutil` | `gsutil -m rsync -r` |

仅使用哪个源，就安装哪个工具，并确保其位于 PATH 中。筛选模式需要 `requirements-lila.txt` 中的依赖；查看目录和 `--dry-run` 不需要安装云工具。公开数据无需设置 AWS 密钥；脚本不修改现有 AWS 配置，使用临时配置控制并发。

## AutoDL 上使用

```bash
git pull --ff-only
python -m pip install -r requirements-lila.txt awscli

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

## 网址列表、串行和并行

直接提供多个网址，默认按输入顺序串行执行：

```bash
python download_lila.py \
  https://lila.science/datasets/ena24detection \
  https://lila.science/datasets/wcscameratraps \
  --mode serial
```

也可创建 `urls.txt`，每行一个网址，支持空行和以 `#` 开头的整行注释（示例文件见 `examples/lila_urls.txt`）：

```text
# Camera trap datasets
https://lila.science/datasets/ena24detection
https://lila.science/datasets/wcscameratraps
```

```bash
# 列表串行下载
python download_lila.py --urls-file urls.txt --mode serial

# 同时下载 2 个数据集，每个数据集最多 16 个文件传输并发
python download_lila.py --urls-file urls.txt --mode parallel --dataset-workers 2 --workers 16
```

`--urls-file` 也接受 `.json` 文件，内容为网址字符串数组。命令行网址和列表文件可同时使用，合并后去重；不同网址若识别为同一数据集，也只调度一次。仍兼容名称和关键词输入：

```bash
python download_lila.py wcs nacti caltech-camera-traps --provider aws --workers 32
```

`--mode serial`（默认）每次执行一个数据集；`--mode parallel` 通过 `--dataset-workers`（默认 2）限制同时运行的数据集数。筛选模式会先获取所有数据集的标注，再逐一分析、确定整批负样本配额，最后按所选模式下载图片。总传输并发可能接近 `dataset-workers × workers`，应结合网络与磁盘性能控制。

各数据集独立记录恢复状态、独立重试；一个数据集的传输失败不阻止其他数据集。显示单个数据集内的任务进度和总体已处理数据集数。并行运行时，云工具的原生传输输出可能交错，可通过带数据集名称的任务行和最终报告确认状态。Ctrl+C 会取消排队任务、终止正在运行的云工具并保存恢复状态；重跑相同命令继续。网址解析与目录检查会在实际下载前完成，无法解析的网址会在这一阶段报错。

失败命令默认额外重试 3 次（等待 5、10、20 秒），`--retries 0` 关闭脚本层重试。可用 `--retry-delay` / `--max-retry-delay` 控制初始等待和上限。单个命令最终失败后继续其他条目，最终汇总报告；有任何失败时脚本退出码为 1。云工具也可能自行重试。

```bash
python download_lila.py https://lila.science/datasets/ena24detection \
  --retries 10 --retry-delay 5 --max-retry-delay 60
```

筛选图片下载显示图片数量进度、处理速度、累计新下载 MiB、跳过数和失败数。标注下载保留云工具的原生输出。分析标注时显示阶段信息，大型 JSON 的扫描可能需要等待；并行下载时各数据集各有进度条。

## 检测框筛选与负样本

默认 `--image-policy boxed --negative-fraction 0.05 --seed 42`。详细的五数据集核查见 [标注审计](lila_annotation_audit.md)。

- 正样本必须有有效的动物框：COCO `[x,y,width,height]` 像素坐标，宽高大于零、有限数值，并按图片尺寸检查越界。排除人类、vehicle、ignore、损坏或私有图片，以及含无效动物框的图片。
- 负样本必须由元数据明确标记为 `empty`、`blank` 等空场景，且没有矛盾的非空标注或物体框。缺框、缺标注、有动物但没有检测框都不能作为负样本；类别 ID 为 0 也不自动代表空图。
- 5% 指**整批最终图片中的负样本比例**，计算 `N=round(P×0.05/0.95)`。先按各数据集正样本数分配，再从其他数据集的明确空图池补足缺额。ENA24 没有明确空图，因此不提供负样本；单独下载 ENA24 会报告不足。
- 空图总数不足时如实报告，不使用未知图片凑数。固定标注版本、输入顺序和 seed 可复现抽样。零正样本或无法解析标注时明确失败，不退回全目录下载。

```bash
# 示例列表已包含本次五个网址；先只下载标注和生成清单
python download_lila.py --urls-file examples/lila_urls.txt --audit-only

# 下载筛选结果：两个数据集同时下载，各 32 个图片线程
python download_lila.py --urls-file examples/lila_urls.txt --mode parallel --dataset-workers 2 --workers 32

# 只要有框图片，不加入空图
python download_lila.py https://lila.science/datasets/caltech-camera-traps --negative-fraction 0

# 明确需要全量图片时才启用
python download_lila.py https://lila.science/datasets/caltech-camera-traps --image-policy all
```

每个数据集的 `selection/` 保存：

| 文件 | 用途 |
| --- | --- |
| `selected_images.txt` / `positives.txt` / `negatives.txt` | 相对于该数据集 `images/` 的路径清单 |
| `selected_images.jsonl` | 图片、角色和框标注，逐行记录 |
| `selected_annotations.json` | COCO 子集，负样本有图片记录、没有对象标注；重新分配图片/标注 ID，保留 `source_image_id` |
| `selection_report.json` | 筛选计数、来源、随机种子和负样本不足统计 |
| `negative_candidates.jsonl` | 明确空图候选池，供整批配额调整 |
| `failed_images.json` | 本次下载失败的图片及原因 |
| `downloaded_images.sqlite` | 完成图片的 URL/大小检查点 |

输出根目录另外保存 `selection_batch_report.json` 和 `lila_download_report.json`。图片仍保留原有相对路径。**已有目录里的多余图片不会被删除，训练必须使用筛选清单/COCO 子集，不能直接递归读取整个旧目录。** 类别仍保留各数据集原始定义；混合训练前需要统一类别并转换 YOLO。`prepare_lila.py` 可将这五个数据集合并为 YOLO 并按相机位置/序列划分，详见 [转换说明](lila_yolo.md)。`prepare_dataset.py` 仅适用于 Animals Detection 的原始格式。

## 重跑、重试与恢复

直接重跑同一命令即可。筛选图片以 `.part` 保存未完成内容，并记录 URL、ETag；服务端支持 Range 时续传，忽略 Range 或对象变化时重下该图片。完成后验证图片格式再原子改名，SQLite 记录 URL/大小；下次按 URL 和本地大小跳过完成项。已有全量下载图片没有检查点时，比较远端长度并验证图片后复用。图片完成检查点不提供每次运行的逐字节校验。

网络错误按 `--retries`、`--retry-delay`、`--max-retry-delay` 退避重试；404 等永久错误记录后继续其他图片。最终存在任何失败时退出码为 1。Ctrl+C 停止排队任务，活动 HTTP 请求在返回或超时后结束，保留部分文件；重新运行会继续。

标注文件仍由云工具下载，完成后记录大小和 SHA-256，重跑校验通过即跳过。`.transfer-state/` 保存这些传输任务状态。标注压缩包不保证字节级续传，筛选时在 `annotations/.json-cache/` 缓存解压后的 JSON，不解压图片包。

`--no-resume` 重新检查标注及图片完成记录；`--refresh` 还刷新官方目录，适用于同 URL 的标注更新，不会清空图片目录。`--image-policy all` 使用原生工具的增量比较规则，不保证半张图片跨运行续传。

## 只下载部分或先下载标注

### 本地存储错误时如何恢复

`sqlite3.OperationalError: unable to open database file` 是本地完成检查点的访问/写入错误，不能仅凭这一条确定原因。尤其在大量图片突然失败后，应先检查目标盘空间、磁盘配额、inode、挂载和目录权限。程序不会因此删除已经完成的图片。

```bash
df -h datasets/lila /tmp
df -i datasets/lila /tmp
du -sh datasets/lila
```

新版在 SQLite 检查点失败，或图片写入遇到磁盘满/配额/权限/只读/IO 错误时停止整批任务，保留完成文件和 `.part`，避免快速产生大量失败记录。修复本地存储后，更新代码并以相同输出目录重跑原下载命令。SQLite 未记录到最后一张已完成图片时，会比较长度和验证图片以复用，不需要删图片或重头下载。

进度条 `100%` 表示任务已处理，仍要检查 `failed` 是否为零。异常发生时最终报告可能来不及写入，以终端错误为准；不要删除 `images/`、`.part` 或 `selection/downloaded_images.sqlite` 来尝试修复空间问题。

`--kind` 支持 `images`、`metadata`、`bbox`、`all`，默认 `all`。在筛选模式选择 `images` 仍会先获取所需标注；单选 `metadata` 或 `bbox` 则仅下载对应文件。metadata 可能只有分类，存在 metadata 并不表示有检测框。

```bash
python download_lila.py https://lila.science/datasets/caltech-camera-traps --kind bbox
python download_lila.py https://lila.science/datasets/ena24detection --kind metadata
```

`--prefix` 限制图片子目录并保留相对路径，可与框筛选结合，不接收通配符或 `..`。`--bbox-url` 可覆盖单个数据集的框标注地址。ENA24、Oregon、Nkhotakota 的框嵌在 metadata 中，没有单独 bbox 下载项。

## 更新目录与标注版本

```bash
python download_lila.py --list --refresh
```

目录缓存位于指定输出根目录下的 `.catalog.csv`。`--refresh` 从官方重新获取；也支持 `--catalog my_lila_catalog.csv` 使用本地清单，离线运行或添加尚未列入官方表的条目。

本地 CSV 必须含 `name`、`short_name`，源地址列为 `image_base_url_aws/azure/gcp`、`metadata_url_aws/azure/gcp`、`bbox_url_aws/azure/gcp`，与官方列名一致。所选操作需要对应列。AWS 接受官方 S3 HTTP 地址或 `s3://`；GCP 接受 `storage.googleapis.com` 地址或 `gs://`；Azure 接受公开 Blob HTTPS 地址。

已知三个 LILA 公共镜像根目录的对象会按相同相对路径转换到所选云源，包括部分表格单元格填入另一云地址的情况。新版对象可能尚未同步到所有镜像，远端返回不存在时可以切换源或指定准确地址。

**官方目录中的部分 bbox 链接是历史版本，刷新目录不意味着标注一定是该数据集页面上的最新版。** 直接输入页面网址时会优先采用页面上可识别的标注链接；同时有带类别/物种和无类别框标注时优先带类别/物种的版本。仅输入名称时仍使用目录中的标注。也可以通过 `--bbox-url` 覆盖单个数据集的框标注地址，这个选项不影响图片目录。

ENA24 的检测框位于其 metadata JSON 中，没有单独的 bbox 下载项；默认 `all` 会下载 metadata，即使提示目录中无单独 bbox 地址也不表示缺少检测框。

例如，[WCS 页面](https://lila.science/datasets/wcscameratraps)提供带物种类别的 2022 年版框标注，而当前目录仍指向 2020 年版。使用 AWS 下载页面列出的带类别版本：

```bash
python download_lila.py wcs --provider aws --kind bbox \
  --bbox-url s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/wcs/wcs_20220205_bboxes_with_classes.zip
```

这个工具按图片是否有有效框进行筛选，不自动转换 COCO Camera Traps 为 YOLO，也不能保证每张入选图片的全部动物都已被人工完整标注。后续训练前应检查标注类型、类别、空图及推荐的数据划分。
