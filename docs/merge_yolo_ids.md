# 保留类别 ID 合并 YOLO 数据集

`merge_yolo_ids.py` 用于已统一全局类别 ID 的 YOLO 检测数据，不绑定动物类别、来源数量、具体 ID 范围或总类别数。不要与带动物类别归并逻辑的 `merge_yolo_datasets.py` 混用。

```bash
python merge_yolo_ids.py \
  --datasets /path/to/dataset_a /path/to/dataset_b \
  --output /path/to/merged_dataset
```

每个输入可以是包含 `data.yaml` 的目录，也可以直接是 YAML 文件；支持单个数据集抽取，或多个数据集合并。输出必须是独立的新目录。

## 只提取指定类别

只取一份数据中的原 ID `0、25、26`，单独生成可训练的三类数据集：

```bash
python merge_yolo_ids.py \
  --datasets /path/to/dataset_a \
  --include-ids 1:0,25,26 \
  --compact-ids \
  --output yolo_datasets/subset_3
```

- `--include-ids 1:0,25,26`：只选择第一个来源的这些类别。支持逗号、范围、重复参数，例如 `--include-ids 1:0,25-26 --include-ids 2:27-35`；未指定筛选的来源全部保留。
- 默认保留原 ID。这个单独导出的例子有编号缺口，必须显式加 `--compact-ids`，按原 ID 升序映射为 `0→0、25→1、26→2`。合并后全局 ID 本就连续时可以不加，不会自动改号。
- 有所选类别框的图片保留，只输出所选类别的标注行；混合图片中的其他类别不再是本次检测任务的目标。不含所选类别的正样本图片排除，不把它们变成负样本。
- 筛选来源的原始空标签负样本默认排除，添加 `--keep-negatives` 才保留。未筛选来源继续按原行为保留负样本。
- 指定的类别必须在对应 YAML 中存在；所选 ID 与 reserved ID 重叠会报错。`--reserved-ids` 仍仅用于未使用的占位定义，不能用它删除真实标注。
- 先检查原标签，再筛选。标签不合法、使用未声明或 reserved ID 仍会报错。筛选后 train 和 val 都必须非空，否则在创建输出前报错。
- 图片默认硬链接、平铺保存；保留原划分。标签独立写入，保留框的坐标字符串不变。报告列出选择的类别、原 ID→新 ID 映射、各来源/划分排除图片数及删除框数，以及最终每类图片数和框数。

合并时只筛选第一份，第二份全部保留：

```bash
python merge_yolo_ids.py \
  --datasets /path/to/dataset_a /path/to/dataset_b \
  --include-ids 1:0,25,26 \
  --compact-ids \
  --output yolo_datasets/merged_subset
```

同 ID 不同名称仍会报错；compact 只压缩统一后的全局编号，不解决语义冲突。其他来源若也需筛选或声明占位，分别加对应来源编号的参数。

规则：

- 从各 YAML 的 `names` 读取类别，支持列表或字典。输入字典可以只声明本数据集的高位 ID。
- 默认所有名称都是真实类别，绝不根据 reserved、reserved_N 等字符串自动忽略。只有显式指定 `--reserved-ids 2:0-26` 才忽略第二个输入中已声明的 0～26；范围与该来源 names 取交集，不存在的 ID 不报错。被忽略的 ID 不能被标签实际使用。名称本身不参与占位判断。第一份数据的真实 0～26 填补后，第二份的 27～82 保持原值，最终 nc=83。
- 参数格式为 `来源序号:ID列表`，来源序号按 --datasets 顺序从 1 开始。支持范围、逗号和重复参数，例如 `--reserved-ids 2:0-26 --reserved-ids 3:0,2-5`。默认不传；范围只作用于指定来源，不影响其他来源的同 ID。
- 标签中实际使用了占位 ID 或未声明 ID 时停止，不能靠另一来源同 ID 的名称掩盖输入错误。
- 同 ID、相同名称允许；同 ID、不同真实名称报错。不同 ID 即使名称相同，也保持原 ID，不自动做语义合并。
- 最终真实类别表必须从 0 连续；没有被真实类别填补的占位缺口会报错。不会自动改 ID；只有显式添加 `--compact-ids` 才重排。
- 合并原 train/val/test，不重新划分。缺少 test 时允许，最终只在有 test 图片时声明 test；train 和 val 必须有图片。
- YAML 的 `path`、各 split 目录（含子目录）、目录列表或图片清单 txt 都支持。相对 `path` 按 YAML 目录解析，相对 split 按数据根目录解析，相对图片清单行按 txt 所在目录解析。清单行是一张图的完整路径，不解析行内注释。
- 图片路径需采用标准 `images/...` 与 `labels/...` 对应布局，标签 TXT 与图片同名。图片支持 jpg/jpeg/png/bmp/webp/tif/tiff。缺失标签报错；空 TXT 保留为负样本；只接受五列的目标检测标签，不支持分割/姿态标签。
- 图片平铺到 images/train、images/val、images/test，文件名加来源编号和路径指纹，避免同名覆盖。指纹只对路径字符串计算，不逐张哈希图片。未启用筛选/重排时标签内容完整保留；启用时仅删除未选类别行、按需改 ID，保留框的坐标字符串不变。
- 默认硬链接，不自动退化成复制；不同文件系统可显式加 `--copy-images`。标签独立写入，图片不解码、不改像素。
- 检查已知同一文件或硬链接跨 split 的情况；不对独立复制的图片做全内容去重，也不能识别相邻帧或判断标注语义。数据源存在重复内容时，应另行做去重/分组后再决定划分。

合并后输出 `data.yaml`、`report.json`、`MERGE_REPORT.md`，列出忽略的占位 ID、各划分图片数、每类图片数及框数。完成前保留 INCOMPLETE.txt；出错或中断后使用新输出目录重新执行。源数据不修改。

这里保留 ID 的前提是它们已经具有全局一致的含义。两个独立训练、都从 ID 0 开始但类别不同的数据集，应先显式重新映射 ID，不能直接靠本脚本忽略冲突。

你的 0～26 与 27～82 数据（第二份 YAML 带占位项）使用：

```bash
python merge_yolo_ids.py \
  --datasets /path/to/dataset_a /path/to/dataset_b \
  --reserved-ids 2:0-26 \
  --output /path/to/merged_83
```

未显式指定占位参数时，冲突按正常类别冲突报错，不静默跳过。报告记录每个被显式忽略的 ID 和原名称。

如果第一份只声明 0～28，也可以增加 `--reserved-ids 1:27-82`：实际仅忽略第一份的 27、28，未声明的 29～82 无需处理。与 `--reserved-ids 2:0-26` 一起使用后，保留第一份 0～26、第二份 27～82。报告区分请求范围、实际忽略项和范围内不存在的 ID。若第一份的标签实际用了 27 或 28，仍会报错，不会删除标注或把图片当负样本。
