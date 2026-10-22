# LILA 合并为 YOLO 并按相机/序列划分

需要自动解决父子类别混淆、原始类别全量统计与逐框审计时，请使用新的 [审计合并入口](lila_audited_merge.md)。新审计入口默认分组划分 90%/5%/5% 的 train/val/test。下文旧入口仅做名称合并，仍保留不同粒度类别和原默认 train/val；也可显式添加 `--val-fraction 0.05 --test-fraction 0.05` 生成三组。

`prepare_lila.py` 读取五个数据集的 `selection/selected_annotations.json`，统一类别、转换归一化检测框、生成可直接传给 `train.py` 的 `data.yaml`。不会使用筛选清单外的图片。

## 命令

```bash
git pull --ff-only
python -m pip install -r requirements-lila.txt PyYAML

# 图片未下完时，可先预览类别和分组；不读取图片，不生成训练 YAML
python prepare_lila.py --plan-only --output datasets/lila_yolo_plan

# 下载完成后正式转换，默认验证集目标比例 20%
python prepare_lila.py --source datasets/lila --output datasets/lila_yolo --val-fraction 0.2 --seed 42

# 训练依赖按照主 README 安装
python train.py --data datasets/lila_yolo/data.yaml --model weights/yolo11s.pt --epochs 100 --batch 8 --device 0
python train.py --data datasets/lila_yolo/data.yaml --model weights/yolo26s.pt --epochs 100 --batch 8 --device 0

# 旧入口默认仅划分 train/val；未指定 --test-fraction 时，评估使用 val
python train.py --mode val --split val --data datasets/lila_yolo/data.yaml --model runs/animals/yolo11s/weights/best.pt --device 0
```

支持 `--datasets` 仅选择其中部分目录名。输出必须为空目录，重跑使用新的 `--output`；规划目录和正式目录分开，防止旧图片/标签污染。

默认 `--copy-mode auto` 优先硬链接，不支持或跨盘时复制；需要独立副本时用 `--copy-mode copy`。硬链接共享文件内容，应将原图视为只读。输入/输出目录不能互相包含。输出文件名使用数据集名和原路径的 SHA-256 避免同名冲突。

## 合并类别

统一小写、空格/下划线/连字符和 grey/gray 拼写，再合并同名类别。明确别名包括 domestic dog → dog、domestic cat → cat、domestic cattle → cow、mountain lion → cougar、bat species → bat。按本次数据集语境，Caltech 的 opossum/raccoon 对应 virginia opossum/northern raccoon，Oregon 的 black bear 对应 american black bear。

`deer` 和具体鹿种、`squirrel` 和具体松鼠物种、`bird` 和具体鸟种等不同粒度保持区分，不能凭名称相似度判断同物种。不同粒度仍可能造成跨数据集监督不一致，训练前检查 `class_mapping.json`，按任务决定是否统一到物种或更粗分类。

`--class-map class_map.json` 可覆盖默认规则。键为标准化的原类别名称，或优先级更高的 `dataset:category_id`；值为目标名称。`null` 排除含该类的整张图片，不把未标物体变为空背景。例：

```json
{
  "bird": "bird",
  "other bird": "bird",
  "unidentified bird": "bird",
  "caltech-camera-traps:34": "deer"
}
```

只对实际存在框的类别生成连续 ID，按名称排序。明确空图生成空 `.txt`，不创建 empty 类。含 car/vehicle/human/ignore/problem/blurred 等排除类别的图片整张排除，即使同时有动物框也不保留。

## 防止相邻帧泄漏

- 优先按 location/camera 分组，同一数据集同一相机整体划分。
- 整个数据集没有相机字段、但每张图片都有序列 ID 时，按序列整体划分；此时只能证明序列隔离，不能证明相机隔离。
- 相机和序列使用连通分组，同一序列即使记录为不同相机，也不会拆开。
- 正式转换逐图计算 SHA-256，相同文件字节会连接不同相机甚至不同数据集的组，避免精确重复文件跨集。相同字节但标签冲突会失败，需人工核查。不同编码及近似重复图片不在哈希检查范围内。
- 缺少可靠分组信息的数据集整体放入训练集；ENA24 当前没有相机/序列字段，采用此规则。其余四个数据集的筛选元数据都有 location。不会根据数字文件名臆测相机或序列。

有额外相机表时可用 `--group-map cameras.json` 补充，键为 `dataset/相对图片路径`，值为相机 ID。例如 `{"ena24/1.jpg":"camera_A","ena24/10.jpg":"camera_A"}`。必须覆盖全部缺失记录，否则该数据集仍整体留在训练集。

按完整组近似各数据集的验证集目标比例，组大小、固定训练数据和重复文件会使实际比例偏离 20%。保持每个类别在训练集至少出现，不保证罕见类同时出现在验证集，缺席类别会记录到 report。不会为满足比例或类别覆盖拆组。输入内容、顺序和 seed 一致时可复现。原始官方 split 字段不沿用，这是新生成的合并训练/验证划分，不能作为官方测试成绩。

正式转换检查图片格式、EXIF 方向、实际尺寸与元数据是否一致，缺图或坏标注则失败，先补完下载。仅裁剪不超过 1 像素的标注越界并报告；其他无效框报错。无法产生独立组或 train/val 缺正样本时拒绝生成训练配置。

## 输出与验证

```text
datasets/lila_yolo/
  data.yaml
  images/train/<dataset>/<hash>.jpg
  images/val/<dataset>/<hash>.jpg
  labels/train/<dataset>/<hash>.txt
  labels/val/<dataset>/<hash>.txt
  class_mapping.json
  split_manifest.jsonl
  report.json
```

manifest 记录原路径、相机、序列、连通组、SHA-256、正负样本角色与 split。report 记录各 split 的图片数、类别框分布、负样本数、逐数据集数量、缺席类别和保守留在训练集的数据集。

`--plan-only` 未校验图片存在、格式、哈希或重复图，正式转换的划分可能因重复图连接变化。正式转换前不会写入训练 YAML；输出中断时也不能继续使用部分结果训练。

负样本沿用下载清单，不再抽样；整相机划分后两边比例不必各为 5%。旧清单中的车辆等进一步排除后，总比例也可能变化，以 report 为准。单元测试覆盖坐标、空标签、别名、缺图拒绝、规划复现、相机/序列/相同文件分组等行为。

本次验证：74 项测试通过；五个实际筛选标注的规划得到 193 类、2,693 个连通组，独立复核相机/序列跨集重叠为 0，全部类别保留训练样本。该规划使用已有筛选快照、没有全量图片哈希检查；正式转换后以新报告为准。未执行全量图片转换或 GPU 训练。

图片校验、哈希和 YOLO 写出支持 `--workers` 有界并发，默认 8，`--workers 1` 可恢复串行。审计入口还会并发独立核验输出图片和标签；不会跳过原有检查。
