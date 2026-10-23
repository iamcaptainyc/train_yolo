# 图片警告定位与修复

`repair_images.py` 递归扫描传入目录下所有支持的图片（JPEG、PNG、BMP、WebP、TIFF），使用 OpenCV 文件解码和 Pillow 完整加载检查。它能定位 `premature end of data segment`、`extraneous bytes before marker` 等原生 JPEG 解码警告，并重新编码仍可解码的 JPEG。

**先暂停训练，再运行修复。** 扫描时不要同时下载、替换或移动这些图片。

```bash
git pull --ff-only

# 扫描整个已合并数据集，包含 train/val/test
python repair_images.py yolo_datasets/camping_animals/images --workers 16

# 或一次扫描多个图片根目录（目录内递归，不需要逐个列出子目录）
python repair_images.py datasets/lila yolo_datasets/camping_animals/images --workers 16
```

通常只扫描训练实际使用的合并数据集即可。修复时写回同一 inode，所有硬链接（包括扫描目录外的链接）同步修复；同一 inode 的并发操作串行化。删除则只删除传入扫描目录内的路径及其对应标签，不会枚举整个磁盘寻找其他链接。需要清除多个数据集的坏图时同时传入这些根目录。

已安装本项目训练环境通常具备依赖。若缺少 cv2/Pillow/tqdm/filelock，可安装：

```bash
python -m pip install opencv-python Pillow tqdm filelock
```

无需为此升级现有训练环境的 OpenCV。Linux 使用持久进程池，默认 8 个 worker，可先用 16；过多 worker 可能使磁盘争用变慢。Windows 为可靠捕获原生解码器 stderr 使用额外子进程，速度较慢；此脚本主要面向 AutoDL Linux。

## 文件如何处理

- 正常图片不重写，不复制，不保存逐图校验记录。
- 有警告但仍能解码的 JPEG：重新编码到同目录临时文件，再验证解码无警告、宽高不变，然后写回原 inode，保留硬链接关系。
- 修复文件验证完成后才原位写入。每个正在修复的问题文件有临时 JPEG 和持久操作日志；写入成功后释放临时图片。强制结束时重跑同一命令可从暂存 JPEG 恢复，恢复前不要训练或删除操作日志/暂存文件。
- 成功修复的图片标签不改，不旋转，不缩放；保留原始像素坐标方向，移除 EXIF 元数据。该处理面向项目已按像素坐标对齐的 YOLO 图片，不适合依靠 EXIF 旋转显示的未整理数据。
- 重新编码是有损的（JPEG quality=95），只能修复文件结构，不能恢复缺失画面。所有修复记录均有 `review_content=true`，建议抽查修复图片。
- 完全无法解码的图片：自动删除图片以及 `images/.../x.jpg` 对应的 `labels/.../x.txt`，包括无法解码的 PNG 等格式。没有 `images` 目录层级的原始图片只删除图片，不猜测标签位置。仅检查模式不删除。权限/I/O 异常不会按坏图删除；非 JPEG 可解码但有警告仍记为失败。
- 自动清理受影响 YOLO 数据集的标签 `.cache`。发生删除时，读取现存图片路径和标签重算 `report.json`、已有的 `SUBSET_REPORT.md` / `MERGE_REPORT.md` / `AUDIT_REPORT.md` / `REPORT.md` 的当前数量、类别数量、各集合数量与比例。无需重复图片解码。类别 ID 不变；清理前详细报告保存在 `*.before_image_repair.*`，原始分组和转换审计明确作为历史记录。

## 输出和断点

默认输出到 `runs/image_repair/`：

```text
REPORT.md          # 中文汇总
issues.jsonl       # 仅问题图片：完整路径、警告、修复/失败状态
checkpoint.json    # 一个进度游标、累计计数和目录清单指纹
operations/        # 仅问题文件的修复/删除日志，支持写入和删除中断恢复
```

有进度条。Ctrl+C 后会等待已提交的少量图片完成，保存其问题记录，再退出；重跑同一命令从游标继续。不保存正常图片逐图缓存，不对图片内容计算哈希。初次定位未识别坏图仍必须完整读取解码全部图片，这无法通过文件名判断。

扫描顺序由排序后的路径决定，脚本自行删除的路径由问题日志补回扫描清单，删除后仍可断点继续；外部增删路径时拒绝混用旧断点。文件内容原地变化不由此游标检测；需要重新检查已扫描图片时使用 `--restart`。该选项会覆盖本次输出目录的旧报告和问题清单，需保留旧报告时换 `--output`。

强制杀进程/断电不是正常 Ctrl+C：先恢复未完成的问题操作，再从最近 64 张进度继续。未记账但已修好的图片可能计为正常；问题操作历史保存在 operations/。删除任务可恢复标签/图片只完成一半的情况。旧版断点需要使用新输出目录，或加 `--restart`。

只定位、不改图片：

```bash
python repair_images.py yolo_datasets/camping_animals/images \
  --check-only --output runs/image_check --workers 16
```

检查模式与修复模式的断点分开。默认修复命令不需要额外 `--fix` 参数。脚本不会保证修复后训练速度提升，磁盘、数据增强、CPU、worker 等也影响吞吐。
