# 训练与调参的评估产物

train.py 训练结束（含 resume）后，重新加载 best.pt，对 val、test、train 分别运行无训练增强的评估。--eval-batch 默认 32，用于训练后的三次评估；--mode val 时仍使用 --batch。没有在 data.yaml 定义的分集，其结果 JSON 标记 skipped。

训练输出目录直接保存：

- train_result.json、val_result.json、test_result.json，包含 P、R、mAP50、mAP50-95、逐类指标、推理耗时和使用的权重路径。
- train.log、val.log、test.log，分别为训练集、验证集、测试集评估日志。此处 train.log 是训练集评估日志，训练过程仍由既有诊断日志和 Ultralytics results.csv 记录。
- train_confusion_matrix.json、val_confusion_matrix.json、test_confusion_matrix.json。
- evaluation/train、evaluation/val、evaluation/test 中的 Ultralytics 图表，包括原始及归一化混淆矩阵 PNG。

已有权重也能单独评估并导出：

```bash
python train.py --mode val --model weights/best.pt \
  --data yolo_datasets/your_data/data.yaml --split train \
  --imgsz 640 --batch 32 --device 0 --project runs/eval
```

将 --split 改为 val 或 test 可输出对应分集。结果 JSON、日志和混淆矩阵 JSON 保存在本次实际评估目录。

tune.py 的新评估会在每个 trial 下保存 val_confusion_matrix.json 和 test_confusion_matrix.json；原有 val_result.json、test_result.json、val.log、test.log 继续保存。val 阶段另做一次训练集评估，产物位于 evaluation/train/，其中 train_result.json、train.log、train_confusion_matrix.json 分别保存训练集评估结果、日志和矩阵。trial 根目录的 train_result.json 和 train.log 继续记录训练阶段。

训练集评估增加一次遍历训练图片的耗时。被 pruner 提前淘汰且没有完成独立评估的 trial，不生成这些矩阵。历史缓存的已成功评估不会自动补矩阵，可对已有 best.pt 使用上述 train.py 命令单独补齐。

矩阵 JSON 中 matrix 是未经裁剪或四舍五入的原始计数，normalized_matrix 按真实类别列归一化，与 Ultralytics 归一化图一致。行是预测类别，列是真实类别，labels 按矩阵顺序记录全部类别，最后一项 background。背景行代表漏检，背景列代表误检。矩阵使用当前安装的 Ultralytics 默认匹配规则，不把 NMS IoU 或 mAP IoU 当作矩阵的匹配阈值。

日志追加保存，失败会记录异常并将本次 result.json 标记为 failed，清除旧的本次分集矩阵，避免把旧结果误认为本次成功。
