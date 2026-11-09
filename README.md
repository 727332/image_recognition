# image_recognition
图像处理与分析大作业

# 手写数字平移增强鲁棒性实验

论文见 `论文.html`（可直接在浏览器打开并打印）或 `论文.md`。

## 实际结果

|方法|原图准确率|平移平均准确率|
|---|---:|---:|
|岭回归基线|92.12%|41.90%|
|水平增强|87.27%|54.01%|
|四方向增强|85.16%|69.14%|

均值来自随机种子17、29、43；本文报告鲁棒性提升及原图性能下降，不主张新算法或最佳性能。

## 运行

Python 3.13.12，NumPy 2.5.4。本包不需要 GPU。

```sh
python -m pip install -r requirements.txt
python train.py --out results
python verify.py
```

`results/*_17.npz` 等为各方法权重；`weights` 为65×10矩阵，最后一行是偏置。原图像素先除以16，再补常数1。`metrics.json` 为未经四舍五入的实验记录，`split_*.npz` 保存全部原始样本索引，`predictions_*.npz` 为逐样本预测。

## 代码与权重仓库

https://github.com/727332/image_recognition

完整模型权重及预测记录位于 results 目录。作者及单位应在提交前填写。数据来源及上游许可见 DATA_SOURCE.md 和 THIRD_PARTY_LICENSE.txt。清单 MANIFEST.sha256 用于检查文件完整性。
