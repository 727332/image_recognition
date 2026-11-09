手写数字平移增强鲁棒性实验
仓库：https://github.com/727332/image_recognition

本工程仅包含实验所需代码、数据、权重和纯文本说明，不包含论文或 Markdown 文件。

一、环境
Python 3.13.12，NumPy 2.5.4，无需 GPU。
在工程文件夹打开 PowerShell，逐行执行：
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

二、重新训练（不覆盖论文对应结果）
.\.venv\Scripts\python.exe train.py --out reproduced_results

三、核验新结果
.\.venv\Scripts\python.exe verify.py --results reproduced_results

四、核验随附的论文结果
.\.venv\Scripts\python.exe verify.py --results results

五、文件说明
train.py：数据划分、增强、调参、训练及结果保存。
verify.py：验证权重、预测、指标和划分隔离。
digits.csv.gz：实验数据，必须保留在 train.py 同目录。
requirements.txt：依赖版本。
results：论文对应的九组权重、数据划分、预测与指标。
DATA_SOURCE.txt 和 THIRD_PARTY_LICENSE.txt：数据来源和上游许可，应保留。

论文、论文生成脚本、图表、Markdown 文档以及论文压缩包不参与训练，移出工程不影响结果。
如需对照论文，不要覆盖 results；自行训练输出至 reproduced_results。
重新上传前请不要加入 .venv、__pycache__ 或本机临时文件。
