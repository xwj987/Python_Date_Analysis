# Week 4：用 Elastic Net 分析 Kaggle Netflix 数据

## 研究问题

本作业使用 Kaggle 的 **Netflix Movies and TV Shows** 数据，研究一个明确的回归问题：根据影片的发行年份、加入 Netflix 的年份/月、分级、国家和类型标签，预测电影时长（分钟）。只保留 `type == Movie` 且 `duration` 能解析为分钟的记录；TV Show 的 `duration` 是季数，和电影分钟数不是同一指标，因此不混合建模。

## 方法

`elastic_net_netflix.py` 默认从 Kaggle 数据集 `shivamb/netflix-shows` 下载，也支持把从 Kaggle 下载的 `netflix_titles.csv` 作为命令行参数：

```bash
python week04/elastic_net_netflix.py
# 或
python week04/elastic_net_netflix.py /path/to/netflix_titles.csv
```

预处理只在训练集的 Pipeline 内拟合：数值变量用中位数填补并 Z-score 标准化；分类变量用众数填补、独热编码，低频水平合并。模型为 `ElasticNetCV`，10 折交叉验证同时选择 `alpha` 和 `l1_ratio`。测试集只用于最后一次评估。

## 结果文件

- `metrics.csv`：数据量、最优超参数、测试 RMSE/MAE/R² 和非零系数数量；
- `actual_vs_predicted.png`：测试集真实时长与预测时长散点图。

Elastic Net 的优势是结合 L1 的变量筛选和 L2 对相关特征的稳定收缩。Netflix 的 `country` 与 `listed_in` 具有相关/重叠信息，Elastic Net 比单纯 Lasso 更适合避免在相关特征中任意选择一个。系数和 R² 应结合预测目标理解：影片时长还受到剧情、制作类型等未观测因素影响，因此该模型主要用于展示正则化回归流程和变量贡献，不应解释为因果关系。
