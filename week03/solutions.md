# Week 3：正则化回归作业

本周按课件第 ⑨ 页完成作业 1、3、4；作业 2 不做。

## 作业 1：正交设计下 Ridge 解的推导

Ridge 的闭合解为

\[
\hat\beta_R=(X^TX+\lambda I)^{-1}X^Ty.
\]

当设计矩阵正交且已标准化时，\(X^TX=I\)。因此

\[
\begin{aligned}
\hat\beta_R
&=((1+\lambda)I)^{-1}X^Ty\\
&=\frac{1}{1+\lambda}X^Ty.
\end{aligned}
\]

同时正交设计下 \(\hat\beta_{OLS}=(X^TX)^{-1}X^Ty=X^Ty\)，所以

\[
\boxed{\hat\beta_R=\frac{1}{1+\lambda}\hat\beta_{OLS}}.
\]

这说明 Ridge 在正交情形下对所有系数进行相同倍数的收缩；当 \(\lambda>0\) 时，系数变小但不会因为 Ridge 惩罚而精确变成 0。

## 作业 3：为什么必须先 Z-score 标准化

Ridge 和 Lasso 惩罚的是系数的大小：\(\lambda\sum_j\beta_j^2\) 或 \(\lambda\sum_j|\beta_j|\)。如果特征量纲不同，系数大小就不能直接比较。例如收入用“元”表示时，系数可能天然很小；比例变量的系数可能相对较大。直接惩罚会使前者几乎不受罚、后者被过度压缩，正则化强度实际上被单位选择绑架。

因此应在训练集上对每个特征做

\[
x_{ij}^{*}=\frac{x_{ij}-\bar x_j}{s_j},
\]

并把同一个训练集的均值和标准差用于验证集、测试集。截距不应被惩罚，通常由 `Pipeline(StandardScaler(), model)` 配合 `fit_intercept=True` 实现。

不标准化的后果是：

1. 不同变量受到不公平的惩罚，模型系数和变量筛选对单位变化敏感；
2. Lasso 的“是否归零”不再反映变量贡献，而可能只是反映量纲；
3. Ridge 的收缩比例失去可比性，交叉验证选出的超参数也难以解释；
4. 若把截距一起中心化/惩罚，还会引入额外偏差。

## 作业 4：Hitters 上的 Ridge、Lasso、Elastic Net

运行 `python week03/regularized_regression.py`，脚本会：

- 从 ISLP 的 Hitters 数据读取并删除缺失值；
- 固定随机种子划分训练集和测试集；
- 在 Pipeline 中先标准化，再分别拟合 `RidgeCV`、`LassoCV`、`ElasticNetCV`，均使用 10 折 CV；
- 生成系数路径图；
- 报告测试集 RMSE、CV 选出的 alpha 和非零变量数；
- 用每个模型的 CV 最小误差加 1 个标准误的规则给出更稀疏候选模型。

运行后生成的 `results.csv` 和 `coefficient_paths.png` 是本题的数值结果与图形证据。由于交叉验证结果依赖 sklearn/数据版本，报告中的表格由脚本自动写出，避免手工抄录造成不一致。

### 1-SE 法则

设最小 CV 均方误差为 \(E_{min}\)，其标准误为 \(SE_{min}\)。1-SE 候选取满足

\[
E(\alpha)\le E_{min}+SE_{min}
\]

的模型中更强正则化、因而更稀疏的那个。它通常牺牲很小的验证误差，换来更少的变量和更好的可解释性。`Ridge` 本身通常没有精确为 0 的系数，因此“稀疏”主要体现在 Lasso 和 Elastic Net 上。
