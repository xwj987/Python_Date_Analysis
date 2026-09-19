# -*- coding: utf-8 -*-
"""
Week 2 作业 - 第 3 题：Python 实操题
使用 Carseats 数据集，以 Sales 为响应变量，
选取 Price、Income、Advertising 以及定性特征 ShelveLoc，
建立多元线性回归模型。

任务：
1. 提取模型拟合报告，指出 ShelveLoc 的基准组
2. 解读 ShelveLoc[Good] 系数的实际商业含义
3. 计算各变量的 VIF，评估是否存在多重共线性风险
"""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ============================================================
# 1. 加载 Carseats 数据集
# ============================================================
url = "https://raw.githubusercontent.com/intro-stat-learning/ISLP/main/ISLP/data/Carseats.csv"
df = pd.read_csv(url)

print("=" * 60)
print("数据集概览")
print("=" * 60)
print(df.head(), "\n")
print("ShelveLoc 的取值（定性特征的水平）：")
print(df["ShelveLoc"].value_counts(), "\n")

# ============================================================
# 2. 建立多元线性回归模型
#    C(ShelveLoc) 表示把 ShelveLoc 当作分类变量处理，
#    statsmodels 默认使用 Treatment 编码，按字母序取第一组为基准
# ============================================================
model = smf.ols("Sales ~ Price + Income + Advertising + C(ShelveLoc)", data=df)
result = model.fit()

print("=" * 60)
print("模型拟合报告")
print("=" * 60)
print(result.summary(), "\n")

# ============================================================
# 3. 回答问题：ShelveLoc 的基准组是什么？
#    Treatment 编码下，按字母序排序后的第一个水平为基准组
# ============================================================
levels = sorted(df["ShelveLoc"].unique())
print("=" * 60)
print("问题 1：ShelveLoc 的基准组")
print("=" * 60)
print(f"ShelveLoc 的水平（按字母序）：{levels}")
print(f"基准组 = 字母序第一个 = {levels[0]}\n")

# ============================================================
# 4. ShelveLoc[Good] 系数的商业含义
# ============================================================
coef_good = result.params["C(ShelveLoc)[T.Good]"]
print("=" * 60)
print("问题 2：ShelveLoc[Good] 系数的商业含义")
print("=" * 60)
print(f"ShelveLoc[Good] 系数 = {coef_good:.4f}")
print("含义：在 Price、Income、Advertising 相同的条件下，")
print(f"货架位置为 Good 的门店，其销售额平均比 Bad 高 {coef_good:.2f} 个单位（千美元），")
print("该效应在统计上高度显著。\n")

# ============================================================
# 5. 计算各定量变量的 VIF，评估多重共线性
#    VIF_j = 1 / (1 - R_j^2)，衡量第 j 个自变量被其余自变量解释的程度
#    经验法则：VIF > 10（或严格一些 > 5）提示存在严重多重共线性
# ============================================================
X = df[["Price", "Income", "Advertising"]].copy()
X = pd.concat([pd.Series(1.0, index=X.index, name="Intercept"), X], axis=1)

print("=" * 60)
print("问题 3：各变量 VIF 与多重共线性评估")
print("=" * 60)
for i, col in enumerate(X.columns):
    vif = variance_inflation_factor(X.values, i)
    print(f"VIF({col}) = {vif:.4f}")

print("\n结论：三个定量变量的 VIF 都接近 1（远小于 10），")
print("说明 Price、Income、Advertising 之间几乎没有线性相关，")
print("模型不存在明显的多重共线性风险。")
print("（ShelveLoc 为定性变量，用哑变量进入模型，通常不直接计算普通 VIF）")
