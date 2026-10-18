# -*- coding: utf-8 -*-
"""
Week 5 作业：使用 SVM 对 Spambase（垃圾邮件）数据集做分类

数据来源：UCI Machine Learning Repository - Spambase
    https://archive.ics.uci.edu/dataset/94/spambase
数据规模：4601 封邮件，57 个特征（48 个词频 + 6 个字符频率 + 3 个大写字母连续长度），标签 0/1
    —— 即课件「案例①：垃圾邮件分类」所描述的数据集

流程（对应课件实践建议）：
    1. 加载数据并划分训练/测试集
    2. 特征标准化（SVM 对特征尺度敏感，必须先做）
    3. 线性核 SVM 作为 baseline
    4. RBF 核 SVM + GridSearchCV 网格搜索 C 与 gamma
    5. 与逻辑回归（Logistic Regression）对比
    6. 输出准确率 / 精确率 / 召回率 / F1 / AUC，绘制混淆矩阵与 ROC 曲线
"""

import os

# Windows 用户名含中文时，joblib/loky 的默认临时目录（C:\Users\<中文名>\...）
# 会导致 ASCII 编码报错，这里把多进程临时目录指到纯 ASCII 路径
os.environ.setdefault("JOBLIB_TEMP_FOLDER", r"D:\2026python\.tmp_joblib")

import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 中文字体设置（Windows）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

RANDOM_STATE = 42

# ============================================================
# 1. 加载 Spambase 数据集
# ============================================================
# 57 个特征的构成：48 个词频 + 6 个字符频率 + 3 个大写字母统计量
column_names = (
    [f"word_freq_{i:02d}" for i in range(48)]
    + [f"char_freq_{i:02d}" for i in range(6)]
    + ["capital_run_length_average", "capital_run_length_longest", "capital_run_length_total"]
    + ["label"]
)

df = pd.read_csv("data/spambase.data", header=None, names=column_names)

print("=" * 62)
print("① 数据集概览")
print("=" * 62)
print(f"样本数：{df.shape[0]}，特征数：{df.shape[1] - 1}")
print(f"类别分布：非垃圾邮件(0) = {(df['label'] == 0).sum()}，"
      f"垃圾邮件(1) = {(df['label'] == 1).sum()} "
      f"（垃圾邮件占比 {df['label'].mean():.1%}）")

X = df.drop(columns="label").values
y = df["label"].values

# ============================================================
# 2. 划分训练/测试集 + 标准化
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # 只在训练集上 fit，防止数据泄漏
X_test_s = scaler.transform(X_test)

print(f"\n训练集：{X_train_s.shape[0]} 条，测试集：{X_test_s.shape[0]} 条（分层抽样保持类别比例）")

# ============================================================
# 3. 训练三个模型（对应课件案例①的对比表）
# ============================================================
results = {}

# --- 3.1 线性核 SVM（baseline）---
t0 = time.time()
svm_linear = SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
svm_linear.fit(X_train_s, y_train)
t_linear = time.time() - t0
results["线性核 SVM (C=1)"] = {"model": svm_linear, "time": t_linear}

# --- 3.2 RBF 核 SVM + 网格搜索（课件建议：C、gamma 对数网格搜索）---
print("\n正在进行 RBF 核网格搜索（5 折交叉验证）...")
t0 = time.time()
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [0.0001, 0.001, 0.01, 0.1],
}
grid = GridSearchCV(
    SVC(kernel="rbf", random_state=RANDOM_STATE),
    param_grid, cv=5, scoring="accuracy", n_jobs=-1
)
grid.fit(X_train_s, y_train)
t_rbf = time.time() - t0
print(f"网格搜索完成，耗时 {t_rbf:.1f} s")
print(f"最优参数：C = {grid.best_params_['C']}，gamma = {grid.best_params_['gamma']}")
print(f"交叉验证最优准确率：{grid.best_score_:.4f}")
results[f"RBF 核 SVM (C={grid.best_params_['C']}, γ={grid.best_params_['gamma']})"] = {
    "model": grid.best_estimator_, "time": t_rbf,
}

# --- 3.3 逻辑回归（对照组，对应课件对比表中的 Logistic Regression）---
t0 = time.time()
logreg = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
logreg.fit(X_train_s, y_train)
t_log = time.time() - t0
results["逻辑回归"] = {"model": logreg, "time": t_log}

# ============================================================
# 4. 在测试集上评估
# ============================================================
print("\n" + "=" * 62)
print("② 测试集性能对比")
print("=" * 62)
header = f"{'模型':<32}{'准确率':>8}{'精确率':>8}{'召回率':>8}{'F1':>8}{'AUC':>8}{'耗时(s)':>9}"
print(header)
print("-" * len(header))

roc_curves = {}
for name, info in results.items():
    m = info["model"]
    t0 = time.time()
    y_pred = m.predict(X_test_s)
    infer_t = time.time() - t0
    y_score = m.decision_function(X_test_s) if hasattr(m, "decision_function") \
        else m.predict_proba(X_test_s)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_curves[name] = (fpr, tpr)

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    info["metrics"] = dict(acc=acc, pre=pre, rec=rec, f1=f1, auc=auc)
    print(f"{name:<32}{acc:>8.4f}{pre:>8.4f}{rec:>8.4f}{f1:>8.4f}{auc:>8.4f}"
          f"{info['time']:>9.2f}")

# 支持向量数量（体现课件讲的稀疏性）
print("\n" + "=" * 62)
print("③ 支持向量与稀疏性")
print("=" * 62)
for name, info in results.items():
    m = info["model"]
    if isinstance(m, SVC):
        n_sv = m.n_support_.sum()
        print(f"{name}：支持向量 {n_sv} 个 / 训练样本 {X_train_s.shape[0]} 个"
              f"（占比 {n_sv / X_train_s.shape[0]:.1%}）")
    else:
        print(f"{name}：非 SVM 模型，无支持向量概念")

# ============================================================
# 5. 可视化：混淆矩阵（RBF 最优模型）+ ROC 曲线对比
# ============================================================
import os
os.makedirs("figures", exist_ok=True)

best_name, best_info = max(
    ((n, i) for n, i in results.items() if isinstance(i["model"], SVC)),
    key=lambda kv: kv[1]["metrics"]["f1"],
)
best_pred = best_info["model"].predict(X_test_s)
cm = confusion_matrix(y_test, best_pred)

fig, ax = plt.subplots(figsize=(5.2, 4.2))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1], ["非垃圾邮件", "垃圾邮件"])
ax.set_yticks([0, 1], ["非垃圾邮件", "垃圾邮件"])
ax.set_xlabel("预测类别")
ax.set_ylabel("真实类别")
ax.set_title(f"混淆矩阵：{best_name}")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig("figures/confusion_matrix.png", dpi=150)

fig, ax = plt.subplots(figsize=(5.2, 4.2))
for name, (fpr, tpr) in roc_curves.items():
    auc = results[name]["metrics"]["auc"]
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="随机猜测")
ax.set_xlabel("假正率 FPR")
ax.set_ylabel("真正率 TPR")
ax.set_title("ROC 曲线对比")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig("figures/roc_curve.png", dpi=150)

print("\n已保存图表：figures/confusion_matrix.png，figures/roc_curve.png")
