# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 20:51:26 2025
@author: 吳振宇
"""

# 📌 匯入套件
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report

# 📌 設定工作路徑與讀取資料
os.chdir("C:\\Users\\88691\\OneDrive\\Desktop\\機器學習\\期中報告")
data = pd.read_excel('default of credit card clients.xls', header=1)

# 📌 前處理步驟
data.drop(columns=["ID"], inplace=True)  # 移除 ID 欄
data = data[~data['EDUCATION'].isin([0, 5, 6])]  # 刪除異常值

# Label Encoding
data['EDUCATION'] = LabelEncoder().fit_transform(data['EDUCATION'])

# One-hot encoding：SEX、MARRIAGE（避免共線性，drop_first=True）
data = pd.get_dummies(data, columns=["SEX", "MARRIAGE"], drop_first=True)

# 重新命名目標欄位
data.rename(columns={'default payment next month': 'target'}, inplace=True)

# 📌 欠抽樣（資料平衡處理）
class_0 = data[data['target'] == 0]
class_1 = data[data['target'] == 1]
class_0_under = class_0.sample(len(class_1), random_state=20250420)
data_balanced = pd.concat([class_0_under, class_1], axis=0).sample(frac=1, random_state=20250420)

# 📌 特徵與標籤分離
X = data_balanced.drop(columns=["target"])
y = data_balanced["target"]

# 📌 訓練與測試資料切分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=20250420)

# 📌 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 📌 加入常數項
X_train_sm = sm.add_constant(X_train_scaled)
X_test_sm = sm.add_constant(X_test_scaled)

# 📌 建立 Logit 模型（MLE 模擬 glm）
model_glm = sm.Logit(y_train, X_train_sm)
result = model_glm.fit(disp=0)  # 不顯示訓練過程

# 📌 顯示模型摘要（含係數、p 值等）
print(result.summary())

# 📌 預測（機率與分類）
y_prob = result.predict(X_test_sm)
y_pred = (y_prob > 0.5).astype(int)

# 📌 評估指標
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
conf_matrix = confusion_matrix(y_test, y_pred)

# 📌 印出評估結果
print(f"\n✅ GLM 模擬 (R glm) 結果：")
print(f"Accuracy   : {acc:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"ROC AUC    : {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)

print("\n📊 分類報告：")
print(classification_report(y_test, y_pred, digits=4))
