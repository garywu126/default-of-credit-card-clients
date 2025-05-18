# -*- coding: utf-8 -*-
"""
Optimized ANN for Credit Card Default Prediction
"""

# 1. 套件準備
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 2. 載入資料
os.chdir("C:\\Users\\88691\\OneDrive\\Desktop\\機器學習\\期中報告")
data = pd.read_excel('default of credit card clients.xls', header=1)

# 3. 移除 ID 欄位
data = data.drop(columns=['ID'])

# 4. 刪除 EDUCATION 異常值（0, 5, 6）
data = data[~data['EDUCATION'].isin([0, 5, 6])]

# 5. Label Encoding：EDUCATION（有順序）
le = LabelEncoder()
data['EDUCATION'] = le.fit_transform(data['EDUCATION'])

# 6. One-hot encoding：SEX, MARRIAGE（保留所有類別，適合 ANN）
data = pd.get_dummies(data, columns=['SEX', 'MARRIAGE'], drop_first=False)

# 7. 重命名目標欄位
data.rename(columns={'default payment next month': 'target'}, inplace=True)

# 8. 欠抽樣平衡資料（1:1）
class_0 = data[data['target'] == 0]
class_1 = data[data['target'] == 1]
class_0_under = class_0.sample(len(class_1), random_state=20250420)
data_balanced = pd.concat([class_0_under, class_1], axis=0).sample(frac=1, random_state=20250420)

# 9. 分離特徵與標籤
X = data_balanced.drop(columns=['target'])
y = data_balanced['target']

# 10. 資料切分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=20250420)

# 11. 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 12. 轉為 PyTorch Tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from itertools import product
import numpy as np

# 假設已有資料張量：X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
input_dim = X_train_tensor.shape[1]

# 激活函數字典
activation_dict = {
    "ReLU": nn.ReLU(),
    "Tanh": nn.Tanh(),
    "LeakyReLU": nn.LeakyReLU()
}

# Optimizer 工廠函數
def get_optimizer(name, params, lr, weight_decay=1e-3):
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported optimizer")

# 自訂 ANN 類別
class CustomANN(nn.Module):
    def __init__(self, input_dim, neurons, act1, act2):
        super().__init__()
        layers = [
            nn.Linear(input_dim, neurons[0]),
            act1,
            nn.Dropout(0.4),
            nn.Linear(neurons[0], neurons[1]),
            act2,
            nn.Dropout(0.4),
            nn.Linear(neurons[1], 1)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)  # 配合 BCEWithLogitsLoss，無 Sigmoid

# 超參數設定
batch_sizes = [32, 64]
learning_rates = [0.001, 0.005]  # 調整為更穩定範圍
activation_combos = [("ReLU", "ReLU"), ("ReLU", "LeakyReLU")]
optimizer_names = ["SGD", "Adam"]
neuron_configs = [(64, 32), (128, 64)]
epoch_list = [100, 150]
thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]

# 組合所有超參數
combinations = list(product(batch_sizes, learning_rates, activation_combos, optimizer_names, neuron_configs, epoch_list))
results = []
print(f"\n🚀 共 {len(combinations)} 組超參數組合要測試（請稍候...）\n")

# 訓練與測試
for i, (batch_size, lr, (act1_name, act2_name), opt_name, neurons, epochs) in enumerate(combinations):
    print(f"▶️ 測試 {i+1}/{len(combinations)}：BS={batch_size}, LR={lr}, Act=({act1_name}, {act2_name}), Opt={opt_name}, Neurons={neurons}, Epochs={epochs}")
    
    # 資料載入器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 建立模型
    model = CustomANN(
        input_dim,
        neurons,
        activation_dict[act1_name],
        activation_dict[act2_name]
    )
    pos_weight = torch.tensor([1.2])  # 適度偏向正類
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = get_optimizer(opt_name, model.parameters(), lr)

    # 提前停止
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    best_model_state = None

    torch.manual_seed(20250420)
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            y_pred = model(batch_X)
            loss = loss_fn(y_pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 驗證
        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_tensor)
            val_loss = loss_fn(val_pred, y_test_tensor)
            if epoch % 10 == 0:  # 每 10 epoch 輸出
                print(f"📌 [Epoch {epoch}] Train Loss: {loss.item():.4f} | Test Loss: {val_loss.item():.4f}")

        # 提前停止
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(best_model_state)
            break

    # 測試集評估
    model.eval()
    best_score = 0
    best_thresh_metrics = {}
    with torch.no_grad():
        test_pred = torch.sigmoid(model(X_test_tensor)).numpy()
        y_true = y_test_tensor.numpy()
        for thresh in thresholds:
            test_label = (test_pred > thresh).astype(int)
            acc = accuracy_score(y_true, test_label)
            recall = recall_score(y_true, test_label, zero_division=0)
            precision = precision_score(y_true, test_label, zero_division=0)
            f1 = f1_score(y_true, test_label, zero_division=0)
            cm = confusion_matrix(y_true, test_label)
            # 選擇兼顧高 Recall 和 Accuracy 的閾值（加權分數）
            score = 0.6 * recall + 0.4 * acc  # 偏向 Recall
            if score > best_score:
                best_score = score
                best_thresh_metrics = {
                    "Threshold": thresh,
                    "Accuracy": acc,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                    "Confusion Matrix": cm
                }

    print(f"✅ Best Threshold: {best_thresh_metrics['Threshold']}")
    print(f"Accuracy: {best_thresh_metrics['Accuracy']:.4f}, Precision: {best_thresh_metrics['Precision']:.4f}, "
          f"Recall: {best_thresh_metrics['Recall']:.4f}, F1: {best_thresh_metrics['F1']:.4f}")
    print(f"Confusion Matrix:\n{best_thresh_metrics['Confusion Matrix']}")
    results.append((
        batch_size, lr, act1_name, act2_name, opt_name, neurons, epochs,
        best_thresh_metrics['Accuracy'], best_thresh_metrics['Precision'],
        best_thresh_metrics['Recall'], best_thresh_metrics['F1']
    ))

# 顯示結果（依加權分數排序）
results.sort(key=lambda x: 0.6 * x[9] + 0.4 * x[7], reverse=True)  # 60% Recall + 40% Accuracy
print("\n🏆 測試結果總表（依加權分數排序：60% Recall + 40% Accuracy）：")
for r in results:
    print(f"BS={r[0]}, LR={r[1]}, Act=({r[2]}, {r[3]}), Opt={r[4]}, Neurons={r[5]}, Epochs={r[6]} → "
          f"Accuracy={r[7]:.4f}, Precision={r[8]:.4f}, Recall={r[9]:.4f}, F1={r[10]:.4f}")

# 保存最佳模型
torch.save(best_model_state, 'best_model.pth')