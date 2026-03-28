import numpy as np
import torch
import numpy as np
import random
import os
base_path = "data/processed"

# === 加载邻接矩阵（固定） ===
adjacency_matrix = np.load(f"{base_path}/adjacency_matrix.npy")

# === 训练集 ===
gat_train = np.load(f"{base_path}/X_gat_train_scaled.npy")      # GAT 特征
y_train = np.load(f"{base_path}/y_train_scaled.npy")                         # 标签
A_train = np.tile(adjacency_matrix, (gat_train.shape[0], 1, 1))       # 邻接矩阵扩展

# === 验证集 ===
gat_val = np.load(f"{base_path}/X_gat_val_scaled.npy")          # GAT 特征
y_val = np.load(f"{base_path}/y_val_scaled.npy")                             # 标签
A_val = np.tile(adjacency_matrix, (gat_val.shape[0], 1, 1))           # 邻接矩阵扩展


# === 转为 tensor 并放到 GPU / CPU ===
gat_train = torch.tensor(gat_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
A_train = torch.tensor(A_train, dtype=torch.float32).to(device)

gat_val = torch.tensor(gat_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
A_val = torch.tensor(A_val, dtype=torch.float32).to(device)


print("✅ 数据加载完成：")
print(f"Train: {gat_train.shape}, {y_train.shape}")
print(f"Val:   {gat_val.shape}, {y_val.shape}")
print(f"邻接矩阵 shape: {adjacency_matrix.shape}")

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadGraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, dropout=0.4):
        super(MultiHeadGraphAttentionLayer, self).__init__()
        assert output_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.linears = nn.ModuleList([nn.Linear(input_dim, self.head_dim, bias=False) for _ in range(num_heads)])
        self.attn_fcs = nn.ModuleList([nn.Linear(2 * self.head_dim, 1, bias=False) for _ in range(num_heads)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, A):
        outputs = []
        for linear, attn_fc in zip(self.linears, self.attn_fcs):
            h = linear(X)
            N = h.shape[1]
            h_i = h.unsqueeze(2).repeat(1, 1, N, 1)
            h_j = h.unsqueeze(1).repeat(1, N, 1, 1)
            attn_input = torch.cat([h_i, h_j], dim=-1)
            e = F.leaky_relu(attn_fc(attn_input).squeeze(-1))
            attention = torch.where(A > 0, e, torch.full_like(e, float('-inf')))
            attention = F.softmax(attention, dim=-1)
            attention = self.dropout(attention)
            h_prime = torch.matmul(attention, h)
            outputs.append(h_prime)
        output = torch.cat(outputs, dim=-1)
        return output

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

class GATGRUModel(nn.Module):
    def __init__(self, node_num=8, feature_dim=5, gat_units=12, num_heads=6, gru_units=64, embedding_dim=64, dropout_rate=0.4):
        super(GATGRUModel, self).__init__()
        self.gat1 = MultiHeadGraphAttentionLayer(input_dim=feature_dim, output_dim=gat_units, num_heads=num_heads, dropout=dropout_rate)
        self.gat2 = MultiHeadGraphAttentionLayer(input_dim=gat_units, output_dim=gat_units, num_heads=num_heads, dropout=dropout_rate)
        self.residual_linear = nn.Linear(feature_dim, gat_units)
        self.norm1 = nn.LayerNorm([node_num, gat_units])
        self.norm2 = nn.LayerNorm([node_num, gat_units])
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=node_num * gat_units, hidden_size=gru_units, num_layers=3, dropout=dropout_rate, batch_first=True)
        self.gru_norm = nn.LayerNorm(gru_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.embedding_layer = nn.Sequential(
            nn.Linear(gru_units, embedding_dim),
            nn.ReLU()
        )
        self.prediction_layer = nn.Linear(embedding_dim, node_num)
        self.apply(init_weights)

    def forward(self, X_gat, A):
        gat_output = self.gat1(X_gat, A)
        gat_output = self.norm1(gat_output)
        gat_output = self.relu(gat_output)
        gat_output = self.gat2(gat_output, A)
        residual = self.residual_linear(X_gat)
        gat_output += residual
        gat_output = self.norm2(gat_output)
        gat_output = self.relu(gat_output)
        x = gat_output.reshape(gat_output.size(0), 1, -1)
        _, h = self.gru(x)
        h = self.dropout(h[-1])
        h = self.gru_norm(h)
        embedding = self.embedding_layer(h)
        prediction = self.prediction_layer(embedding)
        return prediction, embedding



# 构建并训练模型
model = GATGRUModel().to(device)  
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
model.train()
print(model) 

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


model = GATGRUModel().to(device)
optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

batch_size = 32
train_loss_history = []
val_loss_history = []
patience = 300
best_val_loss = float('inf')
counter = 0
os.makedirs("saved_model", exist_ok=True)

gat_train = gat_train.to(device)
A_train = A_train.to(device)
y_train = y_train.to(device)
gat_val = gat_val.to(device)
A_val = A_val.to(device)
y_val = y_val.to(device)


def hybrid_loss(pred, target):
    mse = nn.MSELoss()(pred, target)
    mae = nn.L1Loss()(pred, target)
    huber = nn.SmoothL1Loss()(pred, target)
    return 0.3 * mse + 0.3 * mae + 0.7 * huber

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(gat_train.size(0))
    epoch_loss = 0.0
    batch_count = 0

    for i in range(0, gat_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_gat = gat_train[indices]
        batch_adj = A_train[indices]
        batch_y = y_train[indices]

        optimizer.zero_grad(set_to_none=True)
        preds, _ = model(batch_gat, batch_adj)
        loss = hybrid_loss(preds, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    avg_train_loss = epoch_loss / batch_count
    train_loss_history.append(avg_train_loss)

    model.eval()
    with torch.no_grad():
        val_preds, _ = model(gat_val, A_val)
        val_loss = hybrid_loss(val_preds, y_val).item()
    val_loss_history.append(val_loss)

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "saved_model/gat_gru_model.pt")
        print(f" Saved Best Model at Epoch {epoch+1} with Val Loss {val_loss:.6f}")
    else:
        counter += 1
        if counter >= patience:
            print(f" Early stopping triggered at epoch {epoch+1}, no improvement for {patience} epochs.")
            break

np.savez("saved_model/gat_gru_history.npz", train_loss=train_loss_history, val_loss=val_loss_history)
