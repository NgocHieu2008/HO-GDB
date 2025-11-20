import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import random
import time

# ==========================================
# Bài toán giả định: Phân loại tính chất hóa học của nguyên tử.
# - Baseline: Chỉ biết khối lượng nguyên tử.
# - HO-GNN: Biết khối lượng + Loại phân tử mà nó thuộc về (thông tin từ Subgraph).

def generate_synthetic_data(num_samples=100):
    # Tạo dữ liệu mẫu
    # X_local: Feature của Node (ví dụ: Khối lượng)
    # X_global: Feature của Subgraph (ví dụ: Tính độc hại của phân tử)
    # Y: Nhãn cần dự đoán (phụ thuộc vào cả Node và Subgraph)
    
    data_list_baseline = []
    data_list_ho = []

    for _ in range(num_samples):
        # 1. Tạo đặc trưng Node (Local)
        num_nodes = 5
        x_local = torch.randn(num_nodes, 1)  # Feature ngẫu nhiên
        
        # 2. Tạo đặc trưng Subgraph (Global Context - HO)
        # Giả sử Subgraph có thuộc tính "toxicity" (0 hoặc 1)
        subgraph_feat = torch.randint(0, 2, (1, 1)).float()
        
        # 3. Tạo nhãn (Target)
        # Quy luật giả định: Y = 1 nếu (Node > 0) VÀ (Subgraph_Feat == 1)
        # Đây là mối quan hệ mà Baseline (không thấy Subgraph) sẽ rất khó học
        y = ((x_local > 0).float() * subgraph_feat).long().squeeze()

        # 4. Tạo cạnh (Edge Index giả lập chuỗi thẳng)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], 
                                   [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)

        # --- DATASET 1: BASELINE (Không có HO info) ---
        data_base = Data(x=x_local, edge_index=edge_index, y=y)
        data_list_baseline.append(data_base)

        # --- DATASET 2: HO-ENHANCED (Có thêm HO info) ---
        # "Lifting": Broadcast thông tin Subgraph xuống từng Node
        # Kỹ thuật này mô phỏng việc HO-GDB truyền message từ Subgraph -> Node
        x_ho = torch.cat([x_local, subgraph_feat.repeat(num_nodes, 1)], dim=1)
        
        data_ho = Data(x=x_ho, edge_index=edge_index, y=y)
        data_list_ho.append(data_ho)

    return data_list_baseline, data_list_ho

# ==========================================
# MODEL DEFINITIONS
# ==========================================
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# ==========================================
# TRAINING & EVALUATION LOOP
# ==========================================
def train_model(model, dataset, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in dataset:
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataset)
        loss_history.append(avg_loss)
        
        if epoch % 20 == 0:
            pass 

    return loss_history
# ==========================================
# MAIN EXPERIMENTAL SETUP   
# ==========================================

def main():
    print(">>> [SETUP] Generating Synthetic Dataset for HO Evaluation...")
    # Dataset giả lập mô tả tình huống: Dự đoán tính chất nguyên tử dựa trên cấu trúc phân tử
    dataset_baseline, dataset_ho = generate_synthetic_data(num_samples=200)
    
    # 1. Cấu hình 2 Model tương đương về kiến trúc
    # Baseline: Input dim = 1 (Chỉ có feature node)
    model_baseline = GNNModel(input_dim=1, hidden_dim=16, output_dim=2)
    
    # HO-GNN: Input dim = 2 (Feature node + Feature Subgraph lấy từ HO-GDB)
    model_ho = GNNModel(input_dim=2, hidden_dim=16, output_dim=2)

    print("\n>>> [EXPERIMENT] Training Baseline GNN (w/o HO features)...")
    start_base = time.time()
    losses_base = train_model(model_baseline, dataset_baseline, epochs=150)
    time_base = time.time() - start_base
    print(f"    Done in {time_base:.2f}s. Final Loss: {losses_base[-1]:.4f}")

    print("\n>>> [EXPERIMENT] Training HO-Enhanced GNN (w/ HO features)...")
    start_ho = time.time()
    losses_ho = train_model(model_ho, dataset_ho, epochs=150)
    time_ho = time.time() - start_ho
    print(f"    Done in {time_ho:.2f}s. Final Loss: {losses_ho[-1]:.4f}")

    # 2. Đánh giá kết quả (Analytical Effectiveness)
    improvement = (losses_base[-1] - losses_ho[-1]) / losses_base[-1] * 100
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    print(f"1. Baseline Final Loss : {losses_base[-1]:.4f}")
    print(f"2. HO-GNN Final Loss   : {losses_ho[-1]:.4f}")
    print(f"3. Accuracy Gain       : +{improvement:.2f}% improvement in Loss reduction")
    print("-" * 60)
    
    # 3. Vẽ biểu đồ so sánh (Lưu ra file để xem)
    plt.figure(figsize=(10, 6))
    plt.plot(losses_base, label='Baseline GNN (Standard)', color='orange', linestyle='--')
    plt.plot(losses_ho, label='HO-GNN (With Subgraph Context)', color='blue', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Analytical Effectiveness: HO-GNN vs Standard GNN')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = "ho_gnn_evaluation_chart.png"
    plt.savefig(filename)
    print(f"[VISUALIZATION] Comparison chart saved to '{filename}'")
    print("Check the chart to see faster convergence of HO-GNN.")

if __name__ == "__main__":
    main()