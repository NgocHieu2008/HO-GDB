import sys
import random
import uuid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from tqdm import tqdm

# --- IMPORTS TỪ HỆ THỐNG HO-GDB ---
from HOGDB.graph.node import Node
from HOGDB.graph.edge import Edge
from HOGDB.graph.subgraph import Subgraph
from HOGDB.db.label import Label
from HOGDB.db.property import Property
from HOGDB.db.neo4j import Neo4jDatabase
from HOGDB.graph.graph_with_subgraph_storage import GraphwithSubgraphStorage

# ==========================================
# 1. DATA GENERATION
# ==========================================
def generate_data_into_db(db, num_samples=100):
    print(f">>> [STEP 1] Generating {num_samples} samples into Neo4j using HO-GDB API...")
    storage = GraphwithSubgraphStorage(db)
    storage.clear_graph()

    for i in tqdm(range(num_samples), desc="Persisting Data"):
        toxicity = random.randint(0, 1)
        
        nodes = []
        # Tạo và LƯU TỪNG NODE xuống DB trước
        for j in range(5): 
            feat_val = random.gauss(0, 1)
            ground_truth = 1 if (feat_val > 0 and toxicity == 1) else 0

            node_uid = str(uuid.uuid4())
            
            node_props = [
                Property("uid", str, node_uid), 
                Property("feat", float, feat_val),
                Property("label_y", int, ground_truth)
            ]
            
            node = Node([Label("Atom")], node_props)
            
            storage.add_node(node) 
            nodes.append(node)

        edges = []
        for k in range(4):
            edge = Edge(nodes[k], nodes[k+1], Label("BOND"), [])
            storage.add_edge(edge) 
            edges.append(edge)

        mol_subgraph = Subgraph(
            subgraph_nodes=nodes,
            subgraph_edges=edges,
            labels=[Label("Molecule")],
            properties=[
                Property("mol_id", str, f"MOL_{i}"),
                Property("toxicity", int, toxicity)
            ]
        )
        storage.add_subgraph(mol_subgraph)
    
    print("    [SUCCESS] Data ingestion complete via 'add_node' + 'add_subgraph'.")

# ==========================================
# 2. DATA EXTRACTION (Lifting Pipeline)
# ==========================================
def load_dataset_from_db(db):
    print("\n>>> [STEP 2] Extracting & Lifting Data for GNN...")
    
    query = """
    MATCH (m:Molecule)
    WITH m
    MATCH (m)<-[:_node_membership]-(n:Atom)
    RETURN 
        m.toxicity AS global_feat, 
        collect(n.feat) AS local_feats, 
        collect(n.label_y) AS labels
    """
    
    session = db.start_session()
    records = session.run(query)
    
    dataset_ho = []
    dataset_baseline = []
    
    count = 0
    for rec in records:
        count += 1
        global_feat = float(rec["global_feat"]) 
        local_feats = [float(x) for x in rec["local_feats"]]
        labels = [int(y) for y in rec["labels"]]
        
        num_nodes = len(local_feats)
        if num_nodes == 0: continue 

        # 2. Chuyển đổi sang Tensor
        x_local = torch.tensor(local_feats).view(-1, 1)
        y = torch.tensor(labels, dtype=torch.long)
        
        # Tái tạo cấu trúc cạnh (Linear chain)
        src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
        dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        # --- Baseline Data ---
        dataset_baseline.append(Data(x=x_local, edge_index=edge_index, y=y))

        # --- HO Data (Lifting) ---
        global_tensor = torch.tensor([global_feat]).repeat(num_nodes, 1)
        x_ho = torch.cat([x_local, global_tensor], dim=1)
        dataset_ho.append(Data(x=x_ho, edge_index=edge_index, y=y))

    db.end_session(session)
    
    if count == 0:
        raise RuntimeError("Không tìm thấy dữ liệu trong DB! Kiểm tra lại bước Ingestion.")
        
    print(f"    [SUCCESS] Loaded {count} samples from DB.")
    return dataset_baseline, dataset_ho

# ==========================================
# 3. MODEL & TRAINING
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
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train_model(model, dataset, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    losses = []

    if len(dataset) == 0:
        return [0.0] * epochs

    for _ in range(epochs):
        total_loss = 0
        for data in dataset:
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(dataset))
    return losses

# ==========================================
# MAIN
# ==========================================
def main():
    try:
        db = Neo4jDatabase()
    except Exception as e:
        print(f"Lỗi kết nối Neo4j: {e}")
        return

    generate_data_into_db(db, num_samples=100)

    try:
        ds_baseline, ds_ho = load_dataset_from_db(db)
    except RuntimeError as e:
        print(e)
        return
    print("\n>>> [STEP 3] Training & Comparing...")
    
    model_base = GNNModel(1, 16, 2)
    losses_base = train_model(model_base, ds_baseline)
    
    model_ho = GNNModel(2, 16, 2)
    losses_ho = train_model(model_ho, ds_ho)

    print("-" * 40)
    print(f"Final Loss (Baseline): {losses_base[-1]:.4f}")
    print(f"Final Loss (HO-GNN)  : {losses_ho[-1]:.4f}")
    print("-" * 40)
    
    if losses_ho[-1] < losses_base[-1]:
        print("CONCLUSION: HO-GDB integration successfully improved accuracy!")

    db.close_driver()

if __name__ == "__main__":
    main()