import os
import torch
import torch_geometric
from HOGDB.db.neo4j import Neo4jDatabase

# --- Import rõ ràng các lớp cần thiết ---
from HOGDB.graph.hypergraph_storage import HyperGraphStorage
from HOGDB.graph.node import Node
from HOGDB.graph.edge import Edge
from HOGDB.graph.hyperedge import HyperEdge
from HOGDB.graph.path import Path
from HOGDB.db.label import Label
from HOGDB.db.property import Property

def setup_data(gs: HyperGraphStorage):
    """
    Hàm tiện ích để tạo dữ liệu demo (bao gồm thuộc tính 'feat' và 'id' cho GNN).
    """
    print("Thiết lập dữ liệu demo...")
    gs.clear_graph()

    # --- Tạo Nodes với thuộc tính 'feat' VÀ 'id' ---
    # Thuộc tính 'id' là cần thiết cho 'sort' trong Listing 3
    richard = Node(labels=[Label("Person"), Label("Node")], properties=[Property("name", str, "Richard"), Property("id", int, 0), Property("feat", list, [0.1, 0.2])])
    mary = Node(labels=[Label("Person"), Label("Node")], properties=[Property("name", str, "Mary"), Property("id", int, 1), Property("feat", list, [0.3, 0.1])])
    bob = Node(labels=[Label("Person"), Label("Node")], properties=[Property("name", str, "Bob"), Property("id", int, 2), Property("feat", list, [0.4, 0.5])])
    ford = Node(labels=[Label("Company"), Label("Node")], properties=[Property("name", str, "Ford"), Property("id", int, 3), Property("feat", list, [1.0, 1.0])])
    car = Node(labels=[Label("Car"), Label("Node")], properties=[Property("model", str, "F-150"), Property("id", int, 4), Property("feat", list, [0.9, 0.8])])

    gs.add_node(richard)
    gs.add_node(mary)
    gs.add_node(bob)
    gs.add_node(ford)
    gs.add_node(car)
    print("-> Đã thêm 5 nút (với thuộc tính 'feat' và 'id').")

    # --- Tạo Edges với thuộc tính 'feat' ---
    company_car_edge = Edge(ford, car, Label("Builds"), [Property("feat", list, [0.5])])
    person_car_edge = Edge(richard, car, Label("Owns"), [Property("feat", list, [0.8])])
    marriage_edge = Edge(richard, mary, Label("Married"), [Property("feat", list, [1.0])])
    bob_car_edge = Edge(bob, car, Label("AfraidOf"), [Property("feat", list, [0.2])])
    mary_car_edge = Edge(mary, car, Label("Drives"), [Property("feat", list, [0.7])])

    # Gán nhãn 'Edge' chung cho GNN
    # BỊ XÓA: Vòng lặp for e in [...] đã bị xóa vì Edge không hỗ trợ nhiều labels
    # Thay vào đó, chúng ta sẽ thêm từng cạnh một
    gs.add_edge(company_car_edge)
    gs.add_edge(person_car_edge)
    gs.add_edge(marriage_edge)
    gs.add_edge(bob_car_edge)
    gs.add_edge(mary_car_edge)
    print("-> Đã thêm 5 cạnh (với thuộc tính 'feat').")

    # --- Tạo HyperEdges ---
    family_hyperedge = HyperEdge(
        nodes=[richard, mary, bob],
        label=Label("Family"),
        properties=[Property("domicile", str, "Texas"), Property("last_name", str, "Smith")],
    )
    work_team_hyperedge = HyperEdge(
        nodes=[richard, ford],
        label=Label("WorkTeam"),
        properties=[Property("project", str, "Mustang"), Property("active", bool, True)]
    )
    gs.add_hyperedge(family_hyperedge)
    gs.add_hyperedge(work_team_hyperedge)
    print("-> Đã thêm 2 siêu cạnh ('Family' và 'WorkTeam').")
    print("Thiết lập dữ liệu hoàn tất.")

def main() -> None:
    # Khởi tạo kết nối (Giả sử biến môi trường đã được cài đặt)
    db = Neo4jDatabase()

    gs= HyperGraphStorage(db)
    # Tạo dữ liệu demo trước
    setup_data(gs)

    # =================================================================
    print("\n--- PHẦN 2: THỰC THI OLAP (TÁC VỤ PHÂN TÍCH) ---")
    # =================================================================
    
    print("\n[OLAP] Bước 2.1: Duyệt đường đi Bậc cao (Demo Listing 2)...")
    print("... (Demo: tìm 'WorkTeam' project liên quan đến 'Richard') ...")
    
    try:
        # SỬA LỖI: Sử dụng cú pháp path.add(element, variable="...")
        path = Path()
        
        # Yếu tố 1: Một Node 'Person' tên là 'Richard'
        path.add(
            Node([Label("Person")], [Property("name", str, "Richard")]),
            variable="n1" # Đặt tên biến là 'n1'
        )
        
        # Yếu tố 2: Một HyperEdge 'WorkTeam'
        path.add(
            HyperEdge(label=Label("WorkTeam")),
            variable="h1" # Đặt tên biến là 'h1'
        )

        # Truy vấn này dựa vào tên 'h1'
        results = gs.traverse_path(
            [path], 
            return_values=["h1.project"], # Trả về thuộc tính của HyperEdge 'h1'
            sort=["h1.project"]
        )
        
        # Lấy kết quả từ DataFrame của Pandas
        project_name = results.iloc[0, 0]
        print(f"-> Kết quả (Tên dự án): {project_name}")
        assert project_name == "Mustang"
        print("-> THÀNH CÔNG: Listing 2 (Path Traversal) hoạt động.")

    except Exception as e:
        print(f"\n!!! GẶP LỖI KHI THỰC THI (Listing 2): {e}")


    print("\n[OLAP] Bước 2.2: Tích hợp GNN (Demo Listing 3)...")
    
    try:
        # --- 1. Lấy đặc trưng (features) của Nút ('x') ---
        print("-> 1. Lấy đặc trưng (features) của Nút ('x')...")
        path_x = Path()
        
        # SỬA LỖI: Cú pháp đúng
        path_x.add(Node([Label("Node")]), variable="x")
        
        # Sắp xếp theo thuộc tính 'id' mà chúng ta đã thêm trong setup_data
        node_features_df = gs.traverse_path(
            [path_x], 
            return_values=["x.feat"], 
            sort=["x.id"] # Sắp xếp theo 'id' để đảm bảo thứ tự
        )
        # Chuyển đổi cột 'x.feat' của DataFrame sang tensor
        x = torch.tensor(node_features_df['x.feat'].tolist())

        print(f"-> Đã lấy thành công tensor 'x' (Đặc trưng Nút): {x.shape}")

        # --- 2. Lấy chỉ mục Cạnh ('edge_index') ---
        print("-> 2. Lấy chỉ mục Cạnh ('edge_index')...")
        path_e = Path()
        
        # SỬA LỖI: Cú pháp đúng
        path_e.add(Node([Label("Node")]), variable="s") # Nút nguồn 's'
        # SỬA LỖI: Thay vì tìm "Edge", chúng ta tìm BẤT KỲ cạnh nào bằng cách
        # truyền label=None (hoặc bỏ trống, nhưng None rõ ràng hơn)
        path_e.add(Edge(label=None), variable="e") # Cạnh 'e'
        path_e.add(Node([Label("Node")]), variable="t") # Nút đích 't'
        
        # Lấy thuộc tính 'id' của nút nguồn (s) và nút đích (t)
        edge_index_df = gs.traverse_path(
            [path_e], 
            return_values=["s.id", "t.id"],
            sort=["s.id", "t.id"] # Sắp xếp để nhất quán (tùy chọn)
        )
        
        # Chuyển đổi DataFrame [n_edges, 2] -> Tensor [2, n_edges]
        edge_index = torch.tensor(edge_index_df.values).t().contiguous()
        
        print(f"-> Đã lấy thành công 'edge_index': {edge_index.shape}")

        # --- 3. Lấy đặc trưng (features) của Cạnh ('edge_attr') ---
        print("-> 3. Lấy đặc trưng (features) của Cạnh ('edge_attr')...")
        
        # Dùng lại path_e, chỉ thay đổi return_values
        edge_attr_df = gs.traverse_path(
            [path_e], 
            return_values=["e.feat"],
            sort=["s.id", "t.id"] # Sắp xếp giống như edge_index
        )
        edge_attr = torch.tensor(edge_attr_df['e.feat'].tolist())
        print(f"-> Đã lấy thành công 'edge_attr': {edge_attr.shape}")

        # --- 4. Tạo đối tượng dữ liệu GNN ---
        data = torch_geometric.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        print("\n-> THÀNH CÔNG: Đã tạo đối tượng torch_geometric.Data:")
        print(data)

    except Exception as e:
        print(f"\n!!! GẶP LỖI KHI THỰC THI (Listing 3): {e}")

    # Kết nối được tự động đóng bởi 'with'
    print("\nHoàn thành ví dụ demo OLAP HOGDB!")

if __name__ == "__main__":
    main()