import os
import torch
import torch_geometric

from HOGDB.db.neo4j import Neo4jDatabase

# --- Cải tiến 1: Import rõ ràng các lớp cần thiết ---
from HOGDB.graph.hypergraph_storage import HyperGraphStorage
from HOGDB.graph.node import Node
from HOGDB.graph.edge import Edge
from HOGDB.graph.hyperedge import HyperEdge
# Giả định các lớp này tồn tại dựa trên báo cáo
from HOGDB.graph.path import Path
from HOGDB.db.label import Label
from HOGDB.db.property import Property

def main() -> None:
    # Khởi tạo kết nối
    db = Neo4jDatabase()
    gs = HyperGraphStorage(db)
    print("Đã kết nối và xóa đồ thị cũ...")
    gs.clear_graph()

    # =================================================================
    print("\n--- PHẦN 1: THỰC THI OLTP (TÁC VỤ GIAO DỊCH) ---")
    # =================================================================
    
    print("\n[OLTP] Bước 1.1: Tạo các Nodes")
    richard = Node(labels=[Label("Person")], properties=[Property("name", str, "Richard"), Property("born", int, 1999), Property("feat", list, [1.0, 0.1, 0.0])])
    mary = Node(labels=[Label("Person")], properties=[Property("name", str, "Mary"), Property("born", int, 2001), Property("feat", list, [1.0, 0.2, 0.0])])
    bob = Node(labels=[Label("Person")], properties=[Property("name", str, "Bob"), Property("born", int, 2024), Property("feat", list, [1.0, 0.3, 0.0])])
    ford = Node(labels=[Label("Company")], properties=[Property("name", str, "Ford"), Property("industry", str, "Cars"), Property("feat", list, [0.0, 0.0, 1.0])])
    car = Node(labels=[Label("Car")], properties=[Property("model", str, "F-150"), Property("power", str, "325hp"), Property("feat", list, [0.0, 1.0, 1.0])])

    gs.add_node(richard)
    gs.add_node(mary)
    gs.add_node(bob)
    gs.add_node(ford)
    gs.add_node(car)
    print("-> Đã thêm 5 nút (Person, Company, Car).")

    print("\n[OLTP] Bước 1.2: Tạo các Edges")
    company_car_edge = Edge(ford, car, Label("Builds"), [Property("since", int, 1975)])
    person_car_edge = Edge(richard, car, Label("Owns"), [Property("since", int, 2020)])
    marriage_edge = Edge(richard, mary, Label("Married"), [Property("since", int, 2023)])
    bob_car_edge = Edge(bob, car, Label("AfraidOf"), [Property("reason", str, "too loud")])

    gs.add_edge(company_car_edge)
    gs.add_edge(person_car_edge)
    gs.add_edge(marriage_edge)
    gs.add_edge(bob_car_edge)
    print("-> Đã thêm 4 cạnh (Builds, Owns, Married, AfraidOf).")

    print("\n[OLTP] Bước 1.3: Tạo HyperEdge (Bậc cao)...")
    # Đây là ví dụ 'Family'
    family_hyperedge = HyperEdge(
        nodes=[richard, mary, bob],
        label=Label("Family"),
        properties=[Property("domicile", str, "Texas"), Property("last_name", str, "Smith")],
    )
    gs.add_hyperedge(family_hyperedge) # API 'add_hyperedge' thực hiện 'Lowering'
    print("-> Đã thêm 1 siêu cạnh 'Family' (Richard, Mary, Bob).")

    print("\n[OLTP] Bước 1.4: Đọc (READ) cấu trúc bậc cao...") # <-- Đổi số thứ tự
    # Demo 'Lifting' (như trong báo cáo)
    family_pattern = HyperEdge(label=Label("Family"))
    
    # API 'get_hyperedge' thực hiện 'Lifting'
    retrieved_family = gs.get_hyperedge(family_pattern)
    
    if retrieved_family:
        print("-> Đọc (READ) siêu cạnh 'Family' thành công:")
        member_names = [node["name"] for node in retrieved_family.nodes]
        print(f"   Thành viên: {member_names}")
        assert "Richard" in member_names and "Bob" in member_names
        assert retrieved_family["domicile"] == "Texas" # Xác minh giá trị ban đầu
        print(f"   Nơi ở (ban đầu): {retrieved_family['domicile']}")
        print("   (Xác nhận các thành viên trong siêu cạnh là đúng)")

    # --- BƯỚC MỚI: CẬP NHẬT (UPDATE) ---
    print("\n[OLTP] Bước 1.5: Cập nhật (UPDATE) HyperEdge...")
    # Pattern để tìm 'Family' (có thể dùng lại family_pattern)
    update_pattern = HyperEdge(label=Label("Family"))
    
    # Thuộc tính mới: Cập nhật 'domicile', giữ 'last_name'
    # Lưu ý: update_hyperedge sẽ THAY THẾ toàn bộ thuộc tính
    new_properties = [
        Property("domicile", str, "Nevada"), # Giá trị mới
        Property("last_name", str, "Smith") # Phải bao gồm tất cả thuộc tính muốn giữ
    ]
    
    # Gọi API 'update_hyperedge' từ hypergraph_storage.py
    gs.update_hyperedge(update_pattern, new_properties)
    print(f"-> Đã cập nhật 'domicile' của 'Family' thành 'Nevada'.")

    # --- BƯỚC MỚI: ĐỌC LẠI ĐỂ XÁC MINH CẬP NHẬT ---
    print("\n[OLTP] Bước 1.6: Đọc lại (READ) để xác minh Cập nhật...")
    verified_family = gs.get_hyperedge(family_pattern)
    if verified_family:
        verified_domicile = verified_family["domicile"] # Dùng __getitem__
        print(f"   -> Đã xác minh 'domicile' (mới): {verified_domicile}")
        assert verified_domicile == "Nevada"
        assert verified_family["last_name"] == "Smith" # Đảm bảo thuộc tính kia còn
        print("   (Xác nhận Cập nhật thành công)")
    else:
        print("   (LỖI: Không tìm thấy Family sau khi cập nhật)")

    # --- BƯỚC MỚI: XÓA (DELETE) ---
    print("\n[OLTP] Bước 1.7: Xóa (DELETE) HyperEdge...")
    # Dùng lại family_pattern để tìm và xóa
    gs.delete_hyperedge(family_pattern)
    print("-> Đã xóa siêu cạnh 'Family'.")
    
    # --- BƯỚC MỚI: ĐỌC LẠI ĐỂ XÁC MINH XÓA ---
    print("\n[OLTP] Bước 1.8: Đọc lại (READ) để xác minh Xóa...")
    deleted_family = gs.get_hyperedge(family_pattern)
    hyperedge_count = gs.get_hyperedge_count([Label("Family")])
    
    assert deleted_family is None
    assert hyperedge_count == 0
    print("   (Xác nhận siêu cạnh đã bị xóa (count=0, get=None))")
    print("\nHoàn thành ví dụ demo HOGDB!")

if __name__ == "__main__":
    main()