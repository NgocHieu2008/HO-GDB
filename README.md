# HO-GDB
Team 15_Final Project
- Ngôn ngữ: Python (Python version >= 3.9) 
- Database Backend: Neo4j
- Các bước cài đặt:
1. Install package: pip install
2. Install Neo4j
3. Cấu hình .env
- Folder structure:
1. HOGDB/db/
+ db.py — cung cấp API chuẩn để HO-GDB thao tác DB 
+ neo4j.py — quản lý kết nối tới server Neo4j, mapping giữa kiểu dữ liệu HO (node-tuples, hyperedges, subgraph collections) và các lược đồ lưu trữ trong Neo4j.
+ label.py - Quản lý labels của node/edge trong graph.
+ property.py - Quản lý thuộc tính (property) của node / relationship.
+ schema.py - Quy định schema cho HO-GDB
2. HOGDB/graph/: cung cấp các API để thao tác higher-order graph ví dụ như add_node, add_subgraph,...
