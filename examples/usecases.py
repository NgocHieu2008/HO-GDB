import sys
from typing import List

from HOGDB.graph.node import Node
from HOGDB.graph.edge import Edge
from HOGDB.graph.node_tuple import NodeTuple
from HOGDB.graph.subgraph import Subgraph, SubgraphEdge 
from HOGDB.db.label import Label
from HOGDB.db.property import Property
from HOGDB.db.neo4j import Neo4jDatabase

from HOGDB.graph.graph_with_subgraph_storage import GraphwithSubgraphStorage
from HOGDB.graph.graph_with_tuple_storage import GraphwithTupleStorage

def print_header(title):
    print("\n" + "="*80)
    print(f"COMPLEX SCENARIO: {title}")
    print("="*80)

# ==========================================
# SCENARIO 1: INTERCONNECTED INCIDENTS
# (Shared Nodes & Subgraph-to-Subgraph Edges)
# ==========================================
def run_complex_incidents(db):
    print_header("3.1 Complex Manufacturing Incidents (Shared Resources)")
    sg_storage = GraphwithSubgraphStorage(db)

    # --- BƯỚC 1: Tạo các Tài nguyên dùng chung (Shared Assets) ---
    # Máy bơm này sẽ tham gia vào cả 2 sự cố
    shared_pump = Node([Label("Machine")], [Property("id", str, "Pump_X5"), Property("model", str, "Hydraulic_2024")])
    # Cấu hình hệ thống dùng chung
    shared_config = Node([Label("Config")], [Property("ver", str, "v3.5_Legacy")])
    
    sg_storage.add_node(shared_pump)
    sg_storage.add_node(shared_config)

    assert sg_storage.get_node_count([Label("Machine")]) >= 1
    assert sg_storage.get_node_count([Label("Config")]) >= 1

    print(f">>> [LIFTING] Created Shared Assets: Pump_X5 & Config v3.5")

    # --- BƯỚC 2: Tạo Sự cố 1 (Incident A - Quá nhiệt) ---
    batch_a = Node([Label("Batch")], [Property("id", str, "B8_OVERHEAT")])
    # Edge nội bộ: Batch A chạy trên Pump X5
    edge_a = Edge(batch_a, shared_pump, Label("RUN_ON"), [Property("duration", str, "2h")])
    
    sg_storage.add_node(batch_a)
    sg_storage.add_edge(edge_a)

    assert sg_storage.get_node_count([Label("Batch")]) >= 1
    assert sg_storage.get_edge_count(Label("RUN_ON")) >= 1

    incident_a = Subgraph(
        subgraph_nodes=[batch_a, shared_pump, shared_config],
        subgraph_edges=[edge_a],
        labels=[Label("QualityIncident"), Label("Overheat")],
        properties=[Property("incident_id", str, "INC-001"), Property("severity", str, "High")]
    )

    # --- BƯỚC 3: Tạo Sự cố 2 (Incident B - Rung lắc) ---
    # Lưu ý: Sự cố này KHÁC Batch, nhưng DÙNG CHUNG Pump và Config với sự cố A
    batch_b = Node([Label("Batch")], [Property("id", str, "B9_VIBRATION")])
    edge_b = Edge(batch_b, shared_pump, Label("RUN_ON"), [Property("duration", str, "5h")])

    sg_storage.add_node(batch_b)
    sg_storage.add_edge(edge_b)
    assert sg_storage.get_node_count([Label("Batch")]) >= 2
    assert sg_storage.get_edge_count(Label("RUN_ON")) >= 2

    incident_b = Subgraph(
        subgraph_nodes=[batch_b, shared_pump, shared_config], # Tái sử dụng node shared
        subgraph_edges=[edge_b],
        labels=[Label("QualityIncident"), Label("Vibration")],
        properties=[Property("incident_id", str, "INC-002"), Property("severity", str, "Medium")]
    )

    # --- BƯỚC 4: Tạo mối quan hệ giữa 2 Sự cố (SubgraphEdge) ---
    sim_link = SubgraphEdge(
        start_subgraph=incident_a,
        end_subgraph=incident_b,
        label=Label("SIMILAR_TO"),
        properties=[Property("reason", str, "Same Machine Failure"), Property("confidence", float, 0.85)]
    )

    # --- BƯỚC 5: Lưu xuống DB ---
    print(">>> [LOWERING] Persisting Incident A, Incident B, and their Link...")
    sg_storage.add_subgraph(incident_a)
    sg_storage.add_subgraph(incident_b)
    sg_storage.add_subgraph_edge(sim_link) # Lưu cạnh nối giữa 2 subgraph
    print("    [SUCCESS] Complex Incident Graph Created.")


# ==========================================
# SCENARIO 2: BRANCHING CAUSAL CHAINS
# (Overlapping Tuples)
# ==========================================
def run_complex_causality(db):
    print_header("3.2 Branching Causal Patterns (Overlapping Events)")
    tuple_storage = GraphwithTupleStorage(db)

    # Tạo các sự kiện đơn lẻ
    e_install = Node([Label("Event")], [Property("desc", str, "Sensor Install")])
    e_calib = Node([Label("Event")], [Property("desc", str, "Calibration")])
    
    # Sự kiện trung gian quan trọng (Pivot Event)
    e_warning = Node([Label("Event"), Label("Alert")], [Property("desc", str, "Pressure Warning")])
    
    e_shutdown = Node([Label("Event")], [Property("desc", str, "Emergency Shutdown")])
    e_restart = Node([Label("Event")], [Property("desc", str, "System Restart")])

    for n in [e_install, e_calib, e_warning, e_shutdown, e_restart]:
        tuple_storage.add_node(n)
    print(f">>> [LIFTING] Created Individual Events including Pivot Event 'Pressure Warning'")

    # --- Pattern 1: Nguyên nhân (Operational Chain) ---
    # Install -> Calibration -> Warning
    tuple_op = NodeTuple(
        nodes=[e_install, e_calib, e_warning],
        labels=[Label("CausalPattern"), Label("Operational")],
        properties=[Property("type", str, "RootCause")]
    )
    
    # --- Pattern 2: Hậu quả (Recovery Chain) ---
    # Warning -> Shutdown -> Restart
    # Pattern này chia sẻ sự kiện 'e_warning' với Pattern 1
    tuple_rec = NodeTuple(
        nodes=[e_warning, e_shutdown, e_restart],
        labels=[Label("CausalPattern"), Label("Recovery")],
        properties=[Property("type", str, "Response")]
    )

    print(">>> [LOWERING] Persisting Overlapping Causal Tuples...")
    tuple_storage.add_node_tuple(tuple_op)
    tuple_storage.add_node_tuple(tuple_rec)
    print("    [SUCCESS] Branching Patterns Created.")

# ==========================================
# MAIN
# ==========================================
def main():
    print(">>> [SETUP] Connecting to Neo4j...")
    try:
        db = Neo4jDatabase()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    print(">>> [SETUP] Clearing Graph...")
    temp = GraphwithSubgraphStorage(db)
    temp.clear_graph()

    run_complex_incidents(db)
    run_complex_causality(db)

    db.close_driver()
    print("\n" + "="*80)
    print("DONE. Use the provided Cypher queries to inspect the complex graph.")
    print("="*80)

if __name__ == "__main__":
    main()


# # ==========================================
# INSPECTION QUERIES (CYPHER)
# # ==========================================
# Case 1: Interconnected Incidents
# // Tìm node đại diện cho sự cố (ví dụ lấy sự cố đầu tiên tìm thấy hoặc lọc theo ID)
# MATCH (incident:QualityIncident)

# // Tìm tất cả các node con được kết nối qua quan hệ thành viên
# MATCH (incident)-[r]-(component)

# // Trả về node trung tâm, mối quan hệ và các thành phần
# RETURN incident, r, component

# Truy vết ngữ cảnh sự cố
# MATCH (incident:QualityIncident {incident_id: "INC-001"})
# MATCH (incident)<-[:_node_membership]-(component)

# RETURN 
#     incident.severity AS Severity,
#     labels(component) AS Types,
#     coalesce(component.id, component.ver) AS ID_or_Ver,
#     properties(component) AS All_Details

# Tìm kiếm sự cố tương tự
# // Tìm sự cố gốc
# MATCH (current_incident:QualityIncident {incident_id: "INC-001"})


# // Tìm các sự cố khác được nối bằng cạnh _subgraph_edge (đại diện cho similarity)
# MATCH (current_incident)-[:_subgraph_adjacency]-(link:_subgraph_edge)-[:_subgraph_adjacency]-(past_incident:QualityIncident)


# RETURN 
#     past_incident.incident_id AS Similar_Incident_ID,
#     past_incident.incident_date AS Date,
#     link.confidence AS Similarity_Score
# ORDER BY link.confidence DESC

# Case 2: Branching Causal Chains
# View causal pattern with events
#  // Tìm node đại diện cho mẫu hình nhân quả
# MATCH (pattern:CausalPattern)

# // Tìm các sự kiện thuộc về mẫu hình này
# MATCH (pattern)-[r]-(event)

# RETURN pattern, r, event

# Tái hiện chuỗi sự kiện

# // 1. Tìm mẫu hình có type là "RootCause" (đây là dữ liệu thật do code tạo ra)
# MATCH (pattern:CausalPattern {type: "RootCause"})

# // 2. Lấy các sự kiện thành viên
# MATCH (pattern)<-[r:_node_membership]-(event:Event)

# // 3. Trả về các thuộc tính ĐANG CÓ trong database
# RETURN 
#     pattern.type AS Pattern_Type,      // Thay cho impact
#     r.position_in_tuple AS Step_Order,
#     event.desc AS Event_Description    // Thay cho timestamp (vì không có time)
# ORDER BY r.position_in_tuple ASC

