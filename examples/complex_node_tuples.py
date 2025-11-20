# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

from HOGDB.db.neo4j import Neo4jDatabase
from HOGDB.graph.graph_with_tuple_storage import *


# --- Helpers ---------------------------------------------------------------

def has_label_soft(nt, expected: str) -> bool:
    try:
        labels = getattr(nt, "labels", []) or []
        for l in labels:
            for attr in ("name", "label", "value"):
                v = getattr(l, attr, None)
                if isinstance(v, str) and v.lower() == expected.lower():
                    return True
            # fallback: so sánh string hóa
            if str(l).strip().lower() == expected.lower():
                return True
        return False
    except Exception:
        return False


def prop_dict(obj) -> dict:
    """Chuyển list Property(...) thành dict {key: value} một cách an toàn."""
    d = {}
    try:
        for p in getattr(obj, "properties", []) or []:
            k = getattr(p, "key", None)
            v = getattr(p, "value", None)
            if k is not None:
                d[k] = v
    except Exception:
        pass
    return d


# --- Main example ----------------------------------------------------------

def main() -> None:
    """
    Complex example với nhiều higher-order node-tuples:
    - 5 Person nodes (Alice, Bob, Carol, Dave, Erin)
    - 2 Team tuples    (TeamA = {Alice, Bob, Carol}, TeamB = {Dave, Erin})
    - 2 Project tuples (ProjX = {Alice, Dave}, ProjY = {Bob, Carol, Erin})
    - 1 Year tuple     (Y2024 = {Alice, Bob, Carol})
    """
    # 1) Khởi tạo
    db = Neo4jDatabase()
    gs = GraphwithTupleStorage(db)
    gs.clear_graph()

    # 3) Thêm node Person
    alice = Node(
        [Label("Person")],
        [Property("name", str, "Alice"), Property("role", str, "Engineer"), Property("level", str, "SWE II")],
    )
    bob = Node(
        [Label("Person")],
        [Property("name", str, "Bob"), Property("role", str, "Designer"), Property("level", str, "Senior")],
    )
    carol = Node(
        [Label("Person")],
        [Property("name", str, "Carol"), Property("role", str, "Manager"), Property("level", str, "M1")],
    )
    dave = Node(
        [Label("Person")],
        [Property("name", str, "Dave"), Property("role", str, "Engineer"), Property("level", str, "SWE I")],
    )
    erin = Node(
        [Label("Person")],
        [Property("name", str, "Erin"), Property("role", str, "Data Scientist"), Property("level", str, "L3")],
    )
    for n in (alice, bob, carol, dave, erin):
        gs.add_node(n)

    # Kiểm tra số node Person
    assert gs.get_node_count([Label("Person")]) == 5

    # 4) Tạo các node-tuple
    # 4.1 Team tuples
    team_a = NodeTuple(
        [alice, bob, carol],
        [Label("Team")],
        [Property("id", str, "teamA"), Property("name", str, "Team A"), Property("domain", str, "Product")],
    )
    team_b = NodeTuple(
        [dave, erin],
        [Label("Team")],
        [Property("id", str, "teamB"), Property("name", str, "Team B"), Property("domain", str, "Platform")],
    )
    gs.add_node_tuple(team_a)
    gs.add_node_tuple(team_b)

    # 4.2 Project tuples
    proj_x = NodeTuple(
        [alice, dave],
        [Label("Project")],
        [Property("code", str, "ProjX"), Property("year", int, 2024)],
    )
    proj_y = NodeTuple(
        [bob, carol, erin],
        [Label("Project")],
        [Property("code", str, "ProjY"), Property("year", int, 2024)],
    )
    gs.add_node_tuple(proj_x)
    gs.add_node_tuple(proj_y)

    # 4.3 Year tuple
    year_2024 = NodeTuple(
        [alice, bob, carol],
        [Label("Year")],
        [Property("y", int, 2024)],
    )
    gs.add_node_tuple(year_2024)

    # Kiểm tra tổng số node-tuple
    assert gs.get_node_tuple_count() == 5  # TeamA, TeamB, ProjX, ProjY, Year2024

    # 5) Truy xuất theo thuộc tính khóa (ổn định)
    # TeamA
    retrieved_team_a = gs.get_node_tuple(
        NodeTuple([], [Label("Team")], [Property("id", str, "teamA")])
    )
    assert retrieved_team_a is not None
    assert prop_dict(retrieved_team_a).get("id") == "teamA"

    # Project Y
    retrieved_proj_y = gs.get_node_tuple(
        NodeTuple([], [Label("Project")], [Property("code", str, "ProjY")])
    )
    assert retrieved_proj_y is not None
    assert prop_dict(retrieved_proj_y).get("code") == "ProjY"

    # Year 2024
    retrieved_year_2024 = gs.get_node_tuple(
        NodeTuple([], [Label("Year")], [Property("y", int, 2024)])
    )
    assert retrieved_year_2024 is not None
    assert prop_dict(retrieved_year_2024).get("y") == 2024

    # 6) Cập nhật một tuple: ProjX -> 2025 + thêm desc
    gs.update_node_tuple(
        NodeTuple([], [Label("Project")], [Property("code", str, "ProjX")]),
        [Property("code", str, "ProjX"), Property("year", int, 2025), Property("desc", str, "Refactor & Infra")],
    )

    updated_proj_x = gs.get_node_tuple(
        NodeTuple([], [Label("Project")], [Property("code", str, "ProjX")])
    )
    assert updated_proj_x is not None
    upd_props = prop_dict(updated_proj_x)
    assert upd_props.get("code") == "ProjX"
    assert upd_props.get("year") == 2025
    assert upd_props.get("desc") == "Refactor & Infra"

    # 7) Đóng kết nối
    gs.close_connection()
    print("Complex node-tuple example executed successfully!")


if __name__ == "__main__":
    main()
