from HOGDB.db.neo4j import Neo4jDatabase
from HOGDB.graph.hypergraph_storage import *
def main() -> None:
    # initialize database and graph storage
    db = Neo4jDatabase()
    gs = HyperGraphStorage(db)

    # clear the graph
    gs.clear_graph()

    # add nodes
    richard = Node(labels=[Label("Person")], properties=[Property("name", str, "Richard"), Property("born", int, 1999)])
    mary = Node(labels=[Label("Person")], properties=[Property("name", str, "Mary"), Property("born", int, 2001)])
    bob = Node(labels=[Label("Person")], properties=[Property("name", str, "Bob"), Property("born", int, 2024)])
    ford = Node(labels=[Label("Company")], properties=[Property("name", str, "Ford"), Property("industry", str, "Cars")])
    car = Node(labels=[Label("Car")], properties=[Property("model", str, "F-150"), Property("power", str, "325hp")])

    gs.add_node(richard)
    gs.add_node(mary)
    gs.add_node(bob)
    gs.add_node(ford)
    gs.add_node(car)
    
    print("Đã thêm 5 nút (Person, Company, Car).")

    # assert node count
    assert gs.get_node_count([Label("Person")]) == 3
    assert gs.get_node_count([Label("Company")]) == 1
    assert gs.get_node_count([Label("Car")]) == 1
    print("Xác nhận số lượng nút đúng.")

    # add a hyperedge (family relationship)
    family_hyperedge = HyperEdge(
        nodes=[richard, mary, bob],
        label=Label("Family"),
        properties=[Property("domicile", str, "Texas"), Property("last_name", str, "Smith")],
    )
    gs.add_hyperedge(family_hyperedge)
    print("Đã thêm siêu cạnh Family kết nối Richard, Mary và Bob.")

    # assert hyperedge count
    assert gs.get_hyperedge_count([Label("Family")]) == 1
    print("Xác nhận số lượng siêu cạnh đúng.")

    # add edge between Company and Car
    company_car_edge = Edge(
        start_node=ford,
        end_node=car,
        label=Label("Builds"),
        properties=[Property("since", int, 1975)],
    )
    gs.add_edge(company_car_edge)
    print("Đã thêm cạnh Builds kết nối Ford và F-150.")
    
    # add edge between Person and Car
    person_car_edge = Edge(
        start_node=richard,
        end_node=car,
        label=Label("Owns"),
        properties=[Property("since", int, 2020)],
    )
    gs.add_edge(person_car_edge)
    print("Đã thêm cạnh Owns kết nối Richard và F-150.")

    # add edge between Richard and Mary
    marriage_edge = Edge(
        start_node=richard,
        end_node=mary,
        label=Label("Married"),
        properties=[Property("since", int, 2023)],
    )
    gs.add_edge(marriage_edge)
    print("Đã thêm cạnh Married kết nối Richard và Mary.")

    # add edge between Bob and Car
    bob_car_edge = Edge(
        start_node=bob,
        end_node=car,
        label=Label("AfraidOf"),
        properties=[Property("reason", str, "too loud")],
    )
    gs.add_edge(bob_car_edge)
    print("Đã thêm cạnh AfraidOf kết nối Bob và F-150.")
    # close connection
    gs.close_connection()
    print("Family graph example executed successfully!")
if __name__ == "__main__":
    main()


# MATCH (n)
# OPTIONAL MATCH (n)-[r]-()
# RETURN n, r

# // Tìm Hyperedge đại diện cho gia đình (Dựa trên cơ chế Lowering)
# MATCH (family:Family {last_name: 'Smith', domicile: 'Texas'})

# // Tìm các node thành viên (Person) được kết nối vào Hyperedge này
# MATCH (family)<-[:_adjacency]-(person:Person)

# RETURN 
#     family.last_name AS FamilyName,
#     family.domicile AS Address,
#     collect(person.name) AS Members,
#     avg(person.born) AS Average_Birth_Year

