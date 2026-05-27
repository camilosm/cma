import gurobipy as gp
import networkx as nx

ENV = gp.Env(empty=True)
ENV.setParam("OutputFlag", 0)
ENV.setParam("Threads", 1)
ENV.setParam("TimeLimit", 60)
ENV.start()

def weighted_coloring(G: nx.Graph, env: gp.Env = ENV) -> bool:
    """
    Solves Vertex-Weighted Graph Coloring via the Representative Formulation.
    Assumes G is chordal and has a 'weight' attribute on each node.

    Variables:
        x[v, u]: binary, 1 if v is the representative of u
        y[v]: weight of the heaviest vertex in v's color class

    The model uses a vertex ordering by weights to break symmetry:
        v can represent u only if W[v] >= W[u]
    """

    V = list(G.nodes)
    W = nx.get_node_attributes(G, "weight")
    # order vertices by weight (ascending), breaking ties by vertex id
    # v can represent u only if order[v] <= order[u].
    order = { v: (W[v], v) for v in V }

    model = gp.Model(env=env)

    def N_closed_minus(v):
        return [ u for u in V if not G.has_edge(u, v) and order[u] < order[v] ] + [ v ]

    def N_closed_plus(v):
        return [ v ] + [ u for u in V if not G.has_edge(u, v) and order[u] > order[v] ]

    x = {
        (v, u): model.addVar(vtype=gp.GRB.BINARY, name=f"x_{v}_{u}")
        for v in V
        for u in N_closed_plus(v)
    }

    y = {
        v: model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"y_{v}")
        for v in V
    }

    # (wc:obj): minimize sum of y_v
    model.setObjective(gp.quicksum(y[v] for v in V), gp.GRB.MINIMIZE)

    # (wc:cover) each vertex receives exactly one representative
    for v in V:
        model.addConstr(
            gp.quicksum(x[u, v] for u in N_closed_minus(v)) == 1,
            name=f"cover_{v}"
        )

    # # (wc:edge) adjacent vertices cannot share a representative
    # for v in V:
    #     sub = G.subgraph([u for u in N_closed_plus(v) if u != v])
    #     for u, w in sub.edges():
    #         model.addConstr(
    #             x[v, u] + x[v, w] <= x[v, v],
    #             name=f"edge_{v}_{u}_{w}"
    #         )

    # (wc:edge) vertices in the same clique can't share a representative
    for v in V:
        sub = G.subgraph([u for u in N_closed_plus(v) if u != v])
        for clique in nx.chordal_graph_cliques(sub):
            model.addConstr(
                gp.quicksum(x[v, u] for u in clique) <= x[v, v],
                name=f"clique_{v}_{'_'.join(map(str, clique))}"
            )

    # (wc:conti) y[v] >= w[u] * x[v, u]
    for v in V:
        for u in N_closed_plus(v):
            model.addConstr(
                y[v] >= W[u] * x[v, u],
                name=f"weight_{v}_{u}"
            )

    # (wc:limits) already encoded in the binary variable x

    model.optimize()

    print("\n=== Solution Summary ===")
    if model.status == gp.GRB.OPTIMAL:
        # extract solution
        assignment = {
            (u, v): x[u, v].X
            for (u, v) in x
            if x[u, v].X > 0.5
        }

        # build representative -> vertices mapping
        rep_to_vertices = {}
        for (u, v), val in assignment.items():
            if u not in rep_to_vertices:
                rep_to_vertices[u] = []
            rep_to_vertices[u].append(v)

        # assign colors (just integer labels)
        color_map = {}
        for color_id, (rep, vertices) in enumerate(rep_to_vertices.items()):
            for v in vertices:
                color_map[v] = color_id

        # annotate graph
        nx.set_node_attributes(G, color_map, "color")

        # optional: also store representative
        nx.set_node_attributes(G, {
            v: u for (u, v) in assignment
        }, "representative")

        # print summary
        print(f"Objective (total weight): {model.objVal}")
        print(f"Number of colors: {len(rep_to_vertices)}")
        for i, (rep, vertices) in enumerate(rep_to_vertices.items()):
            print(f"Color {i}: (rep {rep}, weight={W[rep]}): {vertices}")

        return True
    else:
        print("No optimal solution found.")
        return False
