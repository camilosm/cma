import gurobipy as gp
import networkx as nx

ENV = gp.Env(empty=True)
ENV.setParam("OutputFlag", 0)
ENV.setParam("Threads", 1)
ENV.start()

def weighted_coloring(G: nx.Graph, env: gp.Env = ENV) -> bool:
    """
    Solves Vertex-Weighted Graph Coloring via the Representative Formulation.
    Assumes G is chordal and has a 'weight' attribute on each node.

    Variables:
        x[v, u]: binary, 1 if v is the representative of u
        y[v]: weight of the heaviest vertex in v's color class

    Objetive:
        minimize the total weight of the colors (sum of y)

    The model uses a vertex ordering by weights to break symmetry:
        v can represent u only if W[v] >= W[u]
    """

    V = list(G.nodes)
    W = nx.get_node_attributes(G, "weight")
    # order vertices by weight (ascending), breaking ties by vertex id
    # v can represent u only if order[v] >= order[u].
    order = { v: (W[v], v) for v in V }

    model = gp.Model(name="weighted_coloring", env=env)

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

    print("\n=== Solver Metrics ===")
    print(f"Status      : {model.status}")
    print(f"Runtime     : {model.Runtime:.4f} s")
    print(f"Variables   : {model.NumVars}")
    print(f"Constraints : {model.NumConstrs}")
    print(f"Nonzeros    : {model.NumNZs}")
    print(f"Nodes       : {model.NodeCount}")
    print(f"Iterations  : {model.IterCount:.0f}")
    print(f"Solutions   : {model.SolCount}")
    if model.SolCount > 0:
        print(f"Objective   : {model.objVal:.4f}")
        print(f"Best Bound  : {model.ObjBound:.4f}")
        print(f"MIP Gap     : {model.MIPGap:.6f}")
    else:
        print("No feasible solution found.")
        return False

    if model.status != gp.GRB.OPTIMAL:
        print("No optimal solution found.")
        return False

    # build { representative: [members] }
    rep_to_vertices = {
        v: {v} for (v, u), var in x.items()
        if v == u and var.X > 0.5
    }
    for (v, u), var in x.items():
        if v != u and var.X > 0.5:
            rep_to_vertices[v].add(u)

    # rep_to_vertices = {}
    # for (v, u), var in x.items():
    #     if var.X > 0.5:
    #         if v not in rep_to_vertices:
    #             rep_to_vertices[v] = {u}
    #         else:
    #             rep_to_vertices[v].add(u)

    # verify coverage
    assigned = [ u for members in rep_to_vertices.values() for u in members ]
    assert len(assigned) == len(V), "There are unassigned vertices."
    assert len(set(assigned)) == len(assigned), "There are doubly assigned vertices."

    # set nodes atttributes (color and representative)
    color_map, rep_map = {}, {}
    for color_id, (rep, members) in enumerate(rep_to_vertices.items()):
        for v in members:
            color_map[v] = color_id
            rep_map[v] = rep
    nx.set_node_attributes(G, color_map, "color")
    nx.set_node_attributes(G, rep_map,   "representative")

    print("\n=== Solution ===")
    print(f"Number of colors: {len(rep_to_vertices)}")
    for color_id, (rep, members) in enumerate(rep_to_vertices.items()):
        heaviest = max(members, key=lambda u: W[u])
        print(
            f"  Color {color_id}: rep={rep}, "
            f"heaviest={heaviest} (w={W[heaviest]}), "
            f"members={members}"
        )

    return True

def consecutive_coloring(G: nx.Graph, k: int = None, env: gp.Env = ENV):
    """
    Solves Vertex-Weighted Graph Coloring via Consecutive Coloring.
    Assumes G is chordal and has a 'weight' attribute on each node.

    Variables:
        x[v, i]: binary, 1 if v is assigned first color i
        y: total number of colors used (makespan)

    Objective:
        minimize maximum used memory position

    Each vertex v occupies a consecutive block of w_v colors,
    starting at some i <= k - w_v,
    and adjacent vertices cannot share any color in their block.
    """

    V = list(G.nodes)
    W = nx.get_node_attributes(G, "weight")
    # upper bound (all blocks placed sequentially):
    if k is None:
        k = sum(W[v] for v in V)
    # valid start positions: i_v <= k - w_v
    I = { v: range(k - W[v] + 1) for v in V }

    model = gp.Model(name="consecutive_coloring", env=env)

    x = {
        (v, i): model.addVar(vtype=gp.GRB.BINARY, name=f"x_{v}_{i}")
        for v in V
        for i in I[v]
    }

    y = model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name="y")

    # cc:obj
    model.setObjective(y, gp.GRB.MINIMIZE)

    # cc:cover
    for v in V:
        model.addConstr(
            gp.quicksum(x[v, i] for i in I[v]) == 1,
            name=f"cover_{v}"
        )

    # cc:edge
    for v, u in G.edges():
        for fixed, neighbor in [ (v, u), (u, v) ]:
            for i in I[fixed]:
                overlap = [
                    x[neighbor, i + j]
                    for j in range(W[fixed])
                    if (i + j) in I[neighbor]
                ]
                if overlap:
                    model.addConstr(
                        x[fixed, i] + gp.quicksum(overlap) <= 1,
                        name=f"edge_{fixed}_{neighbor}_{i}"
                    )

    # cc:conti
    for v in V:
        model.addConstr(
            y >= gp.quicksum(
                (i + W[v]) * x[v, i]
                for i in I[v]
            ),
            name=f"makespan_{v}"
        )

    # (cc:limits) already encoded in the binary variable x

    model.optimize()

    print("\n=== Solver Metrics ===")
    print(f"Status      : {model.status}")
    print(f"Runtime     : {model.Runtime:.4f} s")
    print(f"Variables   : {model.NumVars}")
    print(f"Constraints : {model.NumConstrs}")
    print(f"Nonzeros    : {model.NumNZs}")
    print(f"Nodes       : {model.NodeCount}")
    print(f"Iterations  : {model.IterCount:.0f}")
    print(f"Solutions   : {model.SolCount}")
    if model.SolCount > 0:
        print(f"Objective   : {model.objVal:.4f}")
        print(f"Best Bound  : {model.ObjBound:.4f}")
        print(f"MIP Gap     : {model.MIPGap:.6f}")
    else:
        print("No feasible solution found.")
        return False

    if model.status != gp.GRB.OPTIMAL:
        print("No optimal solution found.")
        return False

    # extract solution
    start = {}
    for v in V:
        for i in I[v]:
            if x[v, i].X > 0.5:
                start[v] = i
                break

    blocks = {
        v: list(range(start[v], start[v] + W[v]))
        for v in V
    }

    nx.set_node_attributes(G, start, "color")
    nx.set_node_attributes(G, blocks, "color_block")

    print("\n=== Solution ===")
    print(f"Memory used: {int(round(y.X))}")

    for v in sorted(V):
        print(
            f"Vertex {v}: "
            f"start={start[v]}, "
            f"block={blocks[v]}"
        )

    return True
