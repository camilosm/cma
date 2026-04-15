import gurobipy as gp
import networkx as nx

ENV = gp.Env(empty=True)
ENV.setParam("OutputFlag", 0)
ENV.setParam("Threads", 1)
ENV.setParam("TimeLimit", 60)
ENV.start()

def solve_wgc(G: nx.Graph, env: gp.Env=ENV):
    """
    Solves Weighted Graph Coloring via the Representative Formulation.
    Assumes G is chordal and has a 'weight' attribute on each node.
    """
    V = list(G.nodes)
    w = nx.get_node_attributes(G, 'weight')
    cliques = list(nx.chordal_graph_cliques(G))

    m = gp.Model(env=ENV)

    # x[u,v] = 1 if u is the representative of v
    # u can represent v only if w(u) >= w(v)
    x = {
        (u, v): m.addVar(vtype=gp.GRB.BINARY, name=f"x_{u}_{v}")
        for u in V
        for v in V
        if w[u] >= w[v]
    }

    # (1) objective: minimize total weight of representatives
    m.setObjective(
        sum(w[u] * x[u, u] for u in V if (u, u) in x),
        gp.GRB.MINIMIZE
    )

    # (2) each vertex is assigned to exactly one representative
    for v in V:
        m.addConstr(
            sum(x[u, v] for u in V if (u, v) in x) == 1,
            name=f"assign_{v}"
        )

    # (2) a vertex can only be assigned to u if u is a representative
    for u, v in x:
        if u != v:
            m.addConstr(x[u, v] <= x[u, u], name=f"rep_{u}_{v}")

    # (3) clique constraints: within each maximal clique,
    # each representative u can appear at most once
    for u in V:
        for i, clique in enumerate(cliques):
            clique_vars = [ x[u, v] for v in clique if (u, v) in x ]
            if len(clique_vars) >= 2:
                m.addConstr(
                    sum(clique_vars) <= 1,
                    name=f"clique_{u}_{i}"
                )

    m.optimize()

    print("\n=== Solution Summary ===")
    if m.status == gp.GRB.OPTIMAL:
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
        print(f"Objective (total weight): {m.objVal}")
        print(f"Number of colors: {len(rep_to_vertices)}")
        for i, (rep, vertices) in enumerate(rep_to_vertices.items()):
            print(f"Color {i}: (rep {rep}, weight={w[rep]}): {vertices}")

        return True
    else:
        print("No optimal solution found.")
        return False
