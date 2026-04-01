from gurobipy import Model, GRB
import networkx as nx

def solve_wgc(G: nx.Graph):
    """
    Solves Weighted Graph Coloring via the Representative Formulation.
    Assumes G is chordal and has a 'weight' attribute on each node.
    """
    V = list(G.nodes)
    w = nx.get_node_attributes(G, 'weight')
    cliques = list(nx.chordal_graph_cliques(G))

    m = Model()

    # x[u,v] = 1 if u is the representative of v
    # u can represent v only if w(u) >= w(v)
    x = {
        (u, v): m.addVar(vtype=GRB.BINARY, name=f"x_{u}_{v}")
        for u in V
        for v in V
        if w[u] >= w[v]
    }

    m.update()

    # (1) each vertex is assigned to exactly one representative
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
    #     each representative u can appear at most once
    for u in V:
        for i, clique in enumerate(cliques):
            clique_vars = [x[u, v] for v in clique if (u, v) in x]
            if len(clique_vars) >= 2:
                m.addConstr(
                    sum(clique_vars) <= 1,
                    name=f"clique_{u}_{i}"
                )

    # objective: minimize total weight of representatives
    m.setObjective(
        sum(w[u] * x[u, u] for u in V if (u, u) in x),
        GRB.MINIMIZE
    )

    m.optimize()

    return m