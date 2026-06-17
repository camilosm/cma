#!/usr/bin/env python3

import math
import random
import time

import networkx as nx
import matplotlib.pyplot as plt

def generate_subtree(tree, growth_prob, rng):
    start = rng.choice(list(tree.nodes))
    subtree = {start}
    frontier = [start]

    while frontier:
        node = frontier.pop()
        for neighbor in tree.neighbors(node):
            if neighbor not in subtree and rng.random() < growth_prob:
                subtree.add(neighbor)
                frontier.append(neighbor)

    return subtree

def build_intersection_graph(subtrees):
    n = len(subtrees)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if subtrees[i] & subtrees[j]:
                G.add_edge(i, j)

    return G

def build_chordal_graph(num_variables, num_tree_nodes, growth_prob, seed, weight_choices=None):
    """
    Builds a random chordal graph via the subtree intersection method.
    Node weights are stored as a 'weight' attribute on each node.

    Returns:
        G: nx.Graph with node attribute 'weight'
    """
    if weight_choices is None:
        weight_choices = list(range(1, 1000))

    rng_weight = random.Random(seed)
    weights = [ rng_weight.choice(weight_choices) for _ in range(num_variables) ]

    rng_graph = random.Random(seed)
    while True:
        tree = nx.random_labeled_rooted_tree(num_tree_nodes, seed=rng_graph.randint(0, 2**31))
        subtrees = [ generate_subtree(tree, growth_prob, rng_graph) for _ in range(num_variables) ]
        G = build_intersection_graph(subtrees)
        # graph with only one component
        if nx.is_connected(G):
            break

    nx.set_node_attributes(G, dict(enumerate(weights)), name='weight')

    assert nx.is_chordal(G), "Generated graph is not chordal, this should never happen"

    return G

def scale_graph(G: nx.Graph, factor: int, direction: str) -> nx.Graph:
    """
    Returns a copy of G with scaled vertex weights, according to the direction:
        up: w' = w * factor
        down: w' = ceil(w / factor)
    """
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")

    H = G.copy()
    W = nx.get_node_attributes(H, "weight")

    if direction == "up":
        new_W = { v: w * factor for v, w in W.items() }
    elif direction == "down":
        new_W = { v: math.ceil(w / factor) for v, w in W.items() }
    else:
        raise ValueError(f"Invalid scaling direction: {direction}")

    nx.set_node_attributes(H, new_W, "weight")

    return H

def split_graph(G: nx.Graph, p: float) -> tuple[nx.Graph, nx.Graph]:
    """
    Splits G into two subgraphs based on a cutoff weight c, computed as:
    c = p * w[h], where h is G's heaviest node and w[h] its weight.
    G1 (G2) receives nodes with weight < (≥) c.

    Args:
        G:  Input graph with a 'weight' node attribute.
        p:  Fraction in (0, 1] used to compute the cutoff.

    Returns:
        (G1, G2): induced subgraphs sharing G's edge structure.
    """
    if not (0 < p <= 1):
        raise ValueError(f"p must be in (0, 1], got {p}")

    weights = nx.get_node_attributes(G, "weight")

    max_weight = max(weights.values())
    cutoff = p * max_weight

    light = [ v for v, w in weights.items() if w < cutoff ]
    heavy = [ v for v, w in weights.items() if w >= cutoff ]

    G1 = G.subgraph(light).copy()
    G2 = G.subgraph(heavy).copy()

    return G1, G2

def plot_graph(G: nx.Graph):
    colors = [ G.nodes[v].get("color", "grey") for v in G.nodes ]
    labels = { v: f"{v}\n{G.nodes[v]['weight']}" for v in G.nodes }
    # sizes = [ len(labels[v])**2 * 50 for v in G ]
    pos = nx.spring_layout(G, seed=13)
    nx.draw_networkx(G, pos, node_color=colors, labels=labels, node_size=900, node_shape='s')
    plt.show()

def print_dot(G: nx.Graph):
    print("graph G {")
    for u, data in G.nodes(data=True):
        print(f'    {u} [label="{u} ({data["weight"]})"];')
    for u, v in G.edges():
        print(f"    {u} -- {v};")
    print("}")

if __name__ == '__main__':
    # SEED = 13
    SEED = int(time.time())
    print(f"# Seed: {SEED}")

    NUM_VARIABLES = 20
    NUM_TREE_NODES = 15
    SUBTREE_GROWTH_PROB = 0.4

    cg = build_chordal_graph(NUM_VARIABLES, NUM_TREE_NODES, SUBTREE_GROWTH_PROB, SEED)

    print_dot(cg)

    cliques = list(nx.chordal_graph_cliques(cg))
    print(f"# {len(cliques)} maximal cliques found")
