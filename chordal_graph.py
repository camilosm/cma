#!/usr/bin/env python3

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

def plot_graph(G: nx.Graph):
    colors = [ G.nodes[v].get("color", "grey") for v in G.nodes ]
    pos = nx.spring_layout(G, seed=13)
    nx.draw_networkx(G, pos, node_color=colors)
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
