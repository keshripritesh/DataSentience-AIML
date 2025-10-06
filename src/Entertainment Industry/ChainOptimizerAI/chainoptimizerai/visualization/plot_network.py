import argparse
import json
import networkx as nx
import matplotlib.pyplot as plt


def plot_network(cfg):
    G = nx.DiGraph()
    for f in cfg["factories"]:
        G.add_node(f['id'], type='factory')
    for w in cfg["warehouses"]:
        G.add_node(w['id'], type='warehouse')
    for r in cfg["retailers"]:
        G.add_node(r['id'], type='retailer')
    for e in cfg["edges_fw"]:
        G.add_edge(e['source_id'], e['target_id'])
    for e in cfg["edges_wr"]:
        G.add_edge(e['source_id'], e['target_id'])

    pos = nx.spring_layout(G, seed=42)
    colors = []
    for n in G.nodes:
        t = G.nodes[n]['type']
        colors.append({'factory': 'tab:blue', 'warehouse': 'tab:orange', 'retailer': 'tab:green'}[t])
    nx.draw(G, pos, with_labels=True, node_color=colors, arrows=True)
    plt.title('Supply Chain Network')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    plot_network(cfg)


if __name__ == '__main__':
    main()


