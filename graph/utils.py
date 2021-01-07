import matplotlib.pyplot as plt
import numpy as np
from colormap import rgb2hex
import networkx as nx

def color_list(n):
    color_names = []
    sample = np.arange(0.2,1,min(0.4,0.8/np.ceil(n**(1/3))))
    for i in sample:
        for j in sample:
            for k in sample:
                col = rgb2hex(i, j, k, normalised=True)
                color_names.append(col)
    return color_names


def draw_graph(G,out_dir="graph_tmp.pdf"):
    plt.figure(figsize=(30,20))
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    nodes,degrees = zip(*dict(G.degree).items())
    colors = (nx.get_node_attributes(G,'class').items())
    colored = False if len(colors) == 0 else True
    degrees = np.array(degrees)
    degrees = (degrees-np.min(degrees))/(np.max(degrees)-np.min(degrees))
    degrees = tuple(degrees*200+50)
    # pos = nx.spring_layout(G)
    pos = nx.kamada_kawai_layout(G)
    if colored:
        nodes,classes = zip(*colors)
        lbs = np.unique(classes)
        class_map = {}
        for n,lb in enumerate(lbs):
            class_map[lb] = n
        n_class = len(lbs)
        cl = color_list(n_class)
        colors = [cl[class_map[lb]] for lb in classes]
        nx.draw(G, pos, node_color=colors, edgelist=edges, edge_color="k",width=0.5,with_labels=False,node_size=degrees,arrows=False)
    else:
        nx.draw(G, pos, node_color="b", edgelist=edges, edge_color="k",width=0.5,with_labels=False,node_size=degrees,arrows=False)
    plt.savefig(out_dir,dpi=300)

def load_graph(filename):
    try:
        G = nx.read_edgelist(filename,create_using=nx.DiGraph,nodetype=int)
        for v in G.nodes():
            for nn in G.neighbors(v):
                G[v][nn]["weight"] = 1
        print("Directed, unweighted")
    except:
        G = nx.read_edgelist(filename,create_using=nx.DiGraph,nodetype=int,data=(('weight',float),))
        print("Directed, weighted")
    return G
