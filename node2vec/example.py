import networkx as nx
from node2vec import Node2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from graph import utils as gutils
import os

# FILES
EMBEDDING_FILENAME = './embeddings.emb'
PNG_FN = "./tsne.png"
# Create a graph
filename = "./data/email-Eu-core.txt"
# graph = nx.fast_gnp_random_graph(n=100, p=0.5)
y = np.loadtxt("./data/email-Eu-core-department-labels.txt",dtype=int)[:,1]

graph = gutils.load_graph(filename)
for n in np.where((y!=1)&(y!=0))[0]:
    graph.remove_node(n)
if not os.path.isfile(EMBEDDING_FILENAME):
    # Precompute probabilities and generate walks
    node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
    
    ## if d_graph is big enough to fit in the memory, pass temp_folder which has enough disk space
    # Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
    #node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder="/mnt/tmp_data")
    
    # Embed
    model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
    
    # Look for most similar nodes
    model.wv.most_similar('1')  # Output node names are always strings
    
    # Save embeddings for later use
    model.wv.save_word2vec_format(EMBEDDING_FILENAME)

repres = np.loadtxt(EMBEDDING_FILENAME,skiprows=1)
node_id = repres[:,0]
yy = np.stack([0 if int(id) in np.where(y==0)[0] else 1 for id in node_id])
z = repres[:,1:]
tsne = TSNE()
fea = tsne.fit_transform(z)
for lb in [0,1]:
    plt.scatter(fea[:,0][yy==lb],fea[:,1][yy==lb])
plt.savefig(PNG_FN,dpi=300)
for v in graph.nodes():
    graph.nodes[v]['class'] = y[v]
gutils.draw_graph(graph)

