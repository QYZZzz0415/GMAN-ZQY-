import node2vec
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

is_directed = True
p = 2
q = 1
num_walks = 100
walk_length = 80
dimensions = 64
window_size = 10
iter = 1000
Adj_file = 'adj(BJ276).txt'
SE_file = 'SE.txt'


def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight', float),),
        create_using=nx.DiGraph())                                #  从txt文件中读入有向图的函数

    return G


def learn_embeddings(walks, dimensions, output_file):
    walks = [list(map(str, walk)) for walk in walks]                 #map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
    model = Word2Vec(
        walks, vector_size=dimensions, window=10, min_count=0, sg=1,
        workers=8, epochs=iter)
    model.wv.save_word2vec_format(output_file)

    return


nx_G = read_graph(Adj_file)
G = node2vec.Graph(nx_G, is_directed, p, q)
G.preprocess_transition_probs()
walks = G.simulate_walks(num_walks, walk_length)
learn_embeddings(walks, dimensions, SE_file)
