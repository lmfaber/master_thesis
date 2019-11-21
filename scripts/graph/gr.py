import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.pyplot import arrow

node_size = 1000
font_size = 18
edge_width = 5
arrow_style = '-|>'
arrow_size=30
node_color = 'tab:blue'


G = nx.DiGraph()

G.add_edge('A','C')
G.add_edge('B','D')
G.add_edge('C','D')
G.add_edge('C','B')
G.add_edge('C','F')
G.add_edge('C','E')
G.add_edge('A','B')

pos = nx.circular_layout(G)
nx.draw(G, pos=pos, with_labels=True, node_color=node_color, node_size=node_size, font_size=font_size, width=edge_width, arrowstyle=arrow_style, arrowsize=arrow_size)
plt.axis('off')

plt.savefig('unweighted.svg', format='svg')
plt.clf()


# plt.figure(figsize=(20,10))

H = nx.Graph()

H.add_edge('A','B',weight=0.6)
H.add_edge('A','C',weight=0.2)
H.add_edge('C','D',weight=0.1)
H.add_edge('C','E',weight=0.7)
H.add_edge('C','F',weight=0.9)
H.add_edge('A','D',weight=0.3)

elarge = [(u, v) for (u, v, d) in H.edges(data=True) if d['weight'] > 0]
esmall = [(u, v) for (u, v, d) in H.edges(data=True) if d['weight'] == 0.0]

# print(pos)
pos = nx.circular_layout(H)

nx.draw(H, pos=pos, with_labels=True, node_color=node_color, node_size=node_size, font_size=font_size, width=edge_width)
# Nodes
# nx.draw_networkx_nodes(H, pos, node_size=node_size)

# # Edges
# nx.draw_networkx_edges(H, pos, edgelist=elarge,width=edge_width)
# nx.draw_networkx_edges(H, pos, edgelist=esmall,width=edge_width)

# Edge labels of non zero weights
edge_labels = {}
for u,v in H.edges():
    edge_labels[(u,v)] = round(H[u][v]['weight'], 3)
nx.draw_networkx_edge_labels(H, pos, label_pos = 0.65, edge_labels = edge_labels, font_size=font_size)

plt.axis('off')
plt.savefig('weighted.svg', format='svg')