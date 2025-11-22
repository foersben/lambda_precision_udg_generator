"""Quick debug script for BaseResult serialization."""
import sys
sys.path.insert(0, '/home/pilot/projects/lambda_precision_udg_generator/src')

import networkx as nx
from lambdaprecisionudggenerator.utils.json_utils import save_json, load_json
import tempfile
import os

# Create a test graph
graph = nx.Graph()
graph.add_nodes_from([0, 1, 2, 3, 4])
graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])

for node in graph.nodes():
    graph.nodes[node]['means'] = [node % 2, (node + 1) % 2]

print("Original graph:")
print(f"  Nodes: {list(graph.nodes())}")
print(f"  Edges: {list(graph.edges())}")

# Serialize using node_link_data
graph_dict = {
    '__type__': 'networkx_graph',
    'graph_type': graph.__class__.__name__,
    'data': nx.node_link_data(graph)
}

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    temp_path = f.name

try:
    save_json(graph_dict, temp_path)
    loaded_dict = load_json(temp_path)
    
    print("\nLoaded dict type check:")
    print(f"  __type__: {loaded_dict.get('__type__')}")
    print(f"  Has data: {'data' in loaded_dict}")
    
    # Reconstruct
    if loaded_dict.get('__type__') == 'networkx_graph':
        loaded_graph = nx.node_link_graph(loaded_dict['data'])
        print("\nLoaded graph:")
        print(f"  Nodes: {list(loaded_graph.nodes())}")
        print(f"  Edges: {list(loaded_graph.edges())}")
        print(f"  Node 0 attrs: {dict(loaded_graph.nodes[0])}")
    else:
        print("ERROR: Type check failed!")
        
finally:
    os.unlink(temp_path)
