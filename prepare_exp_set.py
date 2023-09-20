import os
import math
import pickle
import torch
import pandas as pd
import networkx as nx
from tqdm import tqdm
from torch_geometric.seed import seed_everything
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import CitationFull, Coauthor, Flickr, RelLinkPredDataset, WordNet18, WordNet18RR
from torch_geometric.utils import train_test_split_edges, k_hop_subgraph, negative_sampling, to_undirected, is_undirected, to_networkx
from ogb.linkproppred import PygLinkPropPredDataset
from framework.utils import *

data_dir = './exp_data'
graph_datasets = ['Cora']

def train_test_split_no_neg_adj(data, val_ratio=0.05, test_ratio=0.1, two_hop_degree=None):
    row, col = data.edge_index
    edge_attr = data.edge_attr

    data.edge_index = data.edge_attr = data.edge_weight = data.edge_year = data.edge_type = None
    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    perm = torch.randperm(row.size(0))
    print('row: ', row)
    print('col: ', col)
    print('perm: ', perm)
    row, col = row[perm], col[perm]
    print('Randomly permuting row and col: ')
    print('row: ', row)
    print('col: ', col)
    print('For Train Data: ')
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    print('r: ', r)
    print('c: ', c)

    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        out = to_undirected(data.train_pos_edge_index, edge_attr[n_v + n_t:])
        data.train_pos_edge_index, data.train_pos_edge_attr = out
    else:
        data.train_pos_edge_index = data.train_pos_edge_index
        # data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)
    
    assert not is_undirected(data.train_pos_edge_index)

    # Test
    print('For Test Data: ')
    r, c = row[:n_t], col[:n_t]
    print('r: ', r)
    print('c: ', c)
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    neg_edge_index = negative_sampling(
            edge_index=data.test_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.test_pos_edge_index.shape[1])
    data.test_neg_edge_index = neg_edge_index

    # Valid
    print('For Valid Data: ')
    r, c = row[n_t:n_t+n_v], col[n_t:n_t+n_v]
    print('r: ', r)
    print('c: ', c)
    data.val_pos_edge_index = torch.stack([r, c], dim=0)

    neg_edge_index = negative_sampling(
            edge_index=data.val_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.val_pos_edge_index.shape[1])

    data.val_neg_edge_index = neg_edge_index

    return data
    

def make_expset():
    for dataset in graph_datasets:
        print('Processing {} dataset...'.format(dataset))
        
        if dataset == 'Cora':
            dataset = CitationFull(root=data_dir, name=dataset)
        
        data = dataset[0]
        data.train_mask = data.val_mask = data.test_mask = None
        graph = to_networkx(data)

        # Get two hop degree for all nodes
        node_to_neighbors = {}
        for n in tqdm(graph.nodes(), desc='Two hop neighbors'):
            neighbor_1 = set(graph.neighbors(n))
            neighbor_2 = sum([list(graph.neighbors(i)) for i in neighbor_1], [])
            # neighbour 2:  [10930, 10943, 611, 1009, 5045, 5379, 5537, 10386, 10918, 10943, 11631, 15631, 16322, 17130]
            neighbor_2 = set(neighbor_2)
            neighbor = neighbor_1 | neighbor_2
            # neighbour OR op:  {5537, 16322, 611, 5379, 10918, 17130, 11631, 15631, 1009, 10930, 10386, 5045, 10934, 10935, 10943}
            node_to_neighbors[n] = neighbor
        
        two_hop_degree = []
        row, col = data.edge_index
        mask = row < col
        row, col = row[mask], col[mask]
        for r, c in tqdm(zip(row, col), total=len(row)):
            neighbor_row = node_to_neighbors[r.item()]
            neighbor_col = node_to_neighbors[c.item()]
            neighbor = neighbor_row | neighbor_col
            
            num = len(neighbor)
            
            two_hop_degree.append(num)

        two_hop_degree = torch.tensor(two_hop_degree)

        data = dataset[0]

        data = train_test_split_no_neg_adj(data, val_ratio=0.05, test_ratio=0.1)
        print(data)

        with open(os.path.join(data_dir, f'org_data.pkl'), 'wb') as f:
                pickle.dump((dataset, data), f)
        
        # Two ways to sample Df from the training set
            ## 1. Df is within 2 hop local enclosing subgraph of Dtest
            ## 2. Df is outside of 2 hop local enclosing subgraph of Dtest

        # Get the 2 hop local enclosing subgraph for all test edges
        _, local_edges, _, mask = k_hop_subgraph(
            data.test_pos_edge_index.flatten().unique(), # taking nodes of test datasets
            2, 
            data.train_pos_edge_index, # getting 2-hop neighbourhood edges of test nodes from train dataset
            num_nodes=dataset[0].num_nodes)
        distant_edges = data.train_pos_edge_index[:, ~mask]

        print('Number of edges. 2-hop Local: ', local_edges.shape[1], 'Distant:', distant_edges.shape[1])
        print('local_edges: ', local_edges)
        print('distant_edges: ', distant_edges)

        print('Complete train edge set: ', data.train_pos_edge_index)

        in_mask = mask # edges in 2-hop local enclosing subgraph
        out_mask = ~mask # edges outside 2-hop local enclosing subgraph

        print('in_mask: ', in_mask)
        print('out_mask: ', out_mask)

        # print number of true values in in_mask and out_mask
        print('Number of true values in in_mask: ', in_mask.sum())
        print('Number of true values in out_mask: ', out_mask.sum())

        # print indexes of true values in in_mask and out_mask
        print('Indexes of true values in in_mask: ', in_mask.nonzero())
        print('Indexes of true values in out_mask: ', out_mask.nonzero())

def main():
    make_expset()
    # process_kg()

if __name__ == "__main__":
    main()