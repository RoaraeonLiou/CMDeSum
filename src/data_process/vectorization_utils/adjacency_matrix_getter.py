import json
import os.path
import networkx as nx
import numpy as np
import torch
import scipy.sparse as sp
from scipy import sparse
from tqdm import tqdm


class AdjacencyMatrixGetter(object):
    def __init__(self, max_node, language):
        self.max_node = max_node
        self.asts = list()
        self.adjacency_matrix = dict()
        for i in range(5):
            self.adjacency_matrix[i + 1] = list()
        language = language.lower()
        assert language in {"java", "python"}
        self.language = language
        self.node_type = {
            "for": "ForStatement",
            "block": "BlockStatement",
            "while": "WhileStatement",
            "if": "IfStatement"
        }
        if language == 'python':
            self.node_type["for"] = "For",
            self.node_type["while"] = "While",
            self.node_type["if"] = "If",

    def clear(self):
        self.asts.clear()
        for i in range(5):
            self.adjacency_matrix[i + 1].clear()

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        row_sum = np.array(mx.sum(1))
        r_inv = np.power(row_sum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def normalize_data(self, data):
        ret = []
        for line in data:
            q = np.sum(line ** 2)
            if q != 0:
                normalized_line = line / np.sqrt(q)
                ret.append(normalized_line)
            else:
                ret.append(line)
        return ret

    def read_ast_file(self, ast_file_path):
        with open(ast_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                ast_json = json.loads(line)
                self.asts.append(ast_json)
            f.close()

    def parse_adjacency_matrix(self):
        ast_num = len(self.asts)
        for i in tqdm(range(ast_num)):
            self.asts[i][0].update({'layer': 0})
            for a in self.asts[i]:
                if 'children' in a:
                    for child_id in a['children']:
                        self.asts[i][child_id].update({'layer': a['layer'] + 1})
            ast = self.asts[i]

            id_to_children = {node['id']: node['children'] for node in ast if 'children' in node}

            edge_list = []
            for node in id_to_children:
                for child in id_to_children[node]:
                    edge_list.append((id, child))

            for_type_node_id_dict = dict()
            block_type_node_id_dict = dict()
            while_type_node_id_dict = dict()
            if_type_node_id_dict = dict()

            for node in ast:
                if 'children' in node:
                    if node['type'] == self.node_type['for']:
                        for_type_node_id_dict[node['id']] = node['children']
                    elif node['type'] == self.node_type['while']:
                        while_type_node_id_dict[node['id']] = node['children']
                    elif node['type'] == self.node_type['if']:
                        if_type_node_id_dict[node['id']] = node['children']
                    elif node['type'] == self.node_type['block']:
                        block_type_node_id_dict[node['id']] = node['children']

            non = []
            for node in ast:
                if "children" not in node:
                    non.append(node["id"])

            add_non_edges = []
            for i in range(len(non) - 1):
                x = (non[i], non[i + 1])
                add_non_edges.append(x)
            for j in add_non_edges:
                edge_list.append(j)

            classified_dict = {}
            for i, d in enumerate(ast):
                layer = d['layer']
                if layer in classified_dict:
                    classified_dict[layer].append(i)
                else:
                    classified_dict[layer] = [i]

            add_nub_edges = []
            for value in classified_dict.values():
                if len(value) >= 2:
                    for i in range(len(value) - 1):
                        xx = (value[i], value[i + 1])
                        add_nub_edges.append(xx)
            for k in add_nub_edges:
                edge_list.append(k)

            add_for_edges = []
            for children_list in for_type_node_id_dict.values():
                if len(children_list) > 0:
                    add_for_edges.append((children_list[0], children_list[-1]))

            for edge in add_for_edges:
                edge_list.append(edge)

            add_block_edges = []
            for children_list in block_type_node_id_dict.values():
                if len(children_list) > 1:
                    for k in range(len(children_list) - 1):
                        add_block_edges.append((children_list[k], children_list[k + 1]))

            for edge in add_block_edges:
                edge_list.append(edge)

            add_while_edges = []
            for children_list in while_type_node_id_dict.values():
                if len(children_list) > 0:
                    add_while_edges.append((children_list[0], children_list[-1]))
            # print(add_while_edges)
            for edge in add_while_edges:
                edge_list.append(edge)

            add_if_edges = []
            for children_list in if_type_node_id_dict.values():
                if len(children_list) > 0:
                    if len(children_list) == 3:
                        for c in range(len(children_list) - 1):
                            add_if_edges.append((children_list[0], children_list[c + 1]))
                    else:
                        add_if_edges.append((children_list[0], children_list[-1]))
            for edge in add_if_edges:
                edge_list.append(edge)

            graph = nx.Graph()

            graph.add_edges_from(edge_list)
            if len(graph.nodes) == 0:
                tmp_node_list = list()
                for a in ast:
                    tmp_node_list.append(a['id'])
                graph.add_nodes_from(tmp_node_list)
            nx.draw(graph, with_labels=True)

            adj = np.array(nx.adjacency_matrix(graph).todense())
            adj = adj + sp.eye(adj.shape[0])
            adj = np.array(adj, dtype=int)

            if len(adj[0]) > self.max_node:
                a = adj[0:self.max_node, 0:self.max_node]
            else:
                a = np.zeros((self.max_node, self.max_node), dtype=int)
                for i in range(adj.shape[0]):
                    for j in range(adj.shape[1]):
                        a[i][j] = adj[i][j]

            a2 = a.dot(a)
            a3 = a.dot(a2)
            a4 = a.dot(a3)
            a5 = a.dot(a4)

            adjacency_matrix_2 = self.normalize_data(a2)
            adjacency_matrix_2 = np.array(adjacency_matrix_2)
            adjacency_matrix_2 = sparse.csr_matrix(adjacency_matrix_2)
            adjacency_matrix_2 = torch.FloatTensor(np.array(adjacency_matrix_2.todense()))
            self.adjacency_matrix[2].append(adjacency_matrix_2)

            adjacency_matrix_3 = self.normalize_data(a3)
            adjacency_matrix_3 = np.array(adjacency_matrix_3)
            adjacency_matrix_3 = sparse.csr_matrix(adjacency_matrix_3)
            adjacency_matrix_3 = torch.FloatTensor(np.array(adjacency_matrix_3.todense()))
            self.adjacency_matrix[3].append(adjacency_matrix_3)

            adjacency_matrix_4 = self.normalize_data(a4)
            adjacency_matrix_4 = np.array(adjacency_matrix_4)
            adjacency_matrix_4 = sparse.csr_matrix(adjacency_matrix_4)
            adjacency_matrix_4 = torch.FloatTensor(np.array(adjacency_matrix_4.todense()))
            self.adjacency_matrix[4].append(adjacency_matrix_4)

            adjacency_matrix_5 = self.normalize_data(a5)
            adjacency_matrix_5 = np.array(adjacency_matrix_5)
            adjacency_matrix_5 = sparse.csr_matrix(adjacency_matrix_5)
            adjacency_matrix_5 = torch.FloatTensor(np.array(adjacency_matrix_5.todense()))
            self.adjacency_matrix[5].append(adjacency_matrix_5)

            a = np.array(a, dtype=float)
            adjacency_matrix_1 = self.normalize(a)
            adjacency_matrix_1 = sp.csr_matrix(adjacency_matrix_1)
            adjacency_matrix_1 = torch.FloatTensor(np.array(adjacency_matrix_1.todense()))
            self.adjacency_matrix[1].append(adjacency_matrix_1)

    def save_to_file(self, output_dir):
        paths = {
            1: os.path.join(output_dir, "adj1.pt"),
            2: os.path.join(output_dir, "adj2.pt"),
            3: os.path.join(output_dir, "adj3.pt"),
            4: os.path.join(output_dir, "adj4.pt"),
            5: os.path.join(output_dir, "adj5.pt"),
        }
        for i in range(5):
            torch.save(self.adjacency_matrix[i + 1], paths[i + 1])
            print("output to " + paths[i + 1] + " success...")

    def run(self, ast_file_path, output_dir):
        self.clear()
        print("Reading the ast file...")
        self.read_ast_file(ast_file_path)
        print("Generating the adjacency matrix...")
        self.parse_adjacency_matrix()
        print("Saving files...")
        self.save_to_file(output_dir)
