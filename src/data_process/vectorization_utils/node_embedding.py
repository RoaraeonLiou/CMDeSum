from bert_serving.client import BertClient
import numpy as np
import torch
import scipy.sparse as sp
import json
import os
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class NodeEmbedding(object):
    def __init__(self, max_node, max_value_len, bert_server_ip):
        self.max_node = max_node
        self.max_value_len = max_value_len
        self.asts = list()
        self.node_embedding = list()
        self.ast_num = 0
        self.bert_server_ip = bert_server_ip
        self.bert_encoding = BertClient(self.bert_server_ip, check_length=False)

    def clear(self):
        self.asts.clear()
        self.node_embedding.clear()

    def read_ast_file(self, ast_file_path):
        with open(ast_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                ast_json = json.loads(line)
                self.asts.append(ast_json)
            self.ast_num = len(self.asts)
            f.close()

        for i in tqdm(range(self.ast_num)):
            self.asts[i][0].update({'layer': 0})
            for a in self.asts[i]:
                if 'children' in a:
                    for child_id in a['children']:
                        self.asts[i][child_id].update({'layer': a['layer'] + 1})

    def get_node_embedding(self):
        for i in tqdm(range(self.ast_num)):
            ast = self.asts[i]

            node_values = list()
            for node in ast:
                if 'value' in node.keys():
                    node_values.append(node['value'])
                else:
                    node_values.append('')

            node_types = [node['type'] for node in ast]

            node_list = list()
            for j in range(0, len(node_types)):
                if node_values[j] != '':
                    node_list.append(node_types[j] + '_' + node_values[j])
                else:
                    node_list.append(node_types[j])

            layer = [node['layer'] for node in ast]
            layer = [str(num) for num in layer]

            node_list = node_list[:self.max_node]
            node_list = [node_value[:self.max_value_len] for node_value in node_list]
            layer = layer[:self.max_node]

            matrix = self.bert_encoding.encode(node_list) + self.bert_encoding.encode(layer)
            matrix = np.array(matrix)
            matrix = sp.csr_matrix(matrix, dtype=np.float32)
            feature = torch.FloatTensor(np.array(matrix.todense()))
            if feature.size(0) > self.max_node:
                features = feature[0:self.max_node]
            else:
                features = torch.zeros(self.max_node, 768)
                for k in range(feature.size(0)):
                    features[k] = feature[k]

            self.node_embedding.append(features)

    def save_to_file(self, output_dir_path):
        torch.save(self.node_embedding, os.path.join(output_dir_path, "node_embedding.pt"))

    def run(self, ast_file_path, output_file_path):
        self.clear()
        print("Reading the ast file...")
        self.read_ast_file(ast_file_path)
        print("Generating the node embedding...")
        self.get_node_embedding()
        print("Saving files...")
        self.save_to_file(output_file_path)
