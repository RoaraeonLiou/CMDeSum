import sys
import os
import argparse

path = '/path/to/project'
max_node = 100
max_value_len = 100
sys.path.append(path)
from vectorization_utils.node_embedding import NodeEmbedding
bert_server_ip = "xxx.xxx.xxx.xxx"

def generate_all():
    input_paths = {
        "java_valid": os.path.join(path, "processed_data/csn", "java", "valid/final.ast"),
        "python_valid": os.path.join(path, "processed_data/csn", "python", "valid/final.ast"),
        "java_test": os.path.join(path, "processed_data/csn", "java", "test/final.ast"),
        "python_test": os.path.join(path, "processed_data/csn", "python", "test/final.ast"),
        "java_train": os.path.join(path, "processed_data/csn", "java", "train/final.ast"),
        "python_train": os.path.join(path, "processed_data/csn", "python", "train/final.ast"),
    }
    output_paths = {
        "java_valid": os.path.join(path, "data_embedding/csn", "java", "valid"),
        "python_valid": os.path.join(path, "data_embedding/csn", "python", "valid"),
        "java_test": os.path.join(path, "data_embedding/csn", "java", "test"),
        "python_test": os.path.join(path, "data_embedding/csn", "python", "test"),
        "java_train": os.path.join(path, "data_embedding/csn", "java", "train"),
        "python_train": os.path.join(path, "data_embedding/csn", "python", "train"),
    }

    node_embedding = NodeEmbedding(max_node, bert_server_ip)
    for data_type in input_paths:
        print("========================================")
        print("Processing the " + data_type + " file...")
        node_embedding.run(input_paths[data_type], output_paths[data_type])
    print("Done...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', type=str, default=None)
    parser.add_argument('--o', type=str, default=None)
    args = parser.parse_args()
    adj_getter = NodeEmbedding(max_node, max_value_len, bert_server_ip)
    adj_getter.run(args.i, args.o)
