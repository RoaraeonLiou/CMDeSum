import json
from collections import Counter

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
import os


class MyDataset(Dataset):
    def __init__(self, root_path, code_word2id, comment_word2id, dataset, max_code_num, max_comment_len,
                 max_keywords_len, file):
        self.ids = list()
        self.code = list()
        self.comment = list()
        self.draft = list()
        self.func_name = list()
        self.ast_node = list()
        self.adj1 = list()
        self.adj2 = list()
        self.adj3 = list()
        self.adj4 = list()
        self.adj5 = list()

        self.max_code_num = max_code_num
        self.max_comment_len = max_comment_len
        self.max_keywords_len = max_keywords_len
        self.code_word2id = code_word2id
        self.comment_word2id = comment_word2id
        self.file = file
        # if self.file == 'train':
        #     self.file = 'test'
        print(self.file)
        base_path_one = os.path.join(root_path, "processed_data/csn", dataset, self.file)
        base_path_two = os.path.join(root_path, "data_embedding/csn", dataset, self.file)
        self.file = file
        print(self.file)
        # read code, comment, func name, sim comment
        code_path = os.path.join(base_path_one, "final.code")
        with open(code_path, 'r') as f:
            source_code_lines = f.readlines()

        comment_path = os.path.join(base_path_one, "final.comment")
        with open(comment_path, 'r') as f:
            comment_lines = f.readlines()

        draft_path = os.path.join(base_path_one, "sim.comment")
        with open(draft_path, 'r') as f:
            draft_lines = f.readlines()

        func_name_path = os.path.join(base_path_one, "final.func_name")
        with open(func_name_path, 'r') as f:
            func_name_lines = f.readlines()

        # read ast node, adj and its powers
        ast_path = os.path.join(base_path_two, "node_embedding.pt")
        adj1_path = os.path.join(base_path_two, "adj1.pt")
        adj2_path = os.path.join(base_path_two, "adj2.pt")
        adj3_path = os.path.join(base_path_two, "adj3.pt")
        adj4_path = os.path.join(base_path_two, "adj4.pt")
        adj5_path = os.path.join(base_path_two, "adj5.pt")

        ast_node_lines = torch.load(ast_path)
        adj1_lines = torch.load(adj1_path)
        adj2_lines = torch.load(adj2_path)
        adj3_lines = torch.load(adj3_path)
        adj4_lines = torch.load(adj4_path)
        adj5_lines = torch.load(adj5_path)

        count_id = 0
        for code, comment, draft, func_name, ast_node, adj1, adj2, adj3, adj4, adj5 in tqdm(
                zip(source_code_lines, comment_lines, draft_lines, func_name_lines, ast_node_lines, adj1_lines,
                    adj2_lines, adj3_lines, adj4_lines, adj5_lines)):

            """
            deal with ID
            """
            count_id += 1
            self.ids.append(count_id)

            """
            deal with code
            """
            code_token_list = code.strip().split(' ')
            source_code_list = [code_word2id[token] if token in code_word2id else code_word2id['<UNK>']
                                for token in code_token_list[:self.max_code_num]]
            while len(source_code_list) < self.max_code_num:
                source_code_list.append(code_word2id['<PAD>'])
            self.code.append(source_code_list)

            """
            deal with comment
            """
            if file != 'test':
                comment_token_list = comment.strip().split(' ')
                comment_list = [comment_word2id[token] if token in comment_word2id else comment_word2id['<UNK>']
                                for token in comment_token_list[:self.max_comment_len]] + [comment_word2id['<EOS>']]
                self.comment.append(comment_list)
            else:
                comment_token_list = comment.strip().split(' ')
                self.comment.append(comment_token_list)

            """
            deal with draft
            """
            draft_token_list = draft.strip().split(' ')
            draft_list = [comment_word2id[token] if token in comment_word2id else comment_word2id['<UNK>']
                          for token in draft_token_list[:self.max_comment_len]]
            self.draft.append(draft_list)

            """
            deal with func name
            """
            func_name_token_list = func_name.strip().split(' ')
            func_name_list = [comment_word2id[token] for token in func_name_token_list
                              if token in comment_word2id]
            self.func_name.append(func_name_list[:self.max_keywords_len])

            """
            deal with ast
            """
            self.ast_node.append(ast_node)
            # adj1 = [adj.float() for adj in adj1]
            # adj2 = [adj.float() for adj in adj2]
            # adj3 = [adj.float() for adj in adj3]
            # adj4 = [adj.float() for adj in adj4]
            # adj5 = [adj.float() for adj in adj5]
            self.adj1.append(adj1)
            self.adj2.append(adj2)
            self.adj3.append(adj3)
            self.adj4.append(adj4)
            self.adj5.append(adj5)

    def __getitem__(self, index):
        return self.code[index], self.comment[index], self.draft[index], self.func_name[index], self.ast_node[index], \
               self.adj1[index], self.adj2[index], self.adj3[index], self.adj4[index], self.adj5[index], \
               len(self.code[index]), len(self.comment[index]), len(self.draft[index]), len(self.func_name[index]), \
               len(self.ast_node[index]), len(self.adj1[index]), len(self.adj2[index]), len(self.adj3[index]), \
               len(self.adj4[index]), len(self.adj5[index]), self.ids[index]

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return_list = []
        for i in dat:
            if i < 10:
                if i == 1 and self.file == 'test':
                    return_list.append(dat[i].tolist())
                elif i < 4:
                    return_list.append(
                        pad_sequence([torch.tensor(x, dtype=torch.int64) for x in dat[i].tolist()], True))
                else:
                    # return_list.append(
                    #     pad_sequence([torch.tensor(x, dtype=torch.float) for x in dat[i].tolist()], True))
                    return_list.append(pad_sequence([x.clone().detach() for x in dat[i].tolist()], True))
            elif i < 20:
                if i == 11 and self.file == 'test':
                    return_list.append(dat[i].tolist())
                else:
                    return_list.append(torch.tensor(dat[i].tolist()))
            else:
                return_list.append(dat[i].tolist())
        return return_list
