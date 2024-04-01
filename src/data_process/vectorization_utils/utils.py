import json

from tqdm import tqdm


class NodeCounter(object):
    def __init__(self):
        self.asts = list()
        self.ast_num = 0
        self.max_node_num = 0
        self.min_node_num = 999
        self.node_num_counter = dict()
        self.percent = list()
        self.node_sum = 0

    def clear(self):
        self.asts.clear()
        self.ast_num = 0
        self.max_node_num = 0
        self.min_node_num = 999
        self.node_num_counter.clear()
        self.percent.clear()
        self.node_sum = 0

    def count(self, ast_file_path):
        with open(ast_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                ast_json = json.loads(line)
                self.asts.append(ast_json)
            self.ast_num = len(self.asts)
            f.close()

        for i in tqdm(range(self.ast_num)):
            node_num = len(self.asts[i])
            if node_num > self.max_node_num:
                self.max_node_num = node_num
            if node_num < self.min_node_num:
                self.min_node_num = node_num
            self.node_sum += node_num
            interval = node_num // 10
            if interval in self.node_num_counter:
                self.node_num_counter[interval] += 1
            else:
                self.node_num_counter[interval] = 1

    def print_res(self):
        print("max node num:", self.max_node_num)
        print("min node num:", self.min_node_num)
        print("avg node num:", self.node_sum/self.ast_num)
        intervals = list(self.node_num_counter.keys())
        intervals.sort()
        ast_counter = 0
        for interval in intervals:
            ast_counter += self.node_num_counter[interval]
            self.percent.append(ast_counter / self.ast_num)
        for i in range(len(intervals)):
            print(intervals[i] * 10, "-", (intervals[i] + 1) * 10, ":\t\t", self.node_num_counter[intervals[i]], ",\t",
                  "%.2f" % (self.percent[i] * 100))

    def run(self, ast_file_path):
        self.count(ast_file_path)
        self.print_res()


class ValueCounter(object):
    def __init__(self):
        self.asts = list()
        self.max_value_len_list = list()
        self.ast_num = 0
        self.node_num_counter = dict()
        self.percent = list()
        self.node_num_counter_2 = dict()
        self.percent_2 = list()
        self.node_num = 0
        self.value_len_sum = 0


    def clear(self):
        self.asts.clear()
        self.max_value_len_list.clear()
        self.node_num_counter.clear()
        self.percent.clear()
        self.node_num_counter_2.clear()
        self.percent_2.clear()
        self.value_len_sum = 0

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
            node_value_len = [len(node) for node in node_list]
            self.node_num += len(node_list)
            for node_len in node_value_len:
                self.value_len_sum += node_len
                interval_t = node_len // 10
                if interval_t in self.node_num_counter_2:
                    self.node_num_counter_2[interval_t] += 1
                else:
                    self.node_num_counter_2[interval_t] = 1
            max_node_value_len = max(node_value_len)
            self.max_value_len_list.append(max_node_value_len)
            interval = max_node_value_len // 10
            if interval in self.node_num_counter:
                self.node_num_counter[interval] += 1
            else:
                self.node_num_counter[interval] = 1

    def print_res(self):
        print("max node value len:", max(self.max_value_len_list))
        print("min node value len:", min(self.max_value_len_list))
        print("avg node value len:", self.value_len_sum/self.ast_num)
        intervals = list(self.node_num_counter.keys())
        intervals.sort()
        ast_counter = 0
        for interval in intervals:
            ast_counter += self.node_num_counter[interval]
            self.percent.append(ast_counter / self.ast_num)
        for i in range(len(intervals)):
            print(intervals[i] * 10, "-", (intervals[i] + 1) * 10, ":\t\t", self.node_num_counter[intervals[i]], ",\t",
                  "%.2f" % (self.percent[i] * 100))

    def print_res_2(self):
        intervals = list(self.node_num_counter_2.keys())
        intervals.sort()
        node_counter = 0
        for interval in intervals:
            node_counter += self.node_num_counter_2[interval]
            self.percent_2.append(node_counter / self.node_num)
        for i in range(len(intervals)):
            print(intervals[i] * 10, "-", (intervals[i] + 1) * 10, ":\t\t", self.node_num_counter_2[intervals[i]], ",\t",
                  "%.2f" % (self.percent_2[i] * 100))

    def run(self, ast_file_path):
        self.read_ast_file(ast_file_path)
        self.print_res()
        print("="*30)
        self.print_res_2()


if __name__ == "__main__":
    path = "./../../../processed_data/csn/python/train/final.ast"
    counter = NodeCounter()
    counter.run(path)
    # counter = ValueCounter()
    # counter.run(path)
