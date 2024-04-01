from tqdm import tqdm


class SimPairSplitter(object):
    def __init__(self):
        self.from_side_list = list()
        self.to_side_list = list()
        self.exclude_set = set()

    def _parse_lucene_result(self, lucene_result_file_path):
        with open(lucene_result_file_path, 'r') as f:
            lines = f.readlines()
            pair_info = list()
            for line in lines:
                pair_info.append(int(line))

            for i in range(len(pair_info)):
                if pair_info[i] != -1:
                    if i not in self.exclude_set:
                        self.from_side_list.append(i)
                        self.to_side_list.append(pair_info[i])
                        self.exclude_set.add(i)
                        self.exclude_set.add(pair_info[i])

    def _generate_data_file(self, data_file_path, output_data_file_path, output_sim_file_path):
        assert len(self.from_side_list) == len(self.to_side_list)
        assert len(self.from_side_list) != 0

        with open(data_file_path, "r") as fr:
            lines = fr.readlines()
            data_num = len(self.from_side_list)
            with open(output_data_file_path, "w+") as fw_1:
                with open(output_sim_file_path, "w+") as fw_2:
                    for i in tqdm(range(data_num)):
                        fw_1.write(lines[self.from_side_list[i]])
                        fw_2.write(lines[self.to_side_list[i]])
                    fw_2.close()
                fw_1.close()
            fr.close()

    def _clear(self):
        self.from_side_list.clear()
        self.to_side_list.clear()
        self.exclude_set.clear()

    def run(self, lucene_result_file_path, data_file_path, output_data_file_path, output_sim_file_path):
        self._clear()
        self._parse_lucene_result(lucene_result_file_path)
        self._generate_data_file(data_file_path, output_data_file_path, output_sim_file_path)
