import json
import os
from tqdm import tqdm
from src.data_process.data_process_utils.aster import Aster
from src.data_process.data_process_utils.my_tokenizer import MyTokenizer
from src.data_process.data_process_utils.sim_pair_splitter import SimPairSplitter


class DataFileGenerator(object):
    def __init__(self, key_name_list=None, split_method=None, language='java'):
        if split_method is None:
            split_method = ["train", "valid", "test"]
        self.split_method = split_method
        if key_name_list is None:
            key_name_list = ["code", "func_name", "comment"]
        self.key_name_list = key_name_list
        self.language = language
        self.ast_getter = Aster(self.language)
        self.sim_pair_splitter = SimPairSplitter()
        self.tokenizer = MyTokenizer()
        self.project_path = "path/of/project"
        self.base_path = ""
        self.type_path = dict()
        self.cleaned_json_data_file_path = dict()
        self.lucene_result_file_path = dict()
        self.final_json_data_file_path = dict()
        self.sim_json_data_file_path = dict()
        self.tmp_file_path = dict()
        self.ast_file_path = dict()
        self.data_feature_file_path = dict()
        self.sim_data_feature_file_path = dict()

    def _generate_path(self, data_set_name):
        self.base_path = os.path.join(self.project_path, "processed_data", data_set_name, self.language)
        self.type_path = dict()
        self.cleaned_json_data_file_path = dict()
        self.lucene_result_file_path = dict()
        self.final_json_data_file_path = dict()
        self.sim_json_data_file_path = dict()
        self.tmp_file_path = dict()
        self.ast_file_path = dict()
        self.data_feature_file_path = dict()

        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        for data_type in self.split_method:
            self.type_path[data_type] = os.path.join(self.base_path, data_type)
            if not os.path.exists(os.path.join(self.type_path[data_type], 'tmp')):
                os.makedirs(os.path.join(self.type_path[data_type], 'tmp'))

        for data_type in self.split_method:
            self.cleaned_json_data_file_path[data_type] = os.path.join(self.type_path[data_type], "cleaned.json")
            self.lucene_result_file_path[data_type] = os.path.join(self.type_path[data_type], "tmp/sim.pair.res")
            self.final_json_data_file_path[data_type] = os.path.join(self.type_path[data_type], "final.json")
            self.sim_json_data_file_path[data_type] = os.path.join(self.type_path[data_type], "sim.json")
            self.tmp_file_path[data_type] = os.path.join(self.type_path[data_type], "tmp/tmp.code")
            self.ast_file_path[data_type] = os.path.join(self.type_path[data_type], "final.ast")
            self.data_feature_file_path[data_type] = dict()
            self.sim_data_feature_file_path[data_type] = dict()
            for key in self.key_name_list:
                self.data_feature_file_path[data_type][key] = os.path.join(self.type_path[data_type], "final." + key)
                self.sim_data_feature_file_path[data_type][key] = os.path.join(self.type_path[data_type], "sim." + key)

    def _data_generate(self, data_file_path, output_file_path_dir, exclude_list):
        """
        Generate data file according to the needed json key names.
        :param data_file_path: The path of input data file
        :param output_file_path_dir: The paths of output files, key is the needed json key name, and value is the path.
        :return: null
        """
        with open(data_file_path, "r") as fr:
            lines = fr.readlines()
            json_objs = list()
            data_num = len(lines)
            print("Loading json file...")
            for i in tqdm(range(data_num)):
                if i not in exclude_list:
                    json_objs.append(json.loads(lines[i]))
            data_num = len(json_objs)
            print("Done!")
            for key_name, path in output_file_path_dir.items():
                print("Writing " + key_name + " file...")
                json_key = key_name
                if "cleaned_" + json_key in json_objs[0]:
                    json_key = "cleaned_" + json_key

                with open(path, "w+") as fw:
                    for i in tqdm(range(data_num)):
                        write_str = self.tokenizer.tokenize(json_objs[i][json_key])
                        fw.write(write_str + "\n")
                    fw.close()
                print("Done!")
            fr.close()

    def test(self, data_set_name):
        self._generate_path(data_set_name)
        print(self.base_path)
        for data_type in self.split_method:
            print(self.type_path[data_type])
            print(self.cleaned_json_data_file_path[data_type])
            print(self.lucene_result_file_path[data_type])
            print(self.final_json_data_file_path[data_type])
            print(self.sim_json_data_file_path[data_type])
            print(self.tmp_file_path[data_type])
            print(self.ast_file_path[data_type])
            print(self.data_feature_file_path[data_type])

    def run(self, data_set_name, code_key_name="code"):
        """
        Run the DataFileGenerator.
        Notice there are something important!
        :param data_set_name:
        :param code_key_name:
        :return:
        """
        excludes = dict()
        # Step 1 : Construct the Path
        print("Constructing the file path...")
        self._generate_path(data_set_name)
        print("Done...")
        # Step 2 : Use SimPairSplitter and Lucene result file to generate data file and sim data file.
        print("Generating the final data file and sim date file...")
        for data_type in self.split_method:
            self.sim_pair_splitter.run(
                data_file_path=self.cleaned_json_data_file_path[data_type],
                lucene_result_file_path=self.lucene_result_file_path[data_type],
                output_data_file_path=self.final_json_data_file_path[data_type],
                output_sim_file_path=self.sim_json_data_file_path[data_type]
            )
        print("Done...")
        # Step 3 : Use ASTGetter generate AST file from data file and sim data file.
        print("Generating the AST from final data file...")
        for data_type in self.split_method:
            excludes[data_type] = self.ast_getter.get_ast_file(
                data_file_path=self.final_json_data_file_path[data_type],
                tmp_file_path=self.tmp_file_path[data_type],
                output_file_path=self.ast_file_path[data_type],
                code_key_name=code_key_name
            )
        print("Done...")
        # Step 4 : Use _data_generate method to generate other needed data feature file.
        print("Generating the data feature file from final data file and sim data file...")
        for data_type in self.split_method:
            self._data_generate(
                data_file_path=self.final_json_data_file_path[data_type],
                output_file_path_dir=self.data_feature_file_path[data_type],
                exclude_list=excludes[data_type],
            )
            self._data_generate(
                data_file_path=self.sim_json_data_file_path[data_type],
                output_file_path_dir=self.sim_data_feature_file_path[data_type],
                exclude_list=excludes[data_type],
            )
        print("Done...")
        for data_type in self.split_method:
            print(data_type, ":", excludes[data_type])
