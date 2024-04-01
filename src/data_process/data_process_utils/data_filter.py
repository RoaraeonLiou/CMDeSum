import os
from tqdm import tqdm
from src.data_process.data_process_utils.data_cleaner import DataCleaner
import json
from src.data_process.data_process_utils.aster import Aster


class DataFilter(object):
    def __init__(self, language='java', split_method=None):
        language = language.lower()
        assert language in {'java', 'python'}
        self.language = language
        if split_method is None:
            split_method = ["train", "test", "valid"]
        self.split_method = split_method
        self.data_types = ["raw", "denoised", "cleaned"]
        self.path = dict()
        self.project_path = "path/of/project"
        self.raw_path = ""
        self.output_path = ""
        self.aster = Aster(self.language)

    def _generate_path(self, dataset_name):
        self.raw_path = os.path.join(self.project_path, "raw_data", dataset_name, self.language)
        self.output_path = os.path.join(self.project_path, "processed_data", dataset_name, self.language)

        for split in self.split_method:
            self.path[split] = dict()
            temp_path = os.path.join(self.output_path, split, 'tmp')
            if not os.path.exists(temp_path):
                os.makedirs(temp_path)
            self.path[split]["raw"] = os.path.join(self.raw_path, split, self.language + '_' + split + '.jsonl')
            self.path[split]["denoised"] = os.path.join(temp_path, "denoised.json")
            self.path[split]["cleaned"] = os.path.join(self.output_path, split, "cleaned.json")

    def _clean_noise_data(self):
        for key in self.path.keys():
            print("processing the {} data".format(key))
            # Create the data cleaner
            data_cleaner = DataCleaner(self.path[key]["raw"], "original_string", "docstring", self.language)
            # Get the clean data item flag list
            clean_flag_list = data_cleaner.get_clean_data()
            # Count the clean data items and the dirty data items
            clean = 0
            dirty = 0
            for flag in clean_flag_list:
                if flag:
                    clean += 1
                else:
                    dirty += 1
            print("\tclean:", clean)
            print("\tdirty:", dirty)
            for k in data_cleaner.noisy_data.keys():
                print("\t\t"+k+": ", len(data_cleaner.noisy_data[k]))
            data_cleaner.generate_clean_data(self.path[key]["denoised"])
            # Count the num of lines in output file
            with open(self.path[key]["denoised"], "r") as f:
                json_lines = f.readlines()
            print("\toutput file:", len(json_lines))

    def _clean_bad_data(self):
        for key in self.path.keys():
            print("processing the {} data".format(key))
            saved_flags = self._clean_the_data_cannot_generate_ast(
                input_path=self.path[key]["denoised"],
                output_path=self.path[key]["cleaned"],
                code_key_name="code"
            )
            saved = 0
            discarded = 0
            for flag in saved_flags:
                if flag:
                    saved += 1
                else:
                    discarded += 1
            print("\tsaved:", saved)
            print("\tdiscarded:", discarded)
            with open(self.path[key]["cleaned"], "r") as f:
                json_lines = f.readlines()
            print("\toutput file:", len(json_lines))

    def _clean_the_data_cannot_generate_ast(self, input_path, output_path, code_key_name):
        with open(input_path, 'r') as fr:
            lines = fr.readlines()
        data_size = len(lines)
        save_flag = list()
        with open(output_path, 'w+') as fw:
            for i in tqdm(range(data_size)):
                json_line = json.loads(lines[i])
                code = json_line[code_key_name].strip()
                ast_if_flag = self.aster.ast_check(code)
                save_flag.append(ast_if_flag)
                if ast_if_flag:
                    fw.write(lines[i])
        return save_flag

    def run(self, dataset_name):
        print("Start filtering the data...")
        self._generate_path(dataset_name)

        print("Cleaning the noise data...")
        self._clean_noise_data()

        print("Cleaning the data can not generate AST...")
        self._clean_bad_data()
        print("Data filtering is complete.")
        print("Please use Lucene for similar code matching before generating the final feature file.")


if __name__ == "__main__":
    data_path = {
        "train": {
            "raw": "./../../raw_data/java/train/java_train.jsonl",
            "denoised": "./../../processed_data/java/train/csn.java.denoised.train",
            "cleaned": "./../../processed_data/java/train/cleaned.json",
        },
        "valid": {
            "raw": "./../../raw_data/java/valid/java_valid.jsonl",
            "denoised": "./../../processed_data/java/valid/csn.java.denoised.valid",
            "cleaned": "./../../processed_data/java/valid/cleaned.json",
        },
        "test": {
            "raw": "./../../raw_data/java/test/java_test.jsonl",
            "denoised": "./../../processed_data/java/test/csn.java.denoised.test",
            "cleaned": "./../../processed_data/java/test/cleaned.json",
        },
    }
