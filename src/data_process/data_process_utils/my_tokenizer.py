import json
import os.path
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm


class MyTokenizer(object):
    def __init__(self):
        self.CHECK_IF_HAS_UPPER_CASE = re.compile("[A-Z._]")
        self.SPLIT_TOKEN = re.compile("[^\da-zA-Z]")
        self.stopwords = set(stopwords.words("english"))
        self.exclude_list = ["t"]
        for word in self.exclude_list:
            self.stopwords.remove(word)
        self.paths = dict()

    def _remove_stopwords_and_tokenized(self, src_string, return_str=False):
        tokens = word_tokenize(src_string)
        filtered_tokens = [word for word in tokens if word.lower() not in self.stopwords]
        if return_str:
            filtered_string = " ".join(filtered_tokens)
            return filtered_string
        else:
            return filtered_tokens

    def _to_words(self, token):
        if not self.CHECK_IF_HAS_UPPER_CASE.search(token):
            return [token]

        words = []
        word = ''


        letters = self.SPLIT_TOKEN.split(token)
        for sub_str in letters:
            part_len = len(sub_str)
            word = ''
            if not sub_str:
                continue

            for index, char in enumerate(sub_str):

                if index == part_len - 1:

                    if char.isupper() and sub_str[index - 1].islower():
                        if word: words.append(word)
                        words.append(char)
                        word = ''
                        continue

                elif index != 0 and char.isupper():
                    if (sub_str[index - 1].islower() and sub_str[index + 1].isalpha()) or (
                            sub_str[index - 1].isalpha() and sub_str[index + 1].islower()):
                        if word:
                            words.append(word)
                        word = ''
                word += char
            if word:
                words.append(word)
        return [word for word in words if word != '']

    def tokenize(self, source_string: str, return_type="string"):
        """
        Tokenize the source string.
        :param source_string: The string needed to be tokenized.
        :param return_type: The type of return value.
        :return: The token list.
        """
        return_type = return_type.lower()
        assert return_type in {"string", "list"}
        token_list = self._remove_stopwords_and_tokenized(source_string)
        split_token_list = list()
        for token in token_list:
            split_token_list.extend(self._to_words(token))
        if return_type == "string":
            return " ".join(split_token_list)
        elif return_type == "list":
            return split_token_list

    def generate_tokenized_code_file(self, input_path, output_path, code_key_name="cleaned_code"):
        with open(input_path, "r") as f:
            lines = f.readlines()

        with open(output_path, "w+") as w:
            for i in tqdm(range(len(lines))):
                json_obj = json.loads(lines[i])
                code = None
                if code_key_name in json_obj:
                    code = json_obj[code_key_name]
                else:
                    code = json_obj["code"]
                output_token_list = self.tokenize(code, return_type="list")
                w.write(" ".join(output_token_list) + "\n")

    def generate_tokenized_code_files(self, project_path, dataset_name='csn', language="java", split_method=None,
                                      code_key_name="cleaned_code"):
        if self.paths == dict():
            self.generate_paths(project_path, dataset_name, language, split_method)
        print("Generating the tokenized code file...")
        for key in self.paths.keys():
            print("Processing the " + key + " file...")
            self.generate_tokenized_code_file(self.paths[key]["input"], self.paths[key]["output"], code_key_name)
            print("Output to " + self.paths[key]["output"])
        print("Done...")

    def generate_paths(self, project_path, dataset_name='csn', language="java", split_method=None):
        if split_method is None:
            split_method = ["train", "valid", "test"]
        language = language.lower()
        assert language in {"java", "python"}

        self.paths = dict()
        for split in split_method:
            self.paths[split] = dict()
            base_path = os.path.join(project_path, "processed_data", dataset_name, language, split)
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            self.paths[split]["input"] = os.path.join(base_path, "cleaned.json")
            tmp_path = os.path.join(project_path, "processed_data", dataset_name, language, split, "tmp")
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)
            self.paths[split]["output"] = os.path.join(tmp_path, "code.token")


if __name__ == "__main__":
    pass
