import re

from src.data_process.data_process_utils.noise_detection import *
from tqdm import tqdm
from bs4 import BeautifulSoup
import random
import os
import json
import warnings


class DataCleaner(object):
    def __init__(self, path, code_key_name, comment_key_name, language='java'):
        """
        :param code_key_name:
        :param comment_key_name:
        """
        self.path = path
        assert os.path.exists(self.path) is True
        self.code_key_name = code_key_name
        self.comment_key_name = comment_key_name
        self.output_dict = dict()
        self.data_is_clean_id_list = list()
        self.clean_code_list = list()
        self.language = language
        self.noisy_data = {'ContentTamper': [], 'NonLiteral': [], 'Interrogation': [], 'UnderDevelop': [],
                           'EmptyFunc': [], 'CommentOut': [], 'BlockComment': [], 'AutoCode': [], 'DuplicatedCode': [],
                           'OnlyParamComment': [], "TooLongCode": []}

    def get_clean_data(self):
        with open(self.path, "r") as f:
            json_lines = f.readlines()

        raw_code_list = list()
        raw_comment_list = list()
        data_size = len(json_lines)
        for line in json_lines:
            example_json = json.loads(line.strip())
            raw_code_list.append(example_json[self.code_key_name])
            raw_comment_list.append(example_json[self.comment_key_name])


        for i in tqdm(range(data_size)):
            raw_code = raw_code_list[i]
            raw_comment = raw_comment_list[i]
            first_sentence = get_first_sentence(raw_comment)
            updated_comment = self.update_ContentTamper(first_sentence)
            updated_code = self.update_BlockComment(raw_code)
            self.clean_code_list.append(updated_code)

            if updated_comment != first_sentence:
                self.noisy_data['ContentTamper'].append((raw_code, raw_comment))
            if len(updated_code) > 40000:
                self.noisy_data['TooLongCode'].append((raw_code, raw_comment))
                self.data_is_clean_id_list.append(False)
                continue
            # remove rules
            if if_ContentTamper(updated_comment):
                self.noisy_data['ContentTamper'].append((raw_code, raw_comment))
                self.data_is_clean_id_list.append(False)
                continue
            if if_NonLiteral(updated_comment):
                self.noisy_data['NonLiteral'].append((raw_code, raw_comment))
                self.data_is_clean_id_list.append(False)
                continue
            if if_Interrogation(updated_comment):
                self.noisy_data['Interrogation'].append((raw_code, raw_comment))
                self.data_is_clean_id_list.append(False)
                continue
            if if_UnderDevelop(updated_comment):
                self.noisy_data['UnderDevelop'].append((raw_code, raw_comment))
                self.data_is_clean_id_list.append(False)
                continue
            if if_AutoCode_by_comment(updated_comment, raw_comment):
                self.noisy_data['AutoCode'].append((raw_code, raw_comment))
                self.data_is_clean_id_list.append(False)
                continue
            if if_only_param_comment(updated_comment, self.language):
                self.noisy_data['OnlyParamComment'].append((raw_code, raw_comment))
                self.data_is_clean_id_list.append(False)
                continue

            if if_CommentedOut(raw_code, self.language):
                self.noisy_data['CommentOut'].append((raw_code, raw_comment))
                self.data_is_clean_id_list.append(False)
                continue

            if updated_code != raw_code:
                self.noisy_data['BlockComment'].append((raw_code, raw_comment))
            if if_AutoCode_by_code(updated_code, self.language):
                self.noisy_data['AutoCode'].append((raw_code, raw_comment))
                self.data_is_clean_id_list.append(False)
                continue
            if if_EmptyFunc(updated_code, self.language):
                self.noisy_data['EmptyFunc'].append((raw_code, raw_comment))
                self.data_is_clean_id_list.append(False)
                continue

            if self.output_dict.get(updated_code) is None:
                self.output_dict[updated_code] = [updated_comment]
                self.data_is_clean_id_list.append(True)
            else:
                self.output_dict[updated_code].append(updated_comment)
                self.data_is_clean_id_list.append(False)

        for updated_code in self.output_dict:
            updated_comment_list = self.output_dict[updated_code]
            if len(updated_comment_list) > 1:
                self.noisy_data['DuplicatedCode'].append((updated_code, updated_comment_list))

        return self.data_is_clean_id_list

    def get_noisy_data(self):
        return self.noisy_data

    def update_BlockComment(self, raw_code):
        new_list = list()
        rows = raw_code.split('\n')
        p_comment = None
        p_start = None
        p_end = None
        p_line = None
        if self.language == 'java':
            p_comment = re.compile(r'^(\s*//)|(//)')
            p_start = re.compile(r'^(/\*)|(\s*/\*)')
            p_end = re.compile(r'(\*/)|(\*/\s*)')
            p_line = re.compile(r'^(\s*/\*)[\w\W]*(\*/\s*)$')

        elif self.language == 'python':
            p_start = re.compile(r'^(\s*""")')
            p_end = re.compile(r'("""\s*)$')
            p_comment = re.compile(r'^(\s*#)')
            p_line = re.compile(r'^(\s*""")[\w\W]*("""\s*)$')

        i = 0
        while i < len(rows):
            if p_line.search(rows[i]):
                i += 1
            elif p_start.search(rows[i]):
                j = i + 1
                while j < len(rows) and not p_end.search(rows[j]):
                    j += 1
                if j == len(rows):
                    j = i
                i = j + 1
            elif p_comment.search(rows[i]):
                i += 1
            else:
                new_list.append(rows[i])
                i += 1
        ret = "\n".join(new_list)
        return ret


    def update_ContentTamper(self, comment):
        warnings.filterwarnings("ignore")
        return BeautifulSoup(comment, "html.parser").get_text()

    def generate_clean_data(self, output_path):
        if not self.data_is_clean_id_list:
            self.get_clean_data()

        with open(self.path, "r") as f:
            json_lines = f.readlines()


        assert len(self.data_is_clean_id_list) == len(json_lines) == len(self.clean_code_list)

        with open(output_path, "w+") as w:
            for i in range(len(json_lines)):
                if self.data_is_clean_id_list[i]:
                    example_json = json.loads(json_lines[i].strip())
                    example_json["cleaned_code"] = self.clean_code_list[i]
                    if len(self.output_dict[self.clean_code_list[i]]) > 1:
                        example_json["cleaned_comment"] = random.choice(self.output_dict[self.clean_code_list[i]])
                    else:
                        example_json["cleaned_comment"] = self.output_dict[self.clean_code_list[i]][0]
                    w.write(json.dumps(example_json))
                    w.write('\n')


if __name__ == '__main__':
    pass
