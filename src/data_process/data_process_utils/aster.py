import ast
import json

import javalang
from tqdm import tqdm

from src.data_process.data_process_utils.ast_getter import JavaASTGetter, PythonASTGetter


class Aster(object):
    def __init__(self, language='java'):
        """
        Code AST Getter
        :param language: Code language choice, default is 'java', support only 'java' and 'python' now.
        """
        language = language.lower()
        assert language in {"java", "python"}
        self.language = language
        self.ast_getter = None
        if self.language == "java":
            self.ast_getter = JavaASTGetter()
        elif self.language == "python":
            self.ast_getter = PythonASTGetter()

    def ast_check(self, code):
        if self.language == 'java':
            tokens = javalang.tokenizer.tokenize(code)
            parser = javalang.parser.Parser(tokens)
            try:
                parser.parse_member_declaration()
                return True
            except Exception as e:
                return False
        elif self.language == 'python':
            code = code.replace('\\n', '\n')
            code = code.replace('\\t', '\t')
            try:
                ast.parse(code)
                return True
            except SyntaxError as e:
                return False

    def ast_file_check(self, ast_file_path):
        with open(ast_file_path, 'r') as fr:
            lines = fr.readlines()
            data_size = len(lines)
            normal = 0
            abnormal = 0
            for i in tqdm(range(data_size)):
                data_json = json.loads(lines[i])
                code = data_json["code"].strip()
                if self.ast_check(code):
                    normal += 1
                else:
                    abnormal += 1
        print("normal:", normal, "; abnormal:", abnormal)

    def get_ast_file(self, data_file_path, tmp_file_path, output_file_path, code_key_name):
        return self.ast_getter.run(data_file_path, tmp_file_path, output_file_path, code_key_name)
