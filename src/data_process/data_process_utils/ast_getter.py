import collections
import javalang
from tqdm import tqdm
from javalang import tree, ast
from collections import deque
from treelib import Tree, Node
import ast
import json
from io import StringIO


class JavaASTGetter(object):
    def __init__(self):
        """
        Code AST Getter
        """
        pass

    def _get_name(self, obj):
        if type(obj).__name__ in ['list', 'tuple']:
            a = []
            for i in obj:
                a.append(self._get_name(i))
            return a
        elif type(obj).__name__ in ['dict', 'OrderedDict']:
            a = {}
            for k in obj:
                a[k] = self._get_name(obj[k])
            return a
        elif type(obj).__name__ not in ['int', 'float', 'str', 'bool']:
            return type(obj).__name__
        else:
            return obj

    def _process_source(self, file_name, save_file, code_key_name):
        with open(file_name, 'r', encoding='utf-8') as source:
            lines = source.readlines()
        with open(save_file, 'w+', encoding='utf-8') as save:
            lines_num = len(lines)
            for i in tqdm(range(lines_num)):
                data_item = json.loads(lines[i])
                code = data_item[code_key_name].strip()
                # code = line.strip()
                tokens = list(javalang.tokenizer.tokenize(code))
                tks = []
                for tk in tokens:
                    if tk.__class__.__name__ == 'String' or tk.__class__.__name__ == 'Character':
                        tks.append('STR_')
                    elif 'Integer' in tk.__class__.__name__ or 'FloatingPoint' in tk.__class__.__name__:
                        tks.append('NUM_')
                    elif tk.__class__.__name__ == 'Boolean':
                        tks.append('BOOL_')
                    else:
                        tks.append(tk.value)
                save.write(" ".join(tks) + '\n')

    def _get_ast(self, file_name, w):
        exclude_items_id_list = list()
        with open(file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(w, 'w+', encoding='utf-8') as wf:
            ign_cnt = 0
            lines_num = len(lines)
            for index in tqdm(range(lines_num)):
                code = lines[index].strip()
                tokens = javalang.tokenizer.tokenize(code)
                token_list = list(javalang.tokenizer.tokenize(code))
                length = len(token_list)
                parser = javalang.parser.Parser(tokens)
                try:
                    tree = parser.parse_member_declaration()
                except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
                    # print(code)
                    # return None
                    continue
                flatten = []
                for path, node in tree:
                    flatten.append({'path': path, 'node': node})

                ign = False
                outputs = []
                stop = False
                for i, Node in enumerate(flatten):
                    d = collections.OrderedDict()
                    path = Node['path']
                    node = Node['node']
                    children = []
                    for child in node.children:
                        child_path = None
                        if isinstance(child, javalang.ast.Node):
                            child_path = path + tuple((node,))
                            for j in range(i + 1, len(flatten)):
                                if child_path == flatten[j]['path'] and child == flatten[j]['node']:
                                    children.append(j)
                        if isinstance(child, list) and child:
                            child_path = path + (node, child)
                            for j in range(i + 1, len(flatten)):
                                if child_path == flatten[j]['path']:
                                    children.append(j)
                    d["id"] = i
                    d["type"] = self._get_name(node)
                    if children:
                        d["children"] = children
                    value = None
                    if hasattr(node, 'name'):
                        value = node.name
                    elif hasattr(node, 'value'):
                        value = node.value
                    elif hasattr(node, 'position') and node.position:
                        for i, token in enumerate(token_list):
                            if node.position == token.position:
                                pos = i + 1
                                value = str(token.value)
                                while (pos < length and token_list[pos].value == '.'):
                                    value = value + '.' + token_list[pos + 1].value
                                    pos += 2
                                break
                    elif type(node) is javalang.tree.This \
                            or type(node) is javalang.tree.ExplicitConstructorInvocation:
                        value = 'this'
                    elif type(node) is javalang.tree.BreakStatement:
                        value = 'break'
                    elif type(node) is javalang.tree.ContinueStatement:
                        value = 'continue'
                    elif type(node) is javalang.tree.TypeArgument:
                        value = str(node.pattern_type)
                    elif type(node) is javalang.tree.SuperMethodInvocation \
                            or type(node) is javalang.tree.SuperMemberReference:
                        value = 'super.' + str(node.member)
                    elif type(node) is javalang.tree.Statement \
                            or type(node) is javalang.tree.BlockStatement \
                            or type(node) is javalang.tree.ForControl \
                            or type(node) is javalang.tree.ArrayInitializer \
                            or type(node) is javalang.tree.SwitchStatementCase \
                            or type(node) is javalang.tree.LambdaExpression:
                        value = 'None'
                    elif type(node) is javalang.tree.VoidClassReference:
                        value = 'void.class'
                    elif type(node) is javalang.tree.SuperConstructorInvocation:
                        value = 'super'
                    elif hasattr(node, 'member') and node.member:
                        if isinstance(node.member, str) and node.member == "new":
                            value = "new"

                    if value is not None and type(value) is type('str'):
                        d['value'] = value
                    if not children and not value:
                        # print('Leaf has no value!')
                        print(type(node))
                        print(code)
                        ign = True
                        ign_cnt += 1
                        # break
                    outputs.append(d)
                if not ign:
                    ast_json = json.dumps(outputs)
                    temp_ast = json.loads(ast_json)
                    if check(temp_ast):
                        wf.write(json.dumps(outputs))
                        wf.write('\n')
                    else:
                        exclude_items_id_list.append(index)
        print(ign_cnt)
        return exclude_items_id_list

    def get_ast(self, tmp_file_path, output_file_path):
        return self._get_ast(tmp_file_path, output_file_path)

    def run(self, data_file_path, tmp_file_path, output_file_path, code_key_name="code"):
        self._process_source(data_file_path, tmp_file_path, code_key_name)
        return self._get_ast(tmp_file_path, output_file_path)


def _sort_list(l):
    id_data = dict()
    for each in l:
        id_data[each['id']] = each
    l.clear()
    l = [id_data[i] for i in range(len(id_data))]
    return l


class GlobalTree(object):
    id = 0
    tree = Tree()

    def clear(self):
        self.id = 0
        self.tree = Tree()


def check(temp_ast):
    try:
        temp_ast[0].update({'layer': 0})
        for a in temp_ast:
            if 'children' in a:
                for child_id in a['children']:
                    temp_ast[child_id].update({'layer': a['layer'] + 1})
        return True
    except KeyError as e:
        return False


class PythonASTGetter(object):
    def __init__(self):
        self.global_tree = GlobalTree()

    def _add_main_node(self, node):
        tag = node.__class__.__name__
        node_id = self.global_tree.id
        data = node
        if self.global_tree.tree.root is None:
            self.global_tree.tree.create_node(tag=tag, identifier=node_id, data=data)
            self.global_tree.id += 1

    def _create_tree(self, node):
        todo = deque([node])
        while todo:
            node = todo.popleft()
            todo.extend(ast.iter_child_nodes(node))
            self._add_main_node(node)
            for i in ast.iter_child_nodes(node):
                self._add_node(i, node)

    def _add_node(self, child, parent):
        tag = child.__class__.__name__
        node_id = self.global_tree.id
        data = child

        # parent_hash = parent
        # if not global_tree.tree.get_node(child) and child.__class__.__name__!='Load':
        def find_id(tree, parent):
            for i in tree.expand_tree():
                if tree[i].data == parent:
                    return i

        parent_id = find_id(self.global_tree.tree, parent)
        # if child.__class__.__name__ != 'Load':
        self.global_tree.tree.create_node(tag=tag, identifier=node_id, data=data, parent=parent_id)
        self.global_tree.id += 1

    def _tree2json(self):
        node_msg_list = []
        for node_id in self.global_tree.tree.expand_tree(mode=self.global_tree.tree.DEPTH):
            node = self.global_tree.tree[node_id]
            node_attr = dict()
            tree_id = self.global_tree.tree.identifier
            node_attr['id'] = node_id
            node_attr['type'] = node.data.__class__.__name__

            if len(node.successors(tree_id)):
                node_attr['children'] = node.successors(tree_id)
            for i in ast.iter_fields(node.data):
                if isinstance(i[1], (int, str)):
                    node_attr['value'] = str(i[1])
            node_msg_list.append(node_attr)
        node_msg_list = _sort_list(node_msg_list)
        node_msg_list = json.dumps(node_msg_list)
        return node_msg_list

    def run(self, data_file_path, tmp_file_path, output_file_path, code_key_name="code"):
        exclude_items_id_list = list()
        with open(data_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [json.loads(line)[code_key_name] for line in lines]
        buf = StringIO()
        error = {}
        success = []

        # for n, line in enumerate(lines):
        for i in tqdm(range(len(lines))):
            line = lines[i]
            try:
                code = line.replace('\\n', '\n')
                code = code.replace('\\t', '\t')
                ast_module = ast.parse(code)
                self._create_tree(ast_module)
                json_data = self._tree2json()
                self.global_tree.clear()
                temp_ast = json.loads(json_data)
                if check(temp_ast):
                    buf.write(json_data)
                    buf.write('\n')
                    success.append(i)
                else:
                    exclude_items_id_list.append(i)
            except SyntaxError as e:
                print(e)
                print(i, 'SyntaxError', line)
                error[i] = line[:50] + '....'

        with open(output_file_path, 'w') as f:
            f.write(buf.getvalue())
        buf.close()
        return exclude_items_id_list


if __name__ == '__main__':

    pass
