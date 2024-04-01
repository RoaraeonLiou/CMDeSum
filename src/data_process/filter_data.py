import sys

path = '/path/to/project'
sys.path.append(path)

from data_process_utils.data_filter import DataFilter


def filter_python():
    print("python")
    data_filter = DataFilter('python')
    data_filter.run('csn')


def filter_java():
    print("java")
    data_filter = DataFilter('java')
    data_filter.run('csn')


if __name__ == '__main__':
    # filter_java()
    filter_python()
