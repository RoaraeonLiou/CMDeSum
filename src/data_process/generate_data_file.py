from data_process_utils.data_file_generator import DataFileGenerator


def generate_java_file():
    java_generator = DataFileGenerator(language='java')
    java_generator.run(data_set_name="csn")


def generate_python_file():
    python_generator = DataFileGenerator(language='python')
    python_generator.run(data_set_name="csn")


if __name__ == "__main__":
    # print(os.getcwd())
    # d = DataFileGenerator(["code"])
    # d.test("csn")
    # generate_python_file()
    generate_java_file()
    pass
