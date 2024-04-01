from data_process_utils.my_tokenizer import MyTokenizer

if __name__ == "__main__":
    tokenizer = MyTokenizer()
    project_path = '/path/to/project'
    dataset_name = "csn"

    # tokenizer.generate_tokenized_code_files(project_path=project_path, dataset_name=dataset_name, language='python')
    tokenizer.generate_tokenized_code_files(project_path=project_path, dataset_name=dataset_name, language='java')
