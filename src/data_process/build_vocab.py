import pickle
from collections import Counter
from tqdm import tqdm


def build_vocab_pkl(config):
    name, code_vocab_size, comment_vocab_size, max_code_len, max_comment_len = config.values()
    file_list = ['train', 'valid', 'test']
    code_token_dict, comment_token_dict = {}, {}
    code_len = []
    comment_len = []
    for file in file_list:
        with open(fr'./../../../processed_data/{name}/{file}/final.code', 'r') as f:
            code_lines = f.readlines()

        with open(fr'./../../../processed_data/{name}/{file}/final.comment', 'r') as f:
            comment_lines = f.readlines()

        for code_line, comment_line in tqdm(zip(code_lines, comment_lines)):
            code_data = code_line.strip()
            code_len.append(len(code_data.split(' ')))
            for token in code_data.split(' ')[:max_code_len]:
                if not token.isspace():
                    if code_token_dict.get(token) is None:
                        code_token_dict[token] = 1
                    else:
                        code_token_dict[token] += 1

            comment_data = comment_line.strip()
            comment_len.append(len(comment_data.split(' ')))
            for token in comment_data.split(' ')[:max_comment_len]:
                if not token.isspace():
                    if comment_token_dict.get(token) is None:
                        comment_token_dict[token] = 1
                    else:
                        comment_token_dict[token] += 1

    print("num_code_token:", len(code_token_dict), "num_comment_token:", len(comment_token_dict))
    print(max(comment_len))
    print(max(code_len))
    print(sum(comment_len) / len(comment_len))
    print(sum(code_len) / len(code_len))

    code_vocab = [tu[0] for tu in Counter(code_token_dict).most_common(code_vocab_size - 4)]
    comment_vocab = [tu[0] for tu in Counter(comment_token_dict).most_common(comment_vocab_size - 4)]
    print("=================comment token=================")
    for k in comment_vocab[:11]:
        print(k, comment_token_dict[k])
    print("=================code token=================")
    for k in code_vocab[:11]:
        print(k, code_token_dict[k])

    print("=================comment token=================")
    for k in comment_vocab[len(comment_vocab) - 11:]:
        print(k, comment_token_dict[k])
    print("=================code token=================")
    for k in code_vocab[len(code_vocab) - 11:]:
        print(k, code_token_dict[k])

    code_vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"] + code_vocab
    comment_vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"] + comment_vocab
    print("num_code_vocab:", len(code_vocab), "num_comment_vocab:", len(comment_vocab))

    code_word2id = {word: idx for idx, word in enumerate(code_vocab)}
    code_id2word = {idx: word for idx, word in enumerate(code_vocab)}
    comment_word2id = {word: idx for idx, word in enumerate(comment_vocab)}
    comment_id2word = {idx: word for idx, word in enumerate(comment_vocab)}

    with open(fr'./../../../data_embedding/{name}/code.word2id', 'wb') as w:
        pickle.dump(code_word2id, w)

    with open(fr'./../../../data_embedding/{name}/code.id2word', 'wb') as w:
        pickle.dump(code_id2word, w)

    with open(fr'./../../../data_embedding/{name}/comment.word2id', 'wb') as w:
        pickle.dump(comment_word2id, w)

    with open(fr'./../../../data_embedding/{name}/comment.id2word', 'wb') as w:
        pickle.dump(comment_id2word, w)


if __name__ == '__main__':
    java = {'name': 'csn/java', 'code_vocab_size': 60000, 'comment_vocab_size': 60000,
            'max_code_len': 300, 'max_comment_len': 50}
    python = {'name': 'csn/python', 'code_vocab_size': 60000, 'comment_vocab_size': 60000,
              'max_code_len': 100, 'max_comment_len': 50}
    build_vocab_pkl(java)
