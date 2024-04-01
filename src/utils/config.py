import json
import os
import pickle


class Config(object):
    def __init__(self, path, dataset_config):
        dataset, max_code_len, max_comment_len, max_keywords_len = dataset_config.values()
        with open(path, 'r') as f:
            param_dict = json.load(f)
        self.cuda = True
        self.dataset = dataset
        self.d_model = param_dict['d_model']
        self.d_ff = param_dict['d_ff']
        self.head_num = param_dict['head_num']
        self.encoder_layer_num = param_dict['encoder_layer_num']
        self.decoder_layer_num = param_dict['decoder_layer_num']

        self.beam_width = param_dict['beam_width']
        self.lr = param_dict['lr']
        self.fineTune_lr = param_dict['fineTune_lr']
        self.batch_size = param_dict['batch_size']
        self.max_iter_num = param_dict['max_iter_num']
        self.dropout = param_dict['dropout']
        self.epochs = param_dict['epochs']
        self.clipping_distance = param_dict['clipping_distance']

        self.base_path = param_dict['base_path']
        self.code_word2id_path = os.path.join(self.base_path, "data_embedding/csn", self.dataset, "code.word2id")
        self.code_id2word_path = os.path.join(self.base_path, "data_embedding/csn", self.dataset, "code.id2word")
        self.comment_word2id_path = os.path.join(self.base_path, "data_embedding/csn", self.dataset, "comment.word2id")
        self.comment_id2word_path = os.path.join(self.base_path, "data_embedding/csn", self.dataset, "comment.id2word")
        with open(self.code_word2id_path, 'rb') as f:
            code_word2id = pickle.load(f)
        with open(self.code_id2word_path, 'rb') as f:
            code_id2word = pickle.load(f)
        with open(self.comment_word2id_path, 'rb') as f:
            comment_word2id = pickle.load(f)
        with open(self.comment_id2word_path, 'rb') as f:
            comment_id2word = pickle.load(f)
        self.code_word2id = code_word2id
        self.code_id2word = code_id2word
        self.comment_word2id = comment_word2id
        self.comment_id2word = comment_id2word
        self.bos_token = self.comment_word2id['<BOS>']
        self.eos_token = self.comment_word2id['<EOS>']
        
        self.max_code_len = max_code_len
        self.max_comment_len = max_comment_len
        self.max_keywords_len = max_keywords_len
        self.code_vocab_size = len(code_word2id)
        self.comment_vocab_size = len(comment_word2id)
        self.input_dim = 768
        self.hidden_num = 768
        self.output_dim = 768
