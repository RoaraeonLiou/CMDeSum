import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import torch
from tqdm import tqdm
import random
from utils.utils import eval_bleu_rouge_meteor, MaskedSoftmaxCELoss, DAMSMLoss, get_parameter_number
from utils.config import Config
from CMDeSum import CMDeSum
from utils.dataloader_getter import DataloaderGetter
import json

seed = 12346

def seed_all(seed=seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def evaluate_model(model, dataloader, bos_token, common_id2word, cuda, max_iter_num):
    losses, comment_reference, ids = [], [], []
    comment_prediction = {i: [] for i in range(3 + 1)}
    model.eval()

    seed_all(seed)
    with torch.no_grad():
        for data in tqdm(dataloader):
            source_code, comment, draft, func_name, ast_node_embedding, adj_1, adj_2, adj_3, adj_4, adj_5, source_code_len, comment_len, draft_len, func_name_len, ast_node_embedding_len, adj_1_len, adj_2_len, adj_3_len, adj_4_len, adj_5_len = [
                d.cuda() if cuda and not isinstance(d, list) else d for d in data[:20]]
            code_id = data[-1]

            bos = torch.tensor([bos_token] * len(comment), device=draft.device).reshape(-1, 1)
            memory = model(source_code, bos, draft, func_name, source_code_len, comment_len, draft_len, func_name_len,
                           ast_node_embedding, adj_1, adj_2, adj_3, adj_4, adj_5, ast_node_embedding_len)

            for i in range(len(comment)):
                ref = comment[i]
                comment_reference.append([ref])

                for j, comment_pred in enumerate(memory):
                    pre = [common_id2word[id] for id in comment_pred[i]]
                    comment_prediction[j].append(pre)

            ids += code_id

    for ii, comment_pred in enumerate(comment_prediction.values()):
        assert len(ids) == len(comment_pred) == len(comment_reference)
        bleu, rouge, meteor, _, _ = eval_bleu_rouge_meteor(ids, comment_pred, comment_reference)
        print(bleu, rouge, meteor)

    return bleu, rouge, meteor, comment_prediction



if __name__ == '__main__':
    print("judge step two")
    # set config
    dataset_config_path = "./csn_java_config.json"
    with open(dataset_config_path, "r") as f:
        dataset_config = json.load(f)

    config_file_path = './hyper_parameters.json'
    config = Config(config_file_path, dataset_config)

    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    cuda = torch.cuda.is_available() and config.cuda
    if cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    # seed
    print("seed...")
    seed_all(seed)

    print("build model...")
    model = CMDeSum(config.batch_size, config.d_model, config.d_ff, config.head_num, config.encoder_layer_num,
                config.decoder_layer_num, config.code_vocab_size,
                config.comment_vocab_size, config.bos_token, config.eos_token, config.max_comment_len,
                config.clipping_distance, config.max_iter_num,
                config.dropout, 4, config.input_dim, config.hidden_num, config.output_dim)

    """
    load best model 
    """
    print("load the best model parameters!")
    model.load_state_dict(torch.load(f"./../saved_model/{config.dataset}/second_step_params.pkl"))

    if cuda:
        model.cuda()

    dataloader_getter = DataloaderGetter(config, data_type=['train', 'test'])
    loaders = dataloader_getter.get()
    train_loader, valid_loader = loaders[0], loaders[1]

    test_Bleu, test_Rouge, test_Meteor, test_prediction = \
        evaluate_model(model, valid_loader, config.bos_token, config.comment_id2word, cuda, config.max_iter_num)

    print('final_results: test_Bleu:{},test_Rouge:{},test_Meteor:{}'.
          format(test_Bleu, test_Rouge, test_Meteor))

    for ii, comment_pred in enumerate(test_prediction.values()):
        with open(f'./../results/{config.dataset}/beam_search_result.{ii}', 'w') as w:
            for comment_list in comment_pred:
                comment = ' '.join(comment_list)
                w.write(comment + '\n')
            print("save", ii)
