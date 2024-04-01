import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import torch
from tqdm import tqdm
from prettytable import PrettyTable
import random
import time
from torch import optim
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


def train(model, seq2seq_loss_function, evaluator_loss_function, dataloader, bos_token, optimizer, epoch, cuda,
          max_iter_num):
    loss_list = [list() for _ in range(2 + 1)]
    model.train()

    for data in tqdm(dataloader):
        source_code, comment, draft, func_name, ast_node_embedding, adj_1, adj_2, adj_3, adj_4, adj_5, source_code_len, comment_len, draft_len, func_name_len, ast_node_embedding_len, adj_1_len, adj_2_len, adj_3_len, adj_4_len, adj_5_len = [
            d.cuda() for d in data[:20]] if cuda else data[:20]
        bos = torch.tensor([bos_token] * comment.size(0), device=comment.device).reshape(-1, 1)

        comment_input = torch.cat([bos, comment[:, :-1]], 1)
        comment_input_len = torch.add(comment_len, -1)

        optimizer.zero_grad()

        memory, anchor, positive, negative = model(source_code, comment_input, draft, func_name, source_code_len,
                                                   comment_input_len,
                                                   draft_len, func_name_len, ast_node_embedding, adj_1, adj_2, adj_3,
                                                   adj_4, adj_5, ast_node_embedding_len)

        loss = None
        for iter_idx in range(2):
            loss_idx = seq2seq_loss_function(memory[iter_idx], comment, comment_len)
            loss_list[iter_idx].append(loss_idx.item())
            if loss is None:
                loss = loss_idx
            else:
                loss += loss_idx

        loss_e = evaluator_loss_function(anchor, positive, negative) * 0.1
        loss_list[-1].append(loss_e.item())

        loss += loss_e

        loss.backward()
        optimizer.step()

    avg_loss = [round(np.sum(losses) / len(losses), 4) for losses in loss_list]
    return avg_loss


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

    table = PrettyTable(['bleu', 'rouge', 'meteor'])
    for ii, comment_pred in enumerate(comment_prediction.values()):
        assert len(ids) == len(comment_pred) == len(comment_reference)
        bleu, rouge, meteor, _, _ = eval_bleu_rouge_meteor(ids, comment_pred, comment_reference)
        table.add_row([bleu, rouge, meteor])
    print(table)
    return bleu, rouge, meteor, comment_prediction


if __name__ == '__main__':
    print("two_pass step two")
    # set config
    dataset_config_path = "./csn_python_config.json"
    with open(dataset_config_path, "r") as f:
        dataset_config = json.load(f)

    config_file_path = './hyper_parameters.json'
    config = Config(config_file_path, dataset_config)
    print("dataset: ", dataset_config['name'])
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
                config.dropout, None, config.input_dim, config.hidden_num, config.output_dim)
    print(model.model_name)
    """
    load best model 
    """
    print("load the best model parameters!")
    model.load_state_dict(torch.load(f"./../saved_model/{config.dataset}/first_judge_params.pkl"))

    if cuda:
        model.cuda()

    # set loss function
    seq2seq_loss = MaskedSoftmaxCELoss()
    evaluator_loss = DAMSMLoss()

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.fineTune_lr)

    print(get_parameter_number(model))

    # get dataloader
    print("get dataloader...")
    dataloader_getter = DataloaderGetter(config)
    loaders = dataloader_getter.get()
    train_loader, test_loader = loaders[0], loaders[1]

    # train and eval
    print("train...")
    last_improve = 0
    best_valid_bleu = 0
    print("epoch:", config.epochs)

    for e in range(config.epochs):
        start_time = time.time()

        train_loss = train(model, seq2seq_loss, evaluator_loss, train_loader, config.bos_token,
                           optimizer, e, cuda, config.max_iter_num)
        print('epoch:{},train_loss:{},time:{}sec'.format(e + 1, train_loss, round(time.time() - start_time, 2)))

        if (e + 1) % 5 == 0 or e >= 55 or e == 0:
            # validation
            valid_bleu, valid_rouge, valid_meteor, valid_prediction = \
                evaluate_model(model, test_loader, config.bos_token, config.comment_id2word, cuda, config.max_iter_num)

            print('epoch:{},valid_bleu:{},valid_rouge:{},valid_meteor:{},time:{}sec'.
                  format(e + 1, valid_bleu, valid_rouge, valid_meteor, round(time.time() - start_time, 2)))

            if valid_bleu > best_valid_bleu:
                best_valid_bleu = valid_bleu
                last_improve = e
                # save the best model parameters
                torch.save(model.state_dict(), f"./../saved_model/{config.dataset}/second_step_params.pkl")
                # output the prediction of comments for test set
                for ii, comment_pred in enumerate(valid_prediction.values()):
                    with open(f'./../results/{config.dataset}/second_step_result.{ii}', 'w') as w:
                        for comment_list in comment_pred:
                            comment = ' '.join(comment_list)
                            w.write(comment + '\n')

            if e - last_improve >= 20:
                print("Auto stopped and save model parameters")
                break

    print("best_valid_bleu:", best_valid_bleu)
