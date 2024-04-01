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


def train(model, seq2seq_loss_function, evaluator_loss_function, dataloader, bos_token, optimizer_list, epoch, cuda,
          max_iter_num):
    loss_list = [list() for _ in range(3)]
    model.train()

    seed_all(seed + epoch)
    for data in tqdm(dataloader):
        source_code, comment, draft, func_name, ast_node_embedding, adj_1, adj_2, adj_3, adj_4, adj_5, source_code_len, comment_len, draft_len, func_name_len, ast_node_embedding_len, adj_1_len, adj_2_len, adj_3_len, adj_4_len, adj_5_len = [
            d.cuda() for d in data[:20]] if cuda else data[:20]
        bos = torch.tensor([bos_token] * comment.size(0), device=comment.device).reshape(-1, 1)

        comment_input = torch.cat([bos, comment[:, :-1]], 1)
        comment_input_len = torch.add(comment_len, -1)

        # optimizer.zero_grad()
        source_code_enc, source_code_len = model.code_encoder(source_code, source_code_len)
        func_name_enc, func_name_len = model.func_name_encoder(func_name, func_name_len)
        ast_enc, ast_embedding = model.ast_encoder(ast_node_embedding, adj_1, adj_2, adj_3, adj_4, adj_5)

        """first_pass"""
        optimizer_list[0].zero_grad()
        draft_enc, draft_len = model.draft_encoder(draft, draft_len)
        comment_pred = model.first_decoder(comment_input, source_code_enc, draft_enc, func_name_enc,
                                           source_code_len, draft_len, func_name_len)
        comment_enc, comment_input_len = model.draft_encoder(comment, comment_input_len)
        anchor, positive, negative = model.evaluator(source_code_enc, source_code_len,
                                                     comment_enc, comment_input_len,
                                                     draft_enc, draft_len)
        loss1 = seq2seq_loss_function(comment_pred, comment, comment_len)
        loss2 = evaluator_loss_function(anchor, positive, negative) * 0.1
        one_loss = loss1 + loss2
        loss_list[0].append(one_loss.item())
        one_loss.backward()
        optimizer_list[0].step()
        draft_1 = torch.argmax(comment_pred.detach(), -1)
        draft_len_1 = comment_input_len

        """second_pass"""
        optimizer_list[1].zero_grad()
        draft_enc_1, draft_len_1 = model.draft_encoder(draft_1, draft_len_1)
        comment_pred = model.second_decoder(comment_input, source_code_enc.detach(), draft_enc_1, ast_enc,
                                            ast_embedding,
                                            source_code_len.detach(), draft_len_1)
        two_loss = seq2seq_loss_function(comment_pred, comment, comment_len)
        loss_list[1].append(two_loss.item())
        two_loss.backward()
        optimizer_list[1].step()
        draft_2 = torch.argmax(comment_pred.detach(), -1)
        draft_len_2 = comment_input_len

        """judge stage"""
        optimizer_list[2].zero_grad()
        draft_enc_2, draft_len_2 = model.draft_encoder(draft_2, draft_len_2)
        comment_pred = model.judge(source_code_enc.detach(), source_code_len.detach(), draft_enc_1.detach(), draft_len_1.detach(),
                                   draft_enc_2, draft_len_2, ast_enc.detach(), ast_node_embedding_len.detach())
        judge_loss = seq2seq_loss_function(comment_pred, comment, comment_len)
        loss_list[2].append(judge_loss.item())
        judge_loss.backward()
        optimizer_list[2].step()

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
    print("two_pass step one")
    # set config
    dataset_config_path = "./csn_python_config.json"
    with open(dataset_config_path, "r") as f:
        dataset_config = json.load(f)

    print("loading config...")
    config_file_path = './hyper_parameters.json'
    config = Config(config_file_path, dataset_config)
    print(dataset_config['name'])
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    cuda = torch.cuda.is_available() and config.cuda
    if cuda:
        print('Running on GPU')
        print(torch.cuda.get_device_name(torch.cuda.current_device()), os.environ["CUDA_VISIBLE_DEVICES"])
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
    # print("load the best model parameters!")
    # model.load_state_dict(torch.load(f"./../saved_model/{config.dataset}/first_step_params.pkl"))

    if cuda:
        model.cuda()

    # set loss function
    seq2seq_loss = MaskedSoftmaxCELoss()
    evaluator_loss = DAMSMLoss()

    optimizer0 = optim.Adam([{'params': [param for name, param in model.named_parameters()
                                         if 'second_decoder' not in name and 'ast_encoder' not in name and 'judge' not in name]}], lr=1e-4)
    optimizer1 = optim.Adam([{'params': [param for name, param in model.named_parameters()
                                         if 'second_decoder' in name or 'ast_encoder' in name]}], lr=1e-4)
    optimizer2 = optim.Adam([{'params': [param for name, param in model.named_parameters()
                                         if 'judge' in name]}], lr=1e-4)
    optimizer_list = [optimizer0, optimizer1, optimizer2]

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
    best_test_bleu = 0

    for e in range(config.epochs):
        start_time = time.time()

        train_loss = train(model, seq2seq_loss, evaluator_loss, train_loader, config.bos_token,
                           optimizer_list, e, cuda, config.max_iter_num)
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
                torch.save(model.state_dict(), f"./../saved_model/{config.dataset}/first_step_params.pkl")
                # output the prediction of comments for test set
                for ii, comment_pred in enumerate(valid_prediction.values()):
                    with open(f'./../results/{config.dataset}/first_step_result.{ii}', 'w') as w:
                        for comment_list in comment_pred:
                            comment = ' '.join(comment_list)
                            w.write(comment + '\n')

            if e - last_improve >= 20:
                print("No optimization for 20 epochs, auto-stopping and save model parameters")
                break

    print("finish!!!")
    print("best_valid_bleu:", best_valid_bleu)
    print("best_test_bleu:", best_test_bleu)
