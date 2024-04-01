from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class Evaluator(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(Evaluator, self).__init__()
        self.linear_proj1 = nn.Linear(d_model, d_ff)
        self.linear_proj2 = nn.Linear(d_ff, d_model)
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source_code_enc, source_code_len, comment_enc, comment_len, template_enc, template_len,
                best_result=None, cur_result=None):
        b_ = source_code_enc.size(0)

        source_code_vec = torch.cumsum(source_code_enc, dim=1)[torch.arange(b_), source_code_len - 1]
        source_code_vec = torch.div(source_code_vec.T, source_code_len).T
        source_code_vec = self.linear_proj2(self.dropout(F.relu(self.linear_proj1(source_code_vec))))

        comment_vec = torch.cumsum(comment_enc, dim=1)[torch.arange(b_), comment_len - 1]
        comment_vec = torch.div(comment_vec.T, comment_len).T
        comment_vec = self.linear_proj2(self.dropout(F.relu(self.linear_proj1(comment_vec))))

        template_vec = torch.cumsum(template_enc, dim=1)[torch.arange(b_), template_len - 1]
        template_vec = torch.div(template_vec.T, template_len).T
        template_vec = self.linear_proj2(self.dropout(F.relu(self.linear_proj1(template_vec))))

        if self.training:
            return source_code_vec, comment_vec, template_vec
        else:
            best_enc = [x for x in template_enc]
            best_len = template_len

            assert best_result is not None
            assert cur_result is not None
            pos_sim = self.cos_sim(source_code_vec, comment_vec)
            neg_sim = self.cos_sim(source_code_vec, template_vec)
            better_index = (pos_sim > neg_sim).nonzero(as_tuple=True)[0]
            for ix in better_index:
                best_result[ix] = cur_result[ix]
                best_enc[ix] = comment_enc[ix]
                best_len[ix] = comment_len[ix]

            return best_result, pad_sequence(best_enc, batch_first=True), best_len
