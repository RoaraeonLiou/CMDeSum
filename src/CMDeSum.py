from model.encoder import *
from model.positional_encoding import *
from torch.nn.utils.rnn import pad_sequence
from model.evaluator import Evaluator
from model.attention import *
from model.base_units import *
from model.beam_search import *
from queue import PriorityQueue


class CDecoderBlock(nn.Module):
    def __init__(self, i, d_model, d_ff, head_num, dropout=0.1):
        super(CDecoderBlock, self).__init__()
        self.i = i
        self.masked_self_attention = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.add_norm1 = AddNorm(d_model)
        self.cross_attention_code = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.cross_attention_keywords = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.gate = nn.Linear(d_model + d_model, 1)
        self.add_norm2 = AddNorm(d_model)
        self.cross_attention_template = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.add_norm3 = AddNorm(d_model)
        self.feedForward = PositionWiseFFN(d_model, d_ff, dropout)
        self.add_norm4 = AddNorm(d_model)

    def forward(self, x, state):
        code_enc, code_len = state[0], state[1]
        template_enc, template_len = state[2], state[3]
        keywords_enc, keywords_len = state[4], state[5]
        if state[6][self.i] is None:
            key_values = x
        else:
            key_values = torch.cat((state[6][self.i], x), axis=1)
        state[6][self.i] = key_values

        if self.training:
            batch_size, num_steps, _ = x.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=x.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        x_self_att = self.masked_self_attention(x, key_values, key_values, dec_valid_lens)
        y = self.add_norm1(x, x_self_att)
        y_cross_code = self.cross_attention_code(y, code_enc, code_enc, code_len)
        y_cross_func_name = self.cross_attention_keywords(y, keywords_enc, keywords_enc, keywords_len)

        gate_weight = torch.sigmoid(self.gate(torch.cat([y_cross_code, y_cross_func_name], dim=-1)))
        y_fusion = gate_weight * y_cross_code + (1. - gate_weight) * y_cross_func_name

        z = self.add_norm2(y, y_fusion)

        z_cross_draft = self.cross_attention_template(z, template_enc, template_enc, template_len)

        z2 = self.add_norm3(z, z_cross_draft)
        z_end = self.add_norm4(z2, self.feedForward(z2))
        return z_end, state


class CDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, head_num, N=6, dropout=0.1):
        super(CDecoder, self).__init__()
        self.num_layers = N
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [CDecoderBlock(i, d_model, d_ff, head_num, dropout) for i in range(self.num_layers)])
        self.dense = nn.Linear(d_model, vocab_size)

    def init_state(self, code_enc, code_len, template_enc, template_len,
                   keywords_enc, keywords_len):
        return [code_enc, code_len, template_enc, template_len,
                keywords_enc, keywords_len, [None] * self.num_layers]

    def forward(self, x, state):
        for layer in self.layers:
            x, state = layer(x, state)

        return self.dense(x), state


class CodeGuidedDecoder(nn.Module):
    def __init__(self, comment_embedding, pos_encoding, d_model, d_ff, head_num, decoder_layer_num, comment_vocab_size,
                 bos_token, eos_token, max_comment_len, dropout=0.1, beam_width=4):
        super(CodeGuidedDecoder, self).__init__()
        self.comment_embedding = comment_embedding
        self.pos_encoding = pos_encoding
        self.comment_decoder = CDecoder(comment_vocab_size, d_model, d_ff, head_num, decoder_layer_num,
                                        dropout)

        self.max_comment_len = max_comment_len
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.d_model = d_model
        self.beam_width = beam_width
        self.num_layers = decoder_layer_num

    def forward(self, comment, source_code_enc, template_enc, keywords_enc,
                source_code_len, template_len, keywords_len):
        b_, seq_comment_num = comment.size()
        dec_state = self.comment_decoder.init_state(source_code_enc, source_code_len,
                                                    template_enc, template_len,
                                                    keywords_enc, keywords_len)

        if self.training:
            comment_pos = torch.arange(seq_comment_num, device=comment.device).repeat(b_, 1)
            comment_embed = self.pos_encoding(self.comment_embedding(comment), comment_pos)
            comment_pred = self.comment_decoder(comment_embed, dec_state)[0]
            return comment_pred
        else:
            if self.beam_width:
                return self.beam_search(b_, comment, dec_state, self.beam_width)
            else:
                return self.greed_search(b_, comment, dec_state)

    def greed_search(self, batch_size, comment, dec_state):
        comment_pred = [[self.bos_token] for _ in range(batch_size)]
        for pos_idx in range(self.max_comment_len):
            comment_pos = torch.arange(pos_idx, pos_idx + 1, device=comment.device).repeat(batch_size, 1)
            comment_embed = self.pos_encoding(self.comment_embedding(comment), comment_pos)
            comment, dec_state = self.comment_decoder(comment_embed, dec_state)
            comment = torch.argmax(comment, -1).detach()
            for i in range(batch_size):
                if comment_pred[i][-1] != self.eos_token:
                    comment_pred[i].append(int(comment[i]))

        comment_pred = [x[1:-1] if x[-1] == self.eos_token and len(x) > 2 else x[1:]
                        for x in comment_pred]
        return comment_pred

    def beam_search(self, batch_size, comment, dec_state, beam_width):
        node_list = []
        batchNode_dict = {i: None for i in range(beam_width)}
        for batch_idx in range(batch_size):
            node_comment = comment[batch_idx].unsqueeze(0)
            node_dec_state = [dec_state[0][batch_idx].unsqueeze(0), dec_state[1][batch_idx].unsqueeze(0),
                              dec_state[2][batch_idx].unsqueeze(0), dec_state[3][batch_idx].unsqueeze(0),
                              dec_state[4][batch_idx].unsqueeze(0), dec_state[5][batch_idx].unsqueeze(0),
                              [None] * self.num_layers]
            node_list.append(BeamSearchNode(node_dec_state, None, node_comment, 0, 0))
        batchNode_dict[0] = BatchNodeWithKeywords(node_list)

        pos_idx = 0
        while pos_idx < self.max_comment_len:
            beamNode_dict = {i: PriorityQueue() for i in range(batch_size)}
            count = 0
            for idx in range(beam_width):
                if batchNode_dict[idx] is None:
                    continue

                batchNode = batchNode_dict[idx]
                comment = batchNode.get_comment()
                dec_state = batchNode.get_dec_state()

                pos = torch.arange(pos_idx, pos_idx + 1, device=comment.device).repeat(batch_size, 1)
                comment = self.pos_encoding(self.comment_embedding(comment), pos)
                tensor, dec_state = self.comment_decoder(comment, dec_state)
                tensor = F.log_softmax(tensor.squeeze(1), -1).detach()
                log_prob, comment_candidates = torch.topk(tensor, beam_width, dim=-1)

                for batch_idx in range(batch_size):
                    pre_node = batchNode.list_node[batch_idx]
                    node_dec_state = [dec_state[0][batch_idx].unsqueeze(0), dec_state[1][batch_idx].unsqueeze(0),
                                      dec_state[2][batch_idx].unsqueeze(0), dec_state[3][batch_idx].unsqueeze(0),
                                      dec_state[4][batch_idx].unsqueeze(0), dec_state[5][batch_idx].unsqueeze(0),
                                      [l[batch_idx].unsqueeze(0) for l in dec_state[6]]]
                    if pre_node.history_word[-1] == self.eos_token:
                        new_node = BeamSearchNode(node_dec_state, pre_node.prevNode, pre_node.commentID,
                                                  pre_node.logp, pre_node.leng)
                        assert new_node.score == pre_node.score
                        assert new_node.history_word == pre_node.history_word
                        beamNode_dict[batch_idx].put((-new_node.score, count, new_node))
                        count += 1
                        continue

                    for beam_idx in range(beam_width):
                        node_comment = comment_candidates[batch_idx][beam_idx].view(1, -1)
                        node_log_prob = float(log_prob[batch_idx][beam_idx])
                        new_node = BeamSearchNode(node_dec_state, pre_node, node_comment, pre_node.logp + node_log_prob,
                                                  pre_node.leng + 1)
                        beamNode_dict[batch_idx].put((-new_node.score, count, new_node))
                        count += 1

            for beam_idx in range(beam_width):
                node_list = [beamNode_dict[batch_idx].get()[-1] for batch_idx in range(batch_size)]
                batchNode_dict[beam_idx] = BatchNodeWithKeywords(node_list)

            pos_idx += 1
        best_node = batchNode_dict[0]
        comment_pred = []
        for batch_idx in range(batch_size):
            history_word = best_node.list_node[batch_idx].history_word
            if history_word[-1] == self.eos_token and len(history_word) > 2:
                comment_pred.append(history_word[1:-1])
            else:
                comment_pred.append(history_word[1:])

        return comment_pred


class ADecoderBlock(nn.Module):
    def __init__(self, i, d_model, d_ff, head_num, dropout=0.1):
        super(ADecoderBlock, self).__init__()
        self.i = i

        self.masked_self_attention = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.add_norm1 = AddNorm(d_model)
        self.cross_attention_keyword = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.add_norm2 = AddNorm(d_model)
        self.cross_attention_multi = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.add_norm3 = AddNorm(d_model)

        self.cross_attention_template = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.add_norm4 = AddNorm(d_model)
        self.feedForward = PositionWiseFFN(d_model, d_ff, dropout)
        self.add_norm5 = AddNorm(d_model)

    def forward(self, x, state):
        co_code_enc, co_code_len = state[0], state[1]
        template_enc, template_len = state[2], state[3]
        co_ast_enc, co_ast_len = state[4], state[5]
        if state[6][self.i] is None:
            key_values = x
        else:
            key_values = torch.cat((state[6][self.i], x), axis=1)
        state[6][self.i] = key_values

        if self.training:
            batch_size, num_steps, _ = x.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=x.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        x_self_att = self.masked_self_attention(x, key_values, key_values, dec_valid_lens)
        y = self.add_norm1(x, x_self_att)

        y_cross_code = self.cross_attention_keyword(y, co_code_enc, co_code_enc, co_code_len)
        y1 = self.add_norm2(y, y_cross_code)
        y_cross_ast = self.cross_attention_multi(y1, co_ast_enc, co_ast_enc, None)
        y2 = self.add_norm3(y1, y_cross_ast)
        y_cross_draft = self.cross_attention_template(y2, template_enc, template_enc, template_len)
        z = self.add_norm4(y2, y_cross_draft)

        z_end = self.add_norm5(z, self.feedForward(z))
        return z_end, state


class ADecoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, head_num, N=6, dropout=0.1):
        super(ADecoder, self).__init__()
        self.num_layers = N
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [ADecoderBlock(i, d_model, d_ff, head_num, dropout) for i in range(self.num_layers)])
        self.dense = nn.Linear(d_model, vocab_size)

    def init_state(self, co_code_enc, keywords_len, template_enc, source_code_len,
                   co_ast_enc, co_ast_len):
        return [co_code_enc, keywords_len, template_enc, source_code_len,
                co_ast_enc, co_ast_len, [None] * self.num_layers]

    def forward(self, x, state):
        for layer in self.layers:
            x, state = layer(x, state)

        return self.dense(x), state


class CoAttention(nn.Module):
    def __init__(self, d_model):
        super(CoAttention, self).__init__()

        self.linear_vision = nn.Linear(d_model, d_model)
        self.linear_language = nn.Linear(d_model, d_model)
        self.linear_combined = nn.Linear(2 * d_model, d_model)

    def forward(self, vision_features, language_features):
        vision_att = torch.matmul(self.linear_vision(vision_features), language_features.transpose(-2, -1))
        vision_weights = nn.functional.softmax(vision_att, dim=-1)

        language_att = torch.matmul(self.linear_language(language_features), vision_features.transpose(-2, -1))
        language_weights = nn.functional.softmax(language_att, dim=-1)

        weighted_vision = torch.matmul(language_weights, vision_features)

        weighted_language = torch.matmul(vision_weights, language_features)

        coattended_vision = self.linear_combined(torch.cat((vision_features, weighted_language), dim=-1))
        coattended_language = self.linear_combined(torch.cat((language_features, weighted_vision), dim=-1))

        return coattended_vision, coattended_language


class Multi_model(nn.Module):
    def __init__(self, d_model, head_num, dropout):
        super(Multi_model, self).__init__()
        self.coatten = CoAttention(d_model)
        self.cross_attention = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.add_norm = AddNorm(d_model)

    def forward(self, gcn_embed, src_embed, AST_embed):
        coatten_ast, coatten_src = self.coatten(gcn_embed, src_embed)
        ast_x = self.cross_attention(coatten_ast, AST_embed, AST_embed, None)
        ast_y = self.add_norm(ast_x, coatten_ast)
        return ast_y, coatten_src


class ASTGuidedDecoder(nn.Module):
    def __init__(self, comment_embedding, pos_encoding, d_model, d_ff, head_num, decoder_layer_num, comment_vocab_size,
                 bos_token, eos_token, max_comment_len, dropout=0.1, beam_width=4):
        super(ASTGuidedDecoder, self).__init__()
        self.comment_embedding = comment_embedding
        self.pos_encoding = pos_encoding
        self.multi = Multi_model(d_model, head_num, dropout)
        self.comment_decoder = ADecoder(comment_vocab_size, d_model, d_ff, head_num, decoder_layer_num,
                                        dropout)

        self.max_comment_len = max_comment_len
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.d_model = d_model
        self.beam_width = beam_width
        self.num_layers = decoder_layer_num

    def forward(self, comment, source_code_enc, template_enc, ast_enc, ast_embedding,
                source_code_len, template_len):
        co_ast_enc, co_code_enc = self.multi(ast_enc, source_code_enc, ast_embedding)
        b_, seq_comment_num = comment.size()
        dec_state = self.comment_decoder.init_state(co_code_enc, source_code_len,
                                                    template_enc, template_len,
                                                    co_ast_enc, None)

        if self.training:
            comment_pos = torch.arange(seq_comment_num, device=comment.device).repeat(b_, 1)
            comment_embed = self.pos_encoding(self.comment_embedding(comment), comment_pos)
            comment_pred = self.comment_decoder(comment_embed, dec_state)[0]
            return comment_pred
        else:
            if self.beam_width:
                return self.beam_search(b_, comment, dec_state, self.beam_width)
            else:
                return self.greed_search(b_, comment, dec_state)

    def greed_search(self, batch_size, comment, dec_state):
        comment_pred = [[self.bos_token] for _ in range(batch_size)]
        for pos_idx in range(self.max_comment_len):
            comment_pos = torch.arange(pos_idx, pos_idx + 1, device=comment.device).repeat(batch_size, 1)
            comment_embed = self.pos_encoding(self.comment_embedding(comment), comment_pos)
            comment, dec_state = self.comment_decoder(comment_embed, dec_state)
            comment = torch.argmax(comment, -1).detach()
            for i in range(batch_size):
                if comment_pred[i][-1] != self.eos_token:
                    comment_pred[i].append(int(comment[i]))

        comment_pred = [x[1:-1] if x[-1] == self.eos_token and len(x) > 2 else x[1:]
                        for x in comment_pred]
        return comment_pred

    def beam_search(self, batch_size, comment, dec_state, beam_width):
        node_list = []
        batchNode_dict = {i: None for i in range(beam_width)}
        for batch_idx in range(batch_size):
            node_comment = comment[batch_idx].unsqueeze(0)
            node_dec_state = [dec_state[0][batch_idx].unsqueeze(0), dec_state[1][batch_idx].unsqueeze(0),
                              dec_state[2][batch_idx].unsqueeze(0), dec_state[3][batch_idx].unsqueeze(0),
                              dec_state[4][batch_idx].unsqueeze(0), None,
                              [None] * self.num_layers]
            node_list.append(BeamSearchNode(node_dec_state, None, node_comment, 0, 0))
        batchNode_dict[0] = BatchNodeWithKeywords(node_list)

        pos_idx = 0
        while pos_idx < self.max_comment_len:
            beamNode_dict = {i: PriorityQueue() for i in range(batch_size)}
            count = 0
            for idx in range(beam_width):
                if batchNode_dict[idx] is None:
                    continue

                batchNode = batchNode_dict[idx]
                comment = batchNode.get_comment()
                dec_state = batchNode.get_dec_state()
                pos = torch.arange(pos_idx, pos_idx + 1, device=comment.device).repeat(batch_size, 1)
                comment = self.pos_encoding(self.comment_embedding(comment), pos)
                tensor, dec_state = self.comment_decoder(comment, dec_state)
                tensor = F.log_softmax(tensor.squeeze(1), -1).detach()
                log_prob, comment_candidates = torch.topk(tensor, beam_width, dim=-1)

                for batch_idx in range(batch_size):
                    pre_node = batchNode.list_node[batch_idx]
                    node_dec_state = [dec_state[0][batch_idx].unsqueeze(0), dec_state[1][batch_idx].unsqueeze(0),
                                      dec_state[2][batch_idx].unsqueeze(0), dec_state[3][batch_idx].unsqueeze(0),
                                      dec_state[4][batch_idx].unsqueeze(0), None,
                                      [l[batch_idx].unsqueeze(0) for l in dec_state[6]]]
                    if pre_node.history_word[-1] == self.eos_token:
                        new_node = BeamSearchNode(node_dec_state, pre_node.prevNode, pre_node.commentID,
                                                  pre_node.logp, pre_node.leng)
                        assert new_node.score == pre_node.score
                        assert new_node.history_word == pre_node.history_word
                        beamNode_dict[batch_idx].put((-new_node.score, count, new_node))
                        count += 1
                        continue

                    for beam_idx in range(beam_width):
                        node_comment = comment_candidates[batch_idx][beam_idx].view(1, -1)
                        node_log_prob = float(log_prob[batch_idx][beam_idx])
                        new_node = BeamSearchNode(node_dec_state, pre_node, node_comment, pre_node.logp + node_log_prob,
                                                  pre_node.leng + 1)
                        beamNode_dict[batch_idx].put((-new_node.score, count, new_node))
                        count += 1

            for beam_idx in range(beam_width):
                node_list = [beamNode_dict[batch_idx].get()[-1] for batch_idx in range(batch_size)]
                batchNode_dict[beam_idx] = BatchNodeWithKeywords(node_list)

            pos_idx += 1
        best_node = batchNode_dict[0]
        comment_pred = []
        for batch_idx in range(batch_size):
            history_word = best_node.list_node[batch_idx].history_word
            if history_word[-1] == self.eos_token and len(history_word) > 2:
                comment_pred.append(history_word[1:-1])
            else:
                comment_pred.append(history_word[1:])

        return comment_pred


class Judge(nn.Module):
    def __init__(self, d_model, d_ff, vocab_size, dropout):
        super(Judge, self).__init__()
        self.d_model = d_model
        self.linear_proj1 = nn.Linear(d_model, d_ff)
        self.linear_proj2 = nn.Linear(d_ff, d_model)
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.dense = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source_code_enc, source_code_len, draft_0_enc, draft_0_len, draft_1_enc, draft_1_len, ast_enc,
                ast_len):
        b_ = source_code_enc.size(0)
        source_code_vec = torch.cumsum(source_code_enc, dim=1)[torch.arange(b_), source_code_len - 1]
        source_code_vec = torch.div(source_code_vec.T, source_code_len).T
        source_code_vec = self.linear_proj2(self.dropout(F.relu(self.linear_proj1(source_code_vec))))

        comment_1_vec = torch.cumsum(draft_0_enc, dim=1)[torch.arange(b_), draft_0_len - 1]
        comment_1_vec = torch.div(comment_1_vec.T, draft_0_len).T
        comment_1_vec = self.linear_proj2(self.dropout(F.relu(self.linear_proj1(comment_1_vec))))

        comment_2_vec = torch.cumsum(draft_1_enc, dim=1)[torch.arange(b_), draft_1_len - 1]
        comment_2_vec = torch.div(comment_2_vec.T, draft_1_len).T
        comment_2_vec = self.linear_proj2(self.dropout(F.relu(self.linear_proj1(comment_2_vec))))

        ast_vec = torch.cumsum(ast_enc, dim=1)[torch.arange(b_), ast_len - 1]
        ast_vec = torch.div(ast_vec.T, ast_len).T
        ast_vec = self.linear_proj2(self.dropout(F.relu(self.linear_proj1(ast_vec))))

        comment_1_code_sim = self.cos_sim(comment_1_vec, source_code_vec)
        comment_1_ast_sim = self.cos_sim(comment_1_vec, ast_vec)
        comment_2_code_sim = self.cos_sim(comment_2_vec, source_code_vec)
        comment_2_ast_sim = self.cos_sim(comment_2_vec, ast_vec)
        comment_1_weight = (comment_1_code_sim + comment_1_ast_sim) / 2
        comment_2_weight = (comment_2_code_sim + comment_2_ast_sim) / 2
        comment_1_weight = comment_1_weight.view(b_, 1, 1).expand_as(draft_0_enc)
        comment_2_weight = comment_2_weight.view(b_, 1, 1).expand_as(draft_1_enc)
        a = draft_0_enc * comment_1_weight
        b = draft_1_enc * comment_2_weight
        if self.training:
            comment_enc = (a + b) / 2
            final_comment = self.dense(comment_enc)
            return final_comment
        else:
            diff = a.size(1) - b.size(1)
            if diff < 0:
                padding = torch.zeros((b_, abs(diff), self.d_model), device=a.device)
                a = torch.cat((a, padding), dim=1)
            else:
                padding = torch.zeros((b_, diff, self.d_model), device=b.device)
                b = torch.cat((b, padding), dim=1)
            comment_enc = (a + b) / 2
            final_comment = self.dense(comment_enc)
            return torch.argmax(final_comment.detach(), -1).tolist()


class CMDeSum(nn.Module):
    def __init__(self, batch_size, d_model, d_ff, head_num, encoder_layer_num, decoder_layer_num, code_vocab_size,
                 comment_vocab_size, bos_token, eos_token, max_comment_len, clipping_distance, max_iter_num,
                 dropout=0.1, beam_width=4, input_dim=768, hidden_num=768, output_dim=768):
        super(CMDeSum, self).__init__()
        self.model_name = "CMDeSum"
        self.code_embedding = nn.Embedding(code_vocab_size, d_model)
        self.comment_embedding = nn.Embedding(comment_vocab_size, d_model)

        self.pos_encoding = LearnablePositionalEncoding(d_model, dropout, max_comment_len + 2)

        self.code_encoder = CodeEncoder(self.code_embedding, d_model, d_ff, head_num,
                                        encoder_layer_num, clipping_distance, dropout)
        self.func_name_encoder = FuncNameEncoder(self.comment_embedding, d_model, d_ff, head_num,
                                                 encoder_layer_num, dropout)
        self.draft_encoder = CommentEncoder(self.comment_embedding, self.pos_encoding, d_model, d_ff, head_num,
                                            encoder_layer_num, dropout)
        self.ast_encoder = AstEncoder2(input_dim, hidden_num, output_dim, d_model, d_ff, batch_size, head_num,
                                       encoder_layer_num, dropout)

        self.first_decoder = CodeGuidedDecoder(self.comment_embedding, self.pos_encoding, d_model,
                                               d_ff, head_num,
                                               decoder_layer_num, comment_vocab_size, bos_token,
                                               eos_token, max_comment_len,
                                               dropout, beam_width)
        self.second_decoder = ASTGuidedDecoder(self.comment_embedding, self.pos_encoding, d_model,
                                               d_ff, head_num,
                                               decoder_layer_num, comment_vocab_size, bos_token,
                                               eos_token, max_comment_len,
                                               dropout, beam_width)

        self.evaluator = Evaluator(d_model, d_ff, dropout)

        self.judge = Judge(d_model, d_ff, comment_vocab_size, dropout)

    def forward(self, source_code, comment, draft, keywords,
                source_code_len, comment_len, draft_len, keywords_len, ast_node, adj_1, adj_2, adj_3, adj_4, adj_5,
                ast_node_len):
        code_enc, code_len = self.code_encoder(source_code, source_code_len)
        func_name_enc, func_name_len = self.func_name_encoder(keywords, keywords_len)
        draft_enc, draft_len = self.draft_encoder(draft, draft_len)
        ast_enc, ast_embedding = self.ast_encoder(ast_node, adj_1, adj_2, adj_3, adj_4, adj_5)

        if self.training:
            comment_enc, comment_len = self.draft_encoder(comment[:, 1:], comment_len)
            anchor, positive, negative = self.evaluator(code_enc, code_len, comment_enc, comment_len, draft_enc,
                                                        draft_len)

            memory = []
            comment_pred_1 = self.first_decoder(comment, code_enc, draft_enc, func_name_enc,
                                                code_len, draft_len, func_name_len)
            memory.append(comment_pred_1)
            draft_1 = torch.argmax(comment_pred_1.detach(), -1)
            draft_len_1 = comment_len
            draft_1_enc, draft_len_1 = self.draft_encoder(draft_1, draft_len_1)
            comment_pred_2 = self.second_decoder(comment, code_enc, draft_1_enc, ast_enc, ast_embedding,
                                                 code_len, draft_len_1)
            memory.append(comment_pred_2)
            draft_2 = torch.argmax(comment_pred_2.detach(), -1)
            draft_len_2 = comment_len
            draft_2_enc, draft_len_2 = self.draft_encoder(draft_2, draft_len_2)
            comment_pred_3 = self.judge(code_enc, code_len, draft_1_enc, draft_len_1, draft_2_enc, draft_len_2, ast_enc,
                                        ast_node_len)
            memory.append(comment_pred_3)
            return memory, anchor, positive, negative


        else:
            memory = []
            best_result = [x.tolist()[:leng] for x, leng in zip(draft, draft_len)]
            best_enc = draft_enc
            best_len = draft_len

            comment_pred_1 = self.first_decoder(comment, code_enc, draft_enc, func_name_enc,
                                                code_len, draft_len, func_name_len)
            memory.append(comment_pred_1)
            draft_1 = pad_sequence([torch.tensor(x, device=comment.device) for x in comment_pred_1],
                                   batch_first=True)
            draft_len_1 = torch.tensor([len(x) for x in comment_pred_1], device=comment.device)
            draft_enc_1, draft_len_1 = self.draft_encoder(draft_1, draft_len_1)
            best_result, best_enc, best_len = self.evaluator(code_enc, source_code_len, draft_enc_1, draft_len_1,
                                                             best_enc, best_len, best_result, comment_pred_1)

            comment_pred_2 = self.second_decoder(comment, code_enc, draft_enc_1, ast_enc, ast_embedding,
                                                 code_len, draft_len_1)
            memory.append(comment_pred_2)
            draft_2 = pad_sequence([torch.tensor(x, device=comment.device) for x in comment_pred_2],
                                   batch_first=True)
            draft_len_2 = torch.tensor([len(x) for x in comment_pred_2], device=comment.device)
            draft_enc_2, draft_len_2 = self.draft_encoder(draft_2, draft_len_2)

            comment_pred_3 = self.judge(code_enc, code_len, draft_enc_1, draft_len_1, draft_enc_2, draft_len_2, ast_enc,
                                        ast_node_len)
            memory.append(comment_pred_3)
            draft_3 = pad_sequence([torch.tensor(x, device=comment.device) for x in comment_pred_3], batch_first=True)
            draft_len_3 = torch.tensor([len(x) for x in comment_pred_3], device=comment.device)
            draft_enc_3, draft_len_3 = self.draft_encoder(draft_3, draft_len_3)
            best_result, best_enc, best_len = self.evaluator(code_enc, source_code_len, draft_enc_3, draft_len_3,
                                                             best_enc, best_len, best_result, comment_pred_2)
            memory.append(best_result)
            assert len(memory) == 4
            return memory


if __name__ == "__main__":
    c = CMDeSum(batch_size=32, d_model=512, d_ff=2048, head_num=8, encoder_layer_num=4, decoder_layer_num=6,
                code_vocab_size=60000, comment_vocab_size=35877, bos_token=0, eos_token=1, max_comment_len=50,
                clipping_distance=16, max_iter_num=3)
    total_num = sum(p.numel() for p in c.parameters())
    trainable_num = sum(p.numel() for p in c.parameters() if p.requires_grad)
    cnt = 0
