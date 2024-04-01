# from model.attention import *
# from model.base_units import *
# from model.gcn import ASTEmbedding
import torch
from gcn import *
from base_units import *
from attention import *


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, head_num, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        self.feedForward = PositionWiseFFN(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x, valid_lens):
        y = self.self_attention(x, x, x, valid_lens)
        y = self.add_norm1(x, y)
        z = self.feedForward(y)
        z = self.add_norm2(y, z)
        return z


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, head_num, stack_num=6, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([EncoderBlock(d_model, d_ff, head_num, dropout) for _ in range(stack_num)])

    def forward(self, x, valid_lens):
        for layer in self.layers:
            x = layer(x, valid_lens)

        return x


class EncoderBlockWithRPR(nn.Module):
    def __init__(self, d_model, d_ff, head_num, clipping_distance, dropout=0.1):
        super(EncoderBlockWithRPR, self).__init__()
        self.self_attention = MultiHeadAttentionWithRPR(d_model, d_model, d_model, d_model, head_num,
                                                        clipping_distance, dropout)
        self.feedForward = PositionWiseFFN(d_model, d_ff, dropout)
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x, valid_lens):
        y = self.self_attention(x, x, x, valid_lens)
        y = self.add_norm1(x, y)
        z = self.feedForward(y)
        z = self.add_norm2(y, z)
        return z


class EncoderWithRPR(nn.Module):
    def __init__(self, d_model, d_ff, head_num, clipping_distance, stack_num=6, dropout=0.1):
        super(EncoderWithRPR, self).__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [EncoderBlockWithRPR(d_model, d_ff, head_num, clipping_distance, dropout) for _ in range(stack_num)])

    def forward(self, x, valid_lens):
        for layer in self.layers:
            x = layer(x, valid_lens)

        return x


class CodeEncoder(nn.Module):
    def __init__(self, code_embedding, d_model, d_ff, head_num, encoder_layer_num, clipping_distance, dropout=0.1):
        super(CodeEncoder, self).__init__()
        self.code_embedding = code_embedding
        self.code_encoder = EncoderWithRPR(d_model, d_ff, head_num, clipping_distance, encoder_layer_num, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source_code, source_code_len):
        source_code_embed = self.dropout(self.code_embedding(source_code))
        source_code_enc = self.code_encoder(source_code_embed, source_code_len)
        return source_code_enc, source_code_len


class CommentEncoder(nn.Module):
    def __init__(self, comment_embedding, pos_encoding, d_model, d_ff, head_num, encoder_layer_num, dropout=0.1):
        super(CommentEncoder, self).__init__()
        self.comment_embedding = comment_embedding
        self.pos_encoding = pos_encoding
        self.template_encoder = Encoder(d_model, d_ff, head_num, encoder_layer_num, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, comment, comment_len):
        b_, seq_comment_num = comment.size()
        comment_pos = torch.arange(seq_comment_num, device=comment.device).repeat(b_, 1)
        comment_embed = self.pos_encoding(self.comment_embedding(comment), comment_pos)
        comment_enc = self.template_encoder(comment_embed, comment_len)

        return comment_enc, comment_len


class FuncNameEncoder(nn.Module):
    def __init__(self, comment_embedding, d_model, d_ff, head_num, encoder_layer_num, dropout=0.1):
        super(FuncNameEncoder, self).__init__()
        self.comment_embedding = comment_embedding
        self.keywords_encoder = Encoder(d_model, d_ff, head_num, encoder_layer_num, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, keywords, keywords_len):
        keywords_embed = self.dropout(self.comment_embedding(keywords))
        keywords_enc = self.keywords_encoder(keywords_embed, keywords_len)

        return keywords_enc, keywords_len


class AstEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, d_model, d_ff, batch_size, head_num, encoder_layer_num,
                 dropout=0.1):
        super(AstEncoder, self).__init__()
        self.ast_embedding = ASTEmbedding(input_dim, hidden_dim, output_dim, d_model, batch_size, dropout)
        self.ast_encoder = Encoder(d_model, d_ff, head_num, encoder_layer_num, dropout)

    def forward(self, x, a, a2, a3, a4, a5, ast_len=None):
        ast_embed = self.ast_embedding(x, a, a2, a3, a4, a5)
        ast_enc = self.ast_encoder(ast_embed, ast_len)
        return ast_enc, ast_len


class AstEncoder2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, d_model, d_ff, batch_size, head_num, encoder_layer_num,
                 dropout=0.1):
        super(AstEncoder2, self).__init__()
        self.ast_embedding = ASTEmbedding(input_dim, hidden_dim, output_dim, d_model, batch_size, dropout)
        self.ast_encoder = Encoder(d_model, d_ff, head_num, encoder_layer_num, dropout)

    def forward(self, x, a, a2, a3, a4, a5, ast_len=None):
        ast_embed = self.ast_embedding(x, a, a2, a3, a4, a5)
        ast_enc = self.ast_encoder(ast_embed, ast_len)
        return ast_enc, ast_embed


if __name__ == "__main__":
    pass
