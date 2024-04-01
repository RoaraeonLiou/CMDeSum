from model.attention import *
from model.base_units import *
import torch
from model.gcn import ASTEmbedding


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, head_num, dropout=0.1):
        """
        编码器基本块
        :param d_model: 模型维度
        :param d_ff: 前馈神经网络维度
        :param head_num: 多头注意力头数
        :param dropout: Dropout率
        """
        super(EncoderBlock, self).__init__()
        """
        构建多头注意力层，此处qkv维度与模型维度相同
        """
        self.self_attention = MultiHeadAttention(d_model, d_model, d_model, d_model, head_num, dropout)
        """
        构建前馈网络层, 从模型维度映射到前馈网络维度
        """
        self.feedForward = PositionWiseFFN(d_model, d_ff, dropout)
        """
        残差链接和层归一化
        """
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x, valid_lens):
        """

        :param x: (batch size, num of qkv, num of hidden)
        :param valid_lens: (batch size, ) or (batch size, num of qkv)
        :return:
        """
        # y (batch size, num of qkv, num of hidden)
        y = self.self_attention(x, x, x, valid_lens)
        y = self.add_norm1(x, y)
        # z (batch size, num of qkv, num of hidden)
        z = self.feedForward(y)
        z = self.add_norm2(y, z)
        # z (batch size, num of qkv, num of hidden)
        return z


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, head_num, stack_num=6, dropout=0.1):
        """

        :param d_model: 模型的维度，也是Q、K、V的维度
        :param d_ff: 前馈网络隐藏层的特征维度
        :param head_num: 多头注意力中的头数
        :param stack_num: 编码器中基本快的数量
        :param dropout: Dropout率
        """
        super(Encoder, self).__init__()
        self.d_model = d_model
        # self.embedding = nn.Embedding(vocab_size, d_model)
        # self.pos_encoding = PositionalEncoding(d_model, dropout)
        """
        创建含有N个Encoder的ModuleList
        """
        self.layers = nn.ModuleList([EncoderBlock(d_model, d_ff, head_num, dropout) for _ in range(stack_num)])

    def forward(self, x, valid_lens):
        """

        :param x: (batch size, num of qkv, num of hidden)
        :param valid_lens: (batch size, ) or (batch size, num of qkv)
        :return: Tensor (batch size, num of qkv, num of hidden)
        """
        # encoder编码时,输入x是CNN输出的embedding
        # x = self.pos_encoding(self.embedding(x) * math.sqrt(self.d_model))
        """
        依次使用六个编码器基本块对x进行编码
        """
        for layer in self.layers:
            # x (batch size, num of qkv, num of hidden)
            x = layer(x, valid_lens)

        return x


class EncoderBlockWithRPR(nn.Module):
    def __init__(self, d_model, d_ff, head_num, clipping_distance, dropout=0.1):
        """
        带有相对位置编码的编码器基本块
        :param d_model: 模型的维度，也是Q、K、V的维度
        :param d_ff: 前馈网络隐层的维度
        :param head_num: 多头注意力头数
        :param clipping_distance: 相对位置编码的剪切距离
        :param dropout: Dropout率
        """
        super(EncoderBlockWithRPR, self).__init__()
        # 带有相对位置编码的多头自注意力层
        self.self_attention = MultiHeadAttentionWithRPR(d_model, d_model, d_model, d_model, head_num,
                                                        clipping_distance, dropout)
        # 前馈网络层
        self.feedForward = PositionWiseFFN(d_model, d_ff, dropout)
        # 第一个残差链接和层归一化
        self.add_norm1 = AddNorm(d_model)
        # 第二个残差链接和层归一化
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x, valid_lens):
        """

        :param x: (batch size, num of qkv, num of hidden)
        :param valid_lens: (batch size, ) or (batch size, num of qkv)
        :return: Tensor (batch size, num of qkv, num of hidden)
        """
        # y (batch size, num of qkv, num of hidden)
        y = self.self_attention(x, x, x, valid_lens)
        y = self.add_norm1(x, y)
        # z (batch size, num of qkv, num of hidden)
        z = self.feedForward(y)
        z = self.add_norm2(y, z)
        # z (batch size, num of qkv, num of hidden)
        return z


class EncoderWithRPR(nn.Module):
    def __init__(self, d_model, d_ff, head_num, clipping_distance, stack_num=6, dropout=0.1):
        """
        带有相对位置编码的编码器
        :param d_model: 模型的维度，也是Q、K、V的维度
        :param d_ff: 前馈网络隐藏层的特征维度
        :param head_num: 多头注意力中的头数
        :param clipping_distance: 相对位置编码的剪切距离
        :param stack_num: 编码器中基本快的数量
        :param dropout:
        """
        super(EncoderWithRPR, self).__init__()
        self.d_model = d_model
        # self.embedding = nn.Embedding(vocab_size, d_model)
        # self.pos_encoding = PositionalEncoding(d_model, dropout)
        # 创建六个带有相对位置编码的编码器基本块
        self.layers = nn.ModuleList(
            [EncoderBlockWithRPR(d_model, d_ff, head_num, clipping_distance, dropout) for _ in range(stack_num)])

    def forward(self, x, valid_lens):
        # encoder编码时,输入x是CNN输出的embedding
        # x = self.pos_encoding(self.embedding(x) * math.sqrt(self.d_model))
        # 依次使用六个编码器基本块对x进行编码
        for layer in self.layers:
            x = layer(x, valid_lens)

        return x


class CodeEncoder(nn.Module):
    def __init__(self, code_embedding, d_model, d_ff, head_num, encoder_layer_num, clipping_distance, dropout=0.1):
        """
        Code Encoder
        :param code_embedding: nn.Module 对Code数据进行嵌入
        :param d_model: 模型的维度，也是QKV的维度
        :param d_ff: 前馈神经网络的隐藏层维度
        :param head_num: 多头注意力的头数
        :param encoder_layer_num: 编码器堆叠层数
        :param clipping_distance: 相对位置编码的剪切距离
        :param dropout: Dropout率
        """
        super(CodeEncoder, self).__init__()
        # 源代码嵌入层
        self.code_embedding = code_embedding
        # 具有相对位置编码的编码器
        self.code_encoder = EncoderWithRPR(d_model, d_ff, head_num, clipping_distance, encoder_layer_num, dropout)
        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, source_code, source_code_len):
        """

        :param source_code: (batch size, sequence length)
        :param source_code_len: (batch size, )
        :return: source_code_enc: (batch size, sequence length, num of hidden)
        :return: source_code_len: (batch size, )
        """
        # source_code_embed(batch size, sequence length, num of hidden)
        source_code_embed = self.dropout(self.code_embedding(source_code))
        # source_code_enc: (batch size, sequence length, num of hidden)
        source_code_enc = self.code_encoder(source_code_embed, source_code_len)
        return source_code_enc, source_code_len


class CommentEncoder(nn.Module):
    def __init__(self, comment_embedding, pos_encoding, d_model, d_ff, head_num, encoder_layer_num, dropout=0.1):
        """
        Comment Encoder
        :param comment_embedding: nn.Module 对Comment数据进行嵌入
        :param pos_encoding: 位置编码层
        :param d_model: 模型的维度，也是Q、K、V的维度
        :param d_ff: 前馈神经网络的隐藏层维度
        :param head_num: 多头注意力中的头数
        :param encoder_layer_num: 编码器堆叠层数
        :param dropout: Dropout率
        """
        super(CommentEncoder, self).__init__()
        # 注释文本嵌入层
        self.comment_embedding = comment_embedding
        # 位置编码层
        self.pos_encoding = pos_encoding
        # 注释编码器
        self.template_encoder = Encoder(d_model, d_ff, head_num, encoder_layer_num, dropout)
        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, comment, comment_len):
        """

        :param comment: (batch size, sequence length)
        :param comment_len: (batch size, )
        :return: comment_enc: (batch size, sequence length, num of hidden)
        :return: comment_len: (batch size, )
        """
        b_, seq_comment_num = comment.size()
        # 创建位置编码张量
        comment_pos = torch.arange(seq_comment_num, device=comment.device).repeat(b_, 1)
        # 对文本注释进行嵌入和位置编码
        comment_embed = self.pos_encoding(self.comment_embedding(comment), comment_pos)
        # 对注释文本进行进一步编码
        comment_enc = self.template_encoder(comment_embed, comment_len)

        return comment_enc, comment_len


class FuncNameEncoder(nn.Module):
    def __init__(self, comment_embedding, d_model, d_ff, head_num, encoder_layer_num, dropout=0.1):
        """
        Keywords Encoder
        :param comment_embedding: nn.Module 对FuncName数据进行嵌入
        :param d_model: 模型的维度，也是Q、K、V的维度
        :param d_ff: 前馈神经网络的隐藏层维度
        :param head_num: 多头注意力中的头数
        :param encoder_layer_num: 编码器层数
        :param dropout: Dropout率
        """
        super(FuncNameEncoder, self).__init__()
        self.comment_embedding = comment_embedding
        self.keywords_encoder = Encoder(d_model, d_ff, head_num, encoder_layer_num, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, keywords, keywords_len):
        """
        :param keywords: (batch size, sequence length)
        :param keywords_len: (batch size, )
        :return: keywords_enc: (batch size, sequence length, num of hidden)
        :return: keywords_len: (batch size, )
        """
        keywords_embed = self.dropout(self.comment_embedding(keywords))
        keywords_enc = self.keywords_encoder(keywords_embed, keywords_len)

        return keywords_enc, keywords_len


class AstEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, d_model, d_ff, batch_size, head_num, encoder_layer_num,
                 dropout=0.1):
        """
        AST Encoder GCN version
        :param input_dim: 输入的节点特征维度
        :param hidden_dim: 隐层的节点特征维度，与input_dim一致
        :param output_dim: 输出的节点特征维度，与input_dim一致
        :param d_model: 模型维度， AstEmbedding输出的特征维度
        :param d_ff: 前馈神经网络的隐藏层维度
        :param batch_size: batch size
        :param head_num: 多头注意力中的头数
        :param encoder_layer_num: 编码器层数
        :param dropout: Dropout率
        """
        super(AstEncoder, self).__init__()
        self.ast_embedding = ASTEmbedding(input_dim, hidden_dim, output_dim, d_model, batch_size, dropout)
        self.ast_encoder = Encoder(d_model, d_ff, head_num, encoder_layer_num, dropout)

    def forward(self, x, a, a2, a3, a4, a5, ast_len=None):
        """

        :param x:
        :param a:
        :param a2:
        :param a3:
        :param a4:
        :param a5:
        :param ast_len: None
        :return:
        """
        ast_embed = self.ast_embedding(x, a, a2, a3, a4, a5)
        ast_enc = self.ast_encoder(ast_embed, ast_len)
        return ast_enc, ast_len


class AstEncoder2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, d_model, d_ff, batch_size, head_num, encoder_layer_num,
                 dropout=0.1):
        """
        AST Encoder GCN version
        :param input_dim: 输入的节点特征维度
        :param hidden_dim: 隐层的节点特征维度，与input_dim一致
        :param output_dim: 输出的节点特征维度，与input_dim一致
        :param d_model: 模型维度， AstEmbedding输出的特征维度
        :param d_ff: 前馈神经网络的隐藏层维度
        :param batch_size: batch size
        :param head_num: 多头注意力中的头数
        :param encoder_layer_num: 编码器层数
        :param dropout: Dropout率
        """
        super(AstEncoder2, self).__init__()
        self.ast_embedding = ASTEmbedding(input_dim, hidden_dim, output_dim, d_model, batch_size, dropout)
        self.ast_encoder = Encoder(d_model, d_ff, head_num, encoder_layer_num, dropout)

    def forward(self, x, a, a2, a3, a4, a5, ast_len=None):
        """

        :param x:
        :param a:
        :param a2:
        :param a3:
        :param a4:
        :param a5:
        :param ast_len: None
        :return:
        """
        ast_embed = self.ast_embedding(x, a, a2, a3, a4, a5)
        ast_enc = self.ast_encoder(ast_embed, ast_len)
        return ast_enc, ast_embed

if __name__ == "__main__":
    batch_size = 2
    qkv_num = 5
    hidden_num = 3
    x = torch.randn((batch_size, qkv_num, hidden_num))
    print(x)
    print(x.shape)

    d_model = 3
    d_ff = 3
    head_num = 3
    encoder_block = EncoderBlock(d_model, d_ff, head_num)
    valid_len = torch.arange(0, 2)
    print(valid_len.shape)
    y = encoder_block(x, valid_len)
    print(y.shape)
    print(y)
    encoder = EncoderWithRPR(d_model, d_ff, head_num, 2, 6, 0.1)
    z = encoder(x, valid_len)
    print(x)
    print(x.shape)
