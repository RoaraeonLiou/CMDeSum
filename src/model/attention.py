import math
import torch
from torch import nn


def transpose_qkv(x: torch.Tensor, num_heads: int):
    """
    # Need Description:
    :param x: (batch size, num of qkv, num of hidden)
    :param num_heads: num of heads
    :return: Tensor(batch size*num of heads, num of qkv, num of hidden/num of heads)
    """
    # 分离出多头注意力每个头qkv的隐层单元
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    # 将头数所在维度提前
    x = x.permute(0, 2, 1, 3)
    # 将头数和batch size整合
    return x.reshape(-1, x.shape[2], x.shape[3])


def transpose_output(x: torch.Tensor, num_heads: int):
    """
    # Need Description:
    :param x: (batch size*num of heads, num of qkv, num of hidden/num of heads)
    :param num_heads: num of heads
    :return: Tensor(batch size, num of qkv, num of hidden)
    """
    # 将头数和batch size拆分
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
    # 将头数所在维度后移
    x = x.permute(0, 2, 1, 3)
    # 将每个头的隐层单元进行整合
    return x.reshape(x.shape[0], x.shape[1], -1)


def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.
    带有掩码的softmax
    Defined in :numref:`sec_attention-scoring-functions`"""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        # 未设置valid lens， 直接对X进行softmax
        return nn.functional.softmax(X, dim=-1)
    else:
        # 获取X的形状
        shape = X.shape
        if valid_lens.dim() == 1:
            # 如果valid lens只有一维，该维度表示针对每一条数据的有效长度
            # valid_len(batch size * num of qkv,)
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # 如果valid lens有二维，第一维表示每一条数据的有效长度，第二条表示对每一个词的有效长度
            # valid lens(batch size * num of qkv, )
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        # 首先将X按照token展开，然后执行序列掩码
        # X(batch size * num of qkv, num of qkv)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        # X(batch size, num of qkv, num of qkv)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.
    序列掩码
    Defined in :numref:`sec_seq2seq_decoder`"""
    # 获取序列token Embedding最大长度
    max_len = X.size(1)
    # 创建掩码张量，将序列有效长度之后的元素设置为False，之前的设置为True
    mask = torch.arange((max_len), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    # 使用掩码将无关元素替代为指定的值（默认为0）
    X[~mask] = value
    return X


class DotProductAttention(nn.Module):
    """Scaled dot product attention.

    Defined in :numref:`subsec_additive-attention`"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, queries, keys, values, valid_lens=None):
        """

        :param queries: (batch size, num of qkv, num of hidden)
        :param keys: (batch size, num of qkv, num of hidden)
        :param values: (batch size, num of qkv, num of hidden)
        :param valid_lens: (batch size, ) or (batch size, num of qkv)
        :return: Tensor(batch size, num of qkv, num of hidden)
        """
        # 获取num of hidden
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        # 获取注意力得分
        # scores: (batch size, num of qkv, num of qkv)
        # = (batch size, num of qkv, num of hidden) * (batch size, num of hidden, num of qkv)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # 掩码计算，获得注意力权重
        # attention_weights: (batch size, num of qkv, num of qkv)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # 返回注意力得分
        # (batch size, num of qkv, num of hidden)
        # = (batch size, num of qkv, num of qkv) * (batch size, num of qkv, num of hidden)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """Multi-head attention.

    Defined in :numref:`sec_multihead-attention`"""

    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """

        :param queries: (batch size, num of qkv, num of hidden)
        :param keys: (batch size, num of qkv, num of hidden)
        :param values: (batch size, num of qkv, num of hidden)
        :param valid_lens: (batch size, ) or (batch size, num of qkv)
        :return:Tensor(batch size, num of qkv, num of hidden)
        """
        # 将传入的qkv按照多头进行转换
        # (batch size, num of qkv, num of hidden) -> (batch size*num of heads, num of qkv, num of hidden/num of heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            # valid_lens: (batch size * num of heads,) or (batch size * num of heads, num of qkv)
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # output: (batch size * num of heads, num of qkv, num of hidden/num of heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat: (batch size, num of qkv, num of hidden)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class DotProductAttentionWithRPR(nn.Module):
    # Relative Position Representations 相对位置表示
    """Scaled dot product attention.
    带有相对位置表示的点积注意力机制
    Defined in :numref:`subsec_additive-attention`"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttentionWithRPR, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, queries, keys, values, pos_k, pos_v, valid_lens=None):
        """
        带有相对位置表示的点积注意力前向传播
        :param queries: (batch size * num of heads, num of qkv, num of hidden / num of heads)
        :param keys: (batch size * num of heads, num of qkv, num of hidden / num of heads)
        :param values: (batch size * num of heads, num of qkv, num of hidden / num of heads)
        :param pos_k: (num of qkv, num of qkv, num of hidden / num of heads)
        :param pos_v: (num of qkv, num of qkv, num of hidden / num of heads)
        :param valid_lens: (batch size, ) or (batch size, num of qkv)
        :return: Tensor (batch size * num of heads, num of qkv, num of hidden / num of heads)
        """
        """
        获取隐层维度
        """
        d = queries.shape[-1]

        """
        计算Query-Key得分和相对位置得分，并生成最终得分
        """
        # score (batch size * num of heads, num of qkv, num of qkv)
        scores = torch.bmm(queries, keys.transpose(1, 2))
        # scores_pos (batch size * num of heads, num of qkv, num of qkv)
        scores_pos = torch.bmm(queries.transpose(0, 1), pos_k.transpose(1, 2)).transpose(0, 1)
        # score (batch size * num of heads, num of qkv, num of qkv)
        scores = (scores + scores_pos) / math.sqrt(d)

        """
        通过Query-Key的分计算注意力权重
        """
        # attention_weights (batch size * num of heads, num of qkv, num of qkv)
        self.attention_weights = masked_softmax(scores, valid_lens)

        """
        利用生成的注意力权重输出的分数和相对位置分数，求和后输出
        """
        # output (batch size * num of heads, num of qkv, num of hidden / num of heads)
        output = torch.bmm(self.dropout(self.attention_weights), values)
        # output_pos (batch size * num of heads, num of qkv, num of hidden / num of heads)
        output_pos = torch.bmm(self.dropout(self.attention_weights.transpose(0, 1)), pos_v).transpose(0, 1)
        # return (batch size * num of heads, num of qkv, num of hidden / num of heads)
        return output + output_pos


class MultiHeadAttentionWithRPR(nn.Module):
    """Multi-head attention.
    Defined in :numref:`sec_multihead-attention`"""

    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, clipping_distance,
                 dropout, bias=False, **kwargs):
        """
        初始化带有相对位置表示的多头注意力机制单元
        :param key_size: Key维度
        :param query_size: Query维度
        :param value_size: Value维度
        :param num_hiddens: 隐层维度
        :param num_heads: 多头注意力头数
        :param clipping_distance: 相对位置的视野半径
        :param dropout: dropout率
        :param bias: 偏置
        :param kwargs: 其他参数
        """
        super(MultiHeadAttentionWithRPR, self).__init__()
        self.num_heads = num_heads
        self.attention_rpr = DotProductAttentionWithRPR(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.relative_pos_v = nn.Embedding(2 * clipping_distance + 1, num_hiddens // num_heads)
        self.relative_pos_k = nn.Embedding(2 * clipping_distance + 1, num_hiddens // num_heads)
        self.clipping_distance = clipping_distance

    def forward(self, queries, keys, values, valid_lens):
        """
        带有相对位置表示的多头点积注意力前向传播
        相对位置用于在计算注意力得分时，使得当前query关注一定范围内的信息
        :param queries: (batch size, num of qkv, num of hidden)
        :param keys: (batch size, num of qkv, num of hidden)
        :param values: (batch size, num of qkv, num of hidden)
        :param valid_lens: (batch size, ) or (batch size, num of qkv), 用于裁剪多余的qkv
        :return: Tensor (batch size, num of qkv, num of hidden)
        """
        """
        多头注意力转换，方便并行计算
        """
        # (batch size, num of qkv, num of hidden) ->  (batch size*num of heads, num of qkv, num of hidden/num of heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        """
        生成相对位置矩阵
        """
        # range_queries (num of qkv, )
        range_queries = torch.arange(queries.size(1), device=queries.device)
        # range_keys (num of qkv, )
        range_keys = torch.arange(keys.size(1), device=keys.device)
        # distance_mat (num of qkv, num of qkv)
        distance_mat = range_keys[None, :] - range_queries[:, None]
        # distance_mat_clipped (num of qkv, num of qkv)
        # 使用clamp函数，将distance_mat的值限制在(-clipping_distance, +clipping_distance)之间
        # 然后加上clipping_distance保证所有的值均为非负值
        distance_mat_clipped = torch.clamp(distance_mat, -self.clipping_distance,
                                           self.clipping_distance) + self.clipping_distance

        # pos_k (num of qkv, num of qkv, num of hidden / num of heads)
        pos_k = self.relative_pos_k(distance_mat_clipped)
        # pos_v (num of qkv, num of qkv, num of hidden / num of heads)
        pos_v = self.relative_pos_v(distance_mat_clipped)

        """
        生成有效长度矩阵
        """
        # valid_lens (batch size * num of heads, ) or (batch size * num of heads, num of qkv)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        """
        计算带有相对位置表示的点积注意力得分
        """
        # output (batch size * num of heads, num of qkv, num of hidden / num of heads)
        output = self.attention_rpr(queries, keys, values, pos_k, pos_v, valid_lens)

        """
        多头注意力输出转换
        """
        # output_concat (batch size, num of qkv, num of hidden)
        output_concat = transpose_output(output, self.num_heads)
        # return (batch size, num of qkv, num of hidden)
        return self.W_o(output_concat)


class Test(object):
    def __init__(self):
        super(Test, self).__init__()

    def test_func(self):
        t = torch.range(0, 24 - 1)
        t = t.reshape(2, 3, 4)
        t = transpose_qkv(t, 2)
        print(t)
        t = transpose_output(t, 2)
        print(t)

    def test_cal(self):
        range_queries = torch.arange(4)
        print(range_queries)
        print(range_queries.shape)
        print(range_queries[None, :])
        print(range_queries[None, :].shape)
        range_keys = torch.arange(4)
        print(range_keys[:, None])
        distance_mat = range_keys[None, :] - range_queries[:, None]
        print(distance_mat)


if __name__ == "__main__":
    tester = Test()
    tester.test_cal()
