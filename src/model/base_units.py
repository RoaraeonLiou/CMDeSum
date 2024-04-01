from torch import nn
import torch.nn.functional as F


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        位置感知的前馈网络
        `d_model`: 输入和输出的特征维度
        `d_ff`: 前馈网络隐藏层的特征维度
        `dropout`: Dropout概率，默认0.1
        """
        super(PositionWiseFFN, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播函数
        X(Tensor): 输入张量，形状为（批量大小*头数，序列长度，嵌入维度）
        """
        # 使用第一个线性变换层，然后使用ReLUctant激活函数和Dropout
        # 使用第二个线性变换层后输出
        return self.W_2(self.dropout(F.relu(self.W_1(x))))


class AddNorm(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1):
        """
        创建一个添加和归一化操作模块
        `embedding_dim`: 输入和输出的特征维度
        """
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(embedding_dim)  # 层归一化
        self.dropout = nn.Dropout(dropout)  # Dropout层

    def forward(self, x, y):
        """
        前向传播函数
        X(Tendor): 输入张量，形状为（批量大小*头数，序列长度，嵌入维度）
        Y(Tensor): 输入张量，形状与X相同
        """
        # 执行残差连接
        # 使用LayerNorm进行层归一化
        return self.norm(x + self.dropout(y))


class KeywordsEncoding(nn.Module):
    """
    ???
    """
    def __init__(self, d_model, dropout, keywords_type=6):
        """
        创建一个关键词编码操作模块。
        `d_model` (int): 输入和输出张量的特征维度。
        `dropout` (float): Dropout 概率，用于关键词编码后的张量。
        `keywords_type` (int, optional): 关键词类型的数量，默认为 6。
        """
        super(KeywordsEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.type_embedding = nn.Embedding(keywords_type, d_model) # 关键词类型嵌入矩阵

    def forward(self, x, keywords_type):
        """
        前向传播函数
        x(Tensor): 输入张量，形状为（批量大小*头数，序列长度，嵌入维度）
        keywords_type(Tensor): 关键词类型张量，形状与x相同，表示每个元素的关键词类型
        """
        # 将关键词类型编码嵌入添加到输入张量中
        x = x + self.type_embedding(keywords_type)
        return self.dropout(x)