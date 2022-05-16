import torch
import math
import torch.nn as nn
from torch.functional import F
import copy
DEVICE = torch.device('cpu')
class Embeddings(nn.Module):
 #模型中，再两个embedding层和pre-softmax线性变换层之间共享相同的权重矩阵,embedding层中，将权重乘以 sqrt(d_model)
    def __init__(self,vocab,d_model):          #vocab,d_model,词的数量，词向量维度
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
#PositionalEncoding位置编码是固定的生成之后，整个训练过程不改变
    def __init__(self, d_model, dropout, max_len=5000):
        #:param d_model: embedding的维数:param dropout: dropout概率:param max_len: 最大句子长度
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 初始化一个 max_len, d_model 维度的全零矩阵, 用来存放位置向量
        pe = torch.zeros(max_len, d_model, device=DEVICE)  # (max_len,1)
        position = torch.arange(0, max_len, device=DEVICE, dtype=torch.float).unsqueeze(1)
        # 使用exp log 实现sin/cos公式中的分母
        div_term = torch.exp(torch.arange(0, d_model, 2, device=DEVICE).float() *
                             (-math.log(10000.0) / d_model))
        # 填充 max_len, d_model 矩阵
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 方便和 一个batch的句子所有词embedding批量相加
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        self.pe.requires_grad_(False)
        x = x + self.pe[:, :x.size(1)]
        return x

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(d_k))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # mask掉矩阵中为0的值
    p_attn = F.softmax(scores, dim=-1)  # 每一行 softmax #TODO 计算每一行的softmax，按列
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        """初始化函数的输入参数有两个, d_model代表词嵌入维度, vocab_size代表词表大小."""
        super(Generator, self).__init__()
        # 首先就是使用nn中的预定义线性层进行实例化, 得到一个对象self.project等待使用,
        # 这个线性层的参数有两个, 就是初始化函数传进来的两个参数: d_model, vocab_size
        self.project = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        """前向逻辑函数中输入是上一层的输出张量x"""
        # 在函数中, 首先使用上一步得到的self.project对x进行线性变化,
        # 然后使用F中已经实现的log_softmax进行的softmax处理.
        # 在这里之所以使用log_softmax是因为和我们这个pytorch版本的损失函数实现有关, 在其他版本中将修复.
        # log_softmax就是对softmax的结果又取了对数, 因为对数函数是单调递增函数,
        # 因此对最终我们取最大的概率值没有影响. 最后返回结果即可.
        return F.log_softmax(self.project(x), dim=-1)

def clones(module, N):                #深拷贝模型，拷贝N份，放在一个列表里
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeaderAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
       #:param h: head的数目:param d_model: embed的维度:param dropout:
        super(MultiHeaderAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # 每一个head的维数
        self.h = h  # head的数量
        # 定义四个全连接函数 WQ,WK,WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # 保存attention结果
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):   #这里他们的输入是batch*词数*词向量维度
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # query维度(transpose之后)：batch_size, h, sequence_len, embedding_dim/h
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # 对query key value 计算 attention
        # attention 返回最后的x 和 atten weight
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 将多个头的注意力矩阵concat起来
        # 输入：x shape: batch_size, h, sequence_len, embed_dim/h(d_k)
        # 输出：x shape: batch_size, sequence_len, embed_dim
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)  # batch_size, sequence_len, embed_dim

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """初始化函数有三个输入参数分别是d_model, d_ff,和dropout=0.1，第一个是线性层的输入维度也是第二个线性层的输出维度，
           因为我们希望输入通过前馈全连接层后输入和输出的维度不变. 第二个参数d_ff就是第二个线性层的输入维度和第一个线性层的输出维度.
           最后一个是dropout置0比率."""
        super(PositionwiseFeedForward, self).__init__()

        # 首先按照我们预期使用nn实例化了两个线性层对象，self.w1和self.w2
        # 它们的参数分别是d_model, d_ff和d_ff, d_model
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        # 然后使用nn的Dropout实例化了对象self.dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """输入参数为x，代表来自上一层的输出"""
        # 首先经过第一个线性层，然后使用Funtional中relu函数进行激活,
        # 之后再使用dropout进行随机置0，最后通过第二个线性层w2，返回最终结果.
        return self.w2(self.dropout(F.relu(self.w1(x))))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """初始化函数有两个参数, 一个是features, 表示词嵌入的维度,
           另一个是eps它是一个足够小的数, 在规范化公式的分母中出现,
           防止分母为0.默认是1e-6."""
        super(LayerNorm, self).__init__()
        # 根据features的形状初始化两个参数张量a2，和b2，第一个初始化为1张量，
        # 也就是里面的元素都是1，第二个初始化为0张量，也就是里面的元素都是0，这两个张量就是规范化层的参数，
        # 因为直接对上一层得到的结果做规范化公式计算，将改变结果的正常表征，因此就需要有参数作为调节因子，
        # 使其即能满足规范化要求，又能不改变针对目标的表征.最后使用nn.parameter封装，代表他们是模型的参数。
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        # 把eps传到类中
        self.eps = eps
    def forward(self, x):
        """输入参数x代表来自上一层的输出"""
        # 在函数中，首先对输入变量x求其最后一个维度的均值，并保持输出维度与输入维度一致.
        # 接着再求最后一个维度的标准差，然后就是根据规范化公式，用x减去均值除以标准差获得规范化的结果，
        # 最后对结果乘以我们的缩放参数，即a2，*号代表同型点乘，即对应位置进行乘法操作，加上位移参数b2.返回即可.
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        """它输入参数有两个, size以及dropout， size一般是都是词嵌入维度的大小，
           dropout本身是对模型结构中的节点数进行随机抑制的比率，
           又因为节点被抑制等效就是该节点的输出都是0，因此也可以把dropout看作是对输出矩阵的随机置0的比率.
        """
        super(SublayerConnection, self).__init__()
        # 实例化了规范化对象self.norm
        self.norm = LayerNorm(size)
        # 又使用nn中预定义的droupout实例化一个self.dropout对象.
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, sublayer):
        """前向逻辑函数中, 接收上一个层或者子层的输入作为第一个参数，
           将该子层连接中的子层函数作为第二个参数"""
        # 我们首先对输出进行规范化，然后将结果传给子层处理，之后再对子层进行dropout操作，
        # 随机停止一些网络中神经元的作用，来防止过拟合. 最后还有一个add操作，
        # 因为存在跳跃连接，所以是将输入x与dropout后的子层输出结果相加作为最终的子层连接输出.
        return self.norm(x + self.dropout(sublayer(x)))


