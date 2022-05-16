import torch
from torch import nn
import Model_small_function as SModel
import Model_encoder
import Model_decoder
import copy
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        """初始化函数中有5个参数, 分别是编码器对象, 解码器对象,
           源数据嵌入函数, 目标数据嵌入函数,  以及输出部分的类别生成器对象
        """
        super(EncoderDecoder, self).__init__()
        # 将参数传入到类中
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator
    def forward(self, source, target, source_mask=None, target_mask=None):

        """在forward函数中，有四个参数, source代表源数据, target代表目标数据,
           source_mask和target_mask代表对应的掩码张量"""
        # 在函数中, 将source, source_mask传入编码函数, 得到结果后,
        # 与source_mask，target，和target_mask一同传给解码函数.
        return self.decode(self.encode(source, source_mask), source_mask,
                            target, target_mask)
    def encode(self, source, source_mask):
        """编码函数, 以source和source_mask为参数"""
        # 使用src_embed对source做处理, 然后和source_mask一起传给self.encoder
        return self.encoder(self.src_embed(source), source_mask)
    def decode(self, memory, source_mask, target, target_mask):
        """解码函数, 以memory即编码器的输出, source_mask, target, target_mask为参数"""
        # 使用tgt_embed对target做处理, 然后和source_mask, target_mask, memory一起传给self.decoder
        return self.generator(self.decoder(self.tgt_embed(target), memory, source_mask, target_mask))

def make_model(source_vocab, target_vocab, N=1,
               d_model=512, d_ff=1024, head=4, dropout=0.1):
    """该函数用来构建模型, 有7个参数，分别是源数据特征(词汇)总数，目标数据特征(词汇)总数，
       编码器和解码器堆叠数，词向量映射维度，前馈全连接网络中变换矩阵的维度，
       多头注意力结构中的多头数，以及置零比率dropout."""

    # 首先得到一个深度拷贝命令，接下来很多结构都需要进行深度拷贝，
    # 来保证他们彼此之间相互独立，不受干扰.
    c = copy.deepcopy
    # 实例化了多头注意力类，得到对象attn
    attn = SModel.MultiHeaderAttention(head, d_model)
    # 然后实例化前馈全连接类，得到对象ff
    ff = SModel.PositionwiseFeedForward(d_model, d_ff, dropout)
    # 实例化位置编码类，得到对象position
    position = SModel.PositionalEncoding(d_model, dropout)
    # 根据结构图, 最外层是EncoderDecoder，在EncoderDecoder中，
    # 分别是编码器层，解码器层，源数据Embedding层和位置编码组成的有序结构，
    # 目标数据Embedding层和位置编码组成的有序结构，以及类别生成器层.
    # 在编码器层中有attention子层以及前馈全连接子层，
    # 在解码器层中有两个attention子层以及前馈全连接层.
    model = EncoderDecoder(
        Model_encoder.Encoder(Model_encoder.EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Model_decoder.Decoder(Model_decoder.DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(SModel.Embeddings(source_vocab,d_model,), c(position)),
        nn.Sequential(SModel.Embeddings( target_vocab,d_model), c(position)),
        SModel.Generator(d_model, target_vocab))
    # 模型结构完成后，接下来就是初始化模型中的参数，比如线性层中的变换矩阵
    # 这里一但判断参数的维度大于1，则会将其初始化成一个服从均匀分布的矩阵，
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
# source = target = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])
#
# source_mask = target_mask = torch.zeros(2, 4, 4)
# model = make_model(1000,1000)
# y = model(source,target,source_mask,target_mask)
# print(y.shape)      #torch.Size([2, 4, 1000])






