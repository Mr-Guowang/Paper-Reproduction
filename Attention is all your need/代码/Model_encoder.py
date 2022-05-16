import torch
from torch import nn
import Model_small_function as SModel
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        #size，其实就是我们词嵌入维度的大小，第二个self_attn,第三个是feed_froward, 前馈全连接层
        super(EncoderLayer, self).__init__()
        # 首先将self_attn和feed_forward传入其中.
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 如图所示, 编码器层中有两个子层连接结构, 所以使用clones函数进行克隆
        self.sublayer = SModel.clones(SModel.SublayerConnection(size, dropout), 2)
        # 把size传入其中
        self.size = size
    def forward(self, x, mask=None):
        """forward函数中有两个输入参数，x和mask，分别代表上一层的输出，和掩码张量mask."""
        # 里面就是按照结构图左侧的流程. 首先通过第一个子层连接结构，其中包含多头自注意力子层，
        # 然后通过第二个子层连接结构，其中包含前馈全连接子层. 最后返回结果.
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
class Encoder(nn.Module):
    def __init__(self, layer, N):
        """初始化函数的两个参数分别代表编码器层和编码器层的个数"""
        super(Encoder, self).__init__()
        # 首先使用clones函数克隆N个编码器层放在self.layers中
        self.layers = SModel.clones(layer, N)
        # 再初始化一个规范化层, 它将用在编码器的最后面.
        self.norm = SModel.LayerNorm(layer.size)
    def forward(self, x, mask=None):
        """forward函数的输入和编码器层相同, x代表上一层的输出, mask代表掩码张量"""
        # 首先就是对我们克隆的编码器层进行循环，每次都会得到一个新的x，
        # 这个循环的过程，就相当于输出的x经过了N个编码器层的处理.
        # 最后再通过规范化层的对象self.norm进行处理，最后返回结果.
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

