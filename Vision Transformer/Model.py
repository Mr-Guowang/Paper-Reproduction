import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class attention(nn.Module):
    def __init__(self,dim ,heads = 8, dim_head = 64, dropout = 0.):#维度，多头数量，每个头的大小，dropout概率
        super(attention,self).__init__()
        inner_dim = dim_head * heads #这里计算一下多头转化前的维度，这样可以正好分为dim_head * heads
        self.heads = heads
        self.scale = dim **-0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        b, n, _, h = *x.shape, self.heads
        """
        x = torch.rand(3,4,5,6)
        a,b,_,d = x.shape
        print(d) 结果为6
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 这里将原始向量b*n*dim先转化为b*n*（inner_dim * 3），然后最后一维一份为三，化为元组,元组里每一个元素大小为
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        #此处qkv元组里的每一个元素大小为b n (dim_head * heads)，转化为b*heads*n*dim_heads
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  #求出不同向量之间关联度的分值
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')#这里把out变回b n (dim_head * heads)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class transformer(nn.Module):
    def __init__(self, dim,  heads, dim_head, mlp_dim, dropout):
        super(transformer, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attention1 = attention(dim=dim,heads=heads,dim_head=dim_head,dropout=dropout)
        self.feedword1 = FeedForward(dim=dim,hidden_dim=mlp_dim,dropout=dropout)
    def forward(self,x):
        out = x + self.attention1(self.norm1(x))
        out = out + self.feedword1(self.norm2(x))
        return out

class ViT(nn.Module):
    def __init__(self,image_size, patch_size, num_classes, dim, depth, heads, mlp_dim
                 , channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super(ViT, self).__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        patch_dim = channels * patch_size ** 2

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = transformer(dim,  heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def forward(self,img):
        p = self.patch_size
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x) #降维 此时x的大小为：batch （h方向个数*y方向个数）*（patch大小）
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)   #这里大小变为batch （h方向个数*y方向个数+1）*（patch大小）

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.mlp_head(x)
        return x


net = ViT(256, 16, 2, 64, 1, 2, 112
                 , channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.)

x = torch.randn(3,3,256,256)
print(net(x).shape)