import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

class FeedForward(nn.Module):
    # TODO: Implement a FFN with two linear layers. 
    #       Remeber to use Layer norm, activation and dropout at appropirate positions.
    #       You should ensure the input and output dim equals "dim".
    
    def __init__(self, dim, hidden_dim, dropout = 0.):
        '''
            Args: Please refer to the descriptions in `SimpleViT`.
        '''
        # raise NotImplementedError()
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # raise NotImplementedError()
        return self.net(x)

class Attention(nn.Module):
    # TODO: Implement the multi-head self-attention mechanism. 
    #       Ensure to include scaled dot-product attention and apply dropout and norm where necessary.
    #       In forward calculation, You may use Tensor.transpose, Tensor.view, ..., to handle tensor shapes. 
    #       You may also use einops.rearrange and einops.repeat to conveniently handle tensor shapes.
    def __init__(self, dim, heads, dim_head, dropout = 0.):
        '''
            Args: Please refer to the descriptions in `SimpleViT`.
        '''
        # raise NotImplementedError()
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        '''
        # raise NotImplementedError()
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        out = torch.matmul(self.attn_dropout(attn), v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerEncoder(nn.Module):
    # TODO: Implement the TransformerEncoder. Ensure that LayerNorm is applied at appropriate points within the architecture.
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        '''
            Args: Please refer to the descriptions in `SimpleViT`.
        '''
        # raise NotImplementedError()
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
            ]))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim). This tensor contains the transformed sequences after passing through the Transformer layers.
        '''
        # raise NotImplementedError()
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class SimpleViT(nn.Module):
    # TODO: Implement a Simple Vision Transformer (ViT) model.
    #       The following steps outline the basic procedure:
    #       1. Split the input image into non-overlapping patches.
    #       2. Embed each patch into a token by calculating its corresponding patch embedding.
    #       3. Add positional encodings to the patch embeddings. You can use either learnable positional encodings or other techniques.
    #       4. Pass the tokens through a Transformer encoder to extract hidden features.
    #       5. Obtain the final classification logits from the encoder's output. You can either use a class token for prediction, or simply take the mean of all patch tokens.
        
    def __init__(self,  dim, depth, heads, mlp_dim, dim_head, dropout, emb_dropout, patch_size, image_size=(32, 32), channels = 3, num_classes=10):
        '''
            Args:
                dim (int): Size of the patch embeddings.
                depth (int): Number of Transformer Encoder layers.
                heads (int): Number of attention heads in each Transformer layer.
                mlp_dim (int): Size of the hidden layer in the feed-forward network.
                dim_head (int): Size of each attention head.
                dropout (float): Dropout rate for the Transformer Encoder layers.
                emb_dropout (float): Dropout rate for the patch embeddings after adding positional encodings.
                image_size (tuple): Dimensions of the input image (height, width).
                patch_size (tuple): Dimensions of each patch (height, width).
                channels (int): Number of channels in the input image (e.g., 3 for RGB).
                num_classes (int): Number of output classes for classification.
        '''
        # raise NotImplementedError()
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])

        self.patch_embed = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        '''
            Args:
                img (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, num_classes). This contains the classification *logits* for each image in the batch.
        '''
        # raise NotImplementedError()
        x = self.patch_embed(img)
        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :x.shape[1]]
        x = self.dropout(x)
        x = self.encoder(x)
        return self.mlp_head(x[:, 0])
    
    def name(self):
        return "SimpleViT"
