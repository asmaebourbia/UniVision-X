"""E3: FastViT + word-level cross attention + ConvGRU + target-presence head.
Standalone relative to E2: it does not alter univision/model.py.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .model import ToyEncoder, MultiScaleFPN, ConvGRUCell


class UniVisionE3(nn.Module):
    def __init__(self, encoder="fastvit_t8.apple_in1k", width=48, text_dim=512,
                 pretrained=True, freeze_encoder=True, attention_size=28, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.encoder_name = encoder
        self.freeze_encoder = freeze_encoder
        self.attention_size = attention_size
        if encoder == "toy":
            self.encoder = ToyEncoder()
        else:
            import timm
            self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True)
        if freeze_encoder:
            self.encoder.requires_grad_(False)
        self.fpn = MultiScaleFPN(self.encoder.feature_info.channels(), width)
        self.text_projection = nn.Sequential(nn.Linear(text_dim, width), nn.LayerNorm(width))
        self.visual_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.attention_gain = nn.Parameter(torch.tensor(0.1))
        self.gru = ConvGRUCell(width)
        self.segmentation_head = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(width, 1, 1))
        self.presence_head = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU(inplace=True), nn.Linear(width, 1))
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1), persistent=False)
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder or not mode:
            self.encoder.eval()
        return self

    def forward(self, video, text_tokens, text_attention_mask, hidden=None):
        """video[B,T,3,H,W], text_tokens[B,L,512], mask[B,L] (1=valid).
        Return segmentation[B,T,1,H,W], presence[B,T], hidden[B,C,h,w].
        """
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError("Expected video [B,T,3,H,W]")
        if text_tokens.ndim != 3 or text_attention_mask.ndim != 2:
            raise ValueError("Expected token embeddings [B,L,D] and attention mask [B,L]")
        B,T,_,H,W = video.shape
        if text_tokens.shape[0] != B or text_attention_mask.shape != text_tokens.shape[:2]:
            raise ValueError("Batch or token dimension mismatch")
        if not text_attention_mask.bool().any(dim=1).all():
            raise ValueError("Every expression must contain at least one unmasked token")
        word = self.text_projection(text_tokens.float())
        key_padding_mask = ~text_attention_mask.bool()
        pooled_text = (word * text_attention_mask.float().unsqueeze(-1)).sum(1) / text_attention_mask.float().sum(1,keepdim=True).clamp_min(1)
        logits_list, presence_list = [], []
        for i in range(T):
            x = (video[:,i] - self.img_mean) / self.img_std
            feat = self.fpn(self.encoder(x))
            h,w = feat.shape[-2:]
            ah, aw = min(h,self.attention_size), min(w,self.attention_size)
            queries = F.adaptive_avg_pool2d(feat, (ah,aw)).flatten(2).transpose(1,2)
            context, _ = self.cross_attention(self.visual_norm(queries), word, word,
                                              key_padding_mask=key_padding_mask, need_weights=False)
            context = context.transpose(1,2).reshape(B,-1,ah,aw)
            if (ah,aw) != (h,w):
                context = F.interpolate(context,size=(h,w),mode="bilinear",align_corners=False)
            feat = feat + torch.tanh(self.attention_gain) * context
            hidden = self.gru(feat,hidden)
            seg = self.segmentation_head(hidden)
            seg = F.interpolate(seg, size=(H,W), mode="bilinear", align_corners=False)
            pooled_visual = hidden.mean((2,3))
            is_present = self.presence_head(torch.cat([pooled_visual, pooled_text],dim=-1)).squeeze(-1)
            logits_list.append(seg)
            presence_list.append(is_present)
        return torch.stack(logits_list,dim=1), torch.stack(presence_list,dim=1), hidden
