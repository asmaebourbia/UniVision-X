"""FastViT + language FiLM + temporal ConvGRU + lightweight FPN."""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyEncoder(nn.Module):
    """Offline smoke-test encoder. NOT a research baseline; use FastViT in experiments."""
    def __init__(self):
        super().__init__()
        widths = [16, 24, 32, 48]
        layers = []
        c = 3
        for w in widths:
            layers.append(nn.Sequential(nn.Conv2d(c, w, 3, 2, 1), nn.BatchNorm2d(w), nn.ReLU()))
            c = w
        self.layers = nn.ModuleList(layers)
        self.feature_info = type("FeatureInfo", (), {"channels": lambda self: widths})()

    def forward(self, x):
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


class LoRAConv1x1(nn.Module):
    """Trainable low-rank delta for eligible pointwise convs; base conv stays frozen."""
    def __init__(self, base: nn.Conv2d, rank=4, scale=1.0):
        super().__init__()
        if base.kernel_size != (1, 1) or base.groups != 1 or base.stride != (1, 1):
            raise ValueError("LoRAConv1x1 supports ungrouped stride-1 pointwise convolutions")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.down = nn.Conv2d(base.in_channels, rank, 1, bias=False)
        self.up = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)
        self.scale = scale / rank

    def forward(self, x):
        return self.base(x) + self.up(self.down(x)) * self.scale


def attach_conv_lora(module, rank=4, max_modules=0):
    """Apply LoRA to all eligible 1x1 convolutions or first max_modules if > 0."""
    updated = [0]
    def visit(root):
        for key, child in list(root.named_children()):
            if isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1) and child.groups == 1 and child.stride == (1, 1) and (max_modules == 0 or updated[0] < max_modules):
                setattr(root, key, LoRAConv1x1(child, rank=rank))
                updated[0] += 1
            elif not isinstance(child, LoRAConv1x1):
                visit(child)
    visit(module)
    if not updated[0]:
        raise ValueError("No eligible pointwise Conv2d layer found for LoRA")
    return updated[0]


class ConvBNReLU(nn.Sequential):
    def __init__(self, inc, outc):
        super().__init__(nn.Conv2d(inc, outc, 3, padding=1, bias=False), nn.BatchNorm2d(outc), nn.ReLU(inplace=True))


class ConvGRUCell(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gates = nn.Conv2d(channels * 2, channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(channels * 2, channels, 3, padding=1)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros_like(x)
        z, r = self.gates(torch.cat([x, h], 1)).sigmoid().chunk(2, 1)
        n = torch.tanh(self.candidate(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


class MultiScaleFPN(nn.Module):
    def __init__(self, feature_channels, width):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, width, 1) for c in feature_channels])
        self.refine = ConvBNReLU(width, width)

    def forward(self, features):
        p = self.lateral[-1](features[-1])
        for idx in range(len(features) - 2, -1, -1):
            p = F.interpolate(p, size=features[idx].shape[-2:], mode="bilinear", align_corners=False)
            p = p + self.lateral[idx](features[idx])
        return self.refine(p)


class UniVisionX(nn.Module):
    def __init__(self, encoder="fastvit_t8.apple_in1k", pretrained=True, width=64,
                 text_dim=512, use_text=True, use_temporal=True,
                 freeze_encoder=True, lora_rank=0):
        super().__init__()
        self.encoder_name = encoder
        self.freeze_encoder = freeze_encoder
        if encoder == "toy":
            self.encoder = ToyEncoder()
        else:
            try:
                import timm
            except ImportError as exc:
                raise ImportError("Install timm for FastViT: pip install timm") from exc
            self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        if lora_rank > 0:
            if encoder == "toy":
                raise ValueError("LoRA experiment requires real FastViT, not toy encoder")
            self.n_lora = attach_conv_lora(self.encoder, rank=lora_rank)
        else:
            self.n_lora = 0
        channels = self.encoder.feature_info.channels()
        self.fpn = MultiScaleFPN(channels, width)
        self.use_text = use_text
        self.use_temporal = use_temporal
        self.text_gate = nn.Linear(text_dim, width * 2) if use_text else None
        if self.text_gate is not None:
            nn.init.zeros_(self.text_gate.weight)
            nn.init.zeros_(self.text_gate.bias)
        self.gru = ConvGRUCell(width) if use_temporal else None
        self.head = nn.Sequential(ConvBNReLU(width, width), nn.Conv2d(width, 1, 1))
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        # A frozen feature extractor must keep its BatchNorm running stats fixed.
        if self.freeze_encoder or not self.training:
            self.encoder.eval()
        return self

    def forward(self, video, text_emb: Optional[torch.Tensor] = None, hidden=None):
        if video.ndim != 5:
            raise ValueError("Expected video tensor [B,T,3,H,W]")
        b, t, c, H, W = video.shape
        if c != 3:
            raise ValueError("Expected RGB frames")
        if self.use_text and text_emb is None:
            raise ValueError("Text embeddings needed for --use-text model")
        film = self.text_gate(text_emb.float()) if self.use_text else None
        outputs = []
        for i in range(t):
            x = (video[:, i] - self.img_mean) / self.img_std
            feats = self.encoder(x)
            feat = self.fpn(feats)
            if film is not None:
                scale, shift = film.chunk(2, dim=1)
                feat = feat * (1 + 0.1 * torch.tanh(scale)[:, :, None, None]) + shift[:, :, None, None]
            if self.use_temporal:
                hidden = self.gru(feat, hidden)
                feat = hidden
            logits = self.head(feat)
            outputs.append(F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False))
        return torch.stack(outputs, 1), hidden
