# UniVision-X — Architecture and model behavior

This note documents the implemented **E2 baseline** and **E3 upgraded model**, their actual input/output interfaces, and the experimental rationale. The diagrams and claims refer to the code in `univision/model.py` and `univision/e3_model.py`; no unimplemented foundation-model components are represented as part of these models.

## Complete two-track architecture

![E2 versus E3 architecture](../assets/architecture_e2_e3.svg)

[Editable Graphviz diagram source](diagrams/model_architecture.dot)

## Components shared by E2 and E3

### FastViT image encoder

- Implemented through `timm.create_model('fastvit_t8.apple_in1k', pretrained=True, features_only=True)`.
- Extracts a list of feature maps at different spatial scales.
- Remains frozen in the reported training runs; its batch-normalization statistics are not updated.
- Raw RGB frames are normalized with standard ImageNet means and standard deviations in the model forward pass.

### Feature fusion — `MultiScaleFPN`

A lateral `1 × 1` convolution projects each backbone feature map to a fixed width. Starting from the coarsest scale, the module bilinearly upsamples the representation and adds the corresponding lateral feature before a `3 × 3` convolution, batch normalization and ReLU.

### Temporal state — `ConvGRUCell`

A convolutional gated recurrent unit updates one hidden feature map per incoming frame. At full-video evaluation, the state is passed from one small inference chunk to the next (`hidden.detach()` across chunks), retaining sequential context without storing a full video in GPU memory.

This is temporal feature modeling, **not explicit object identity tracking**. The code does not implement a learned optical-flow field or a video-level object association graph.

## E2 — FiLM from pooled CLIP text features

| Step | Implementation | Purpose |
|---|---|---|
| Language | A pooled CLIP embedding of shape `[B, 512]` | Global sentence context |
| Visual | FastViT + FPN feature map `[B, C, h, w]` | Semantic and spatial clues |
| Conditioning | Linear projection of the sentence vector to scale/shift (`2C`) | Modulate each visual channel |
| Temporal | ConvGRU over successive conditioned feature maps | Propagate video context |
| Segmentation | Light 2D convolutional head | Predict one binary mask per frame |

The conditioning is channel-wise and global. It does **not directly attend from each image location to individual text tokens**. Consequently, relations such as *"person feeding panda"* and distinctions between similar instances remain challenging.

## E3 — Token-level cross-attention

The updated E3 network accepts:

- Video tensor: `video[B, T, 3, H, W]`.
- Cached CLIP token hidden states: `text_tokens[B, L, 512]`.
- Token attention mask: `text_attention_mask[B, L]`.
- Optional recurrent state from an earlier chunk.

### A. Language pathway

`univision/e3_text_cache.py` precomputes the CLIP token hidden states, padded/truncated to **32 tokens** for the reported experiment. During model training, the CLIP text model is not fine-tuned; a learnable linear projection + LayerNorm maps cached states to model width **48**. The tokens also produce a masked average for presence estimation.

**Clarification:** *token-level states* include the tokenizer's special tokens as handled by the stored attention mask. They are not independent linguistic parses, manually annotated words or language-model reasoning traces.

### B. Spatial cross-attention

For each frame:

1. FPN creates visual feature maps of shape `[B, 48, h, w]`.
2. Adaptive pooling produces a spatial query grid of up to `28 × 28` locations.
3. Each visual query attends over the projected CLIP tokens with `nn.MultiheadAttention` (4 heads): visual queries, text keys and text values.
4. The resulting context features are reshaped, bilinearly upsampled to `[h, w]` and **added residually** to the FPN map, scaled by a learnable `tanh(attention_gain)`.
5. The conditioned features enter the ConvGRU.

This is a **spatially varying language conditioning** mechanism. The implementation is not an object-query detector, a full multimodal transformer decoder or a token-supervised grounder.

### C. Per-frame mask and presence heads

The recurrent feature map goes to:

- **Segmentation head:** a `3 × 3` convolution + ReLU + `1 × 1` convolution, then bilinear upsampling to the input image size.
- **Presence head:** concatenate global-average-pooled recurrent visual features with masked-average text features; an MLP outputs one presence logit for the current frame.

During inference in `univision/e3_evaluate.py`, the final binary mask is:

```text
raw_mask  = sigmoid(segmentation_logit) > 0.5
is_visible = sigmoid(presence_logit) > 0.5
final_mask = raw_mask AND is_visible
```

The mask is also saved to original image dimensions using nearest-neighbor interpolation. **Local reported J and Dice were calculated earlier at the model's resized resolution.**

## Supervision and checkpointing

- Training uses clips of **3 frames** at **224 × 224** resolution with batch size **1**.
- E3 optimizes a segmentation loss and a supervised per-frame target-presence loss (`presence_weight=0.5`).
- FastViT and cached CLIP stay frozen.
- A pretrained encoder plus learnable FPN/attention/ConvGRU/heads is trained on MeViS `train`.
- Epoch selection uses validation clip quality and false positives, **not** the full-video IoU table.
- For the reported run, the best checkpoint is **epoch 6 of 12**.

## How to test individual contributions scientifically

The current results support a *system-level* E2 → E3 gain, not causal credit for an individual module. Future controlled ablations should hold training augmentations, epoch budget, validation sampling, seed and encoder fixed while toggling:

1. pooled FiLM vs. token-level attention **without a presence head**;
2. presence-head gating on/off for the **same attention model**;
3. ConvGRU vs. independent-frame inference;
4. training degradations on/off;
5. multiple seeds with uncertainty grouped by video.

These are **proposed experiments**, not reported results.
