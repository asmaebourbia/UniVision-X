import tempfile
from pathlib import Path
import numpy as np
import torch
from univision.synthetic import make_dataset, rle_encode_uncompressed
from univision.data import MeViS, decode_rle
from univision.metrics import mask_iou, segmentation_loss, boundary_f
from univision.model import UniVisionX, ConvGRUCell
from univision.corruptions import corrupt


def test_rle_roundtrip():
    mask = np.zeros((9, 12), np.uint8)
    mask[2:6, 3:8] = 1
    assert np.array_equal(decode_rle(rle_encode_uncompressed(mask)), mask)


def test_corruption_is_deterministic():
    img = torch.ones(4, 3, 32, 32) * .5
    a = corrupt(img, "noise", 3, seed=91)
    b = corrupt(img, "noise", 3, seed=91)
    assert torch.equal(a, b)
    assert not torch.equal(img, a)


def test_data_model_and_backward():
    with tempfile.TemporaryDirectory() as tmp:
        make_dataset(tmp, image_size=64, n_train=2, n_val=1, nframes=5)
        ds = MeViS(tmp, "train", image_size=64, clip_length=2,
                   text_cache=str(Path(tmp)/"SYNTHETIC_TEXT_ONLY.pt"), train=True)
        sample = ds[0]
        assert sample["video"].shape == (2,3,64,64)
        assert sample["mask"].shape == (2,1,64,64)
        model = UniVisionX(encoder="toy", pretrained=False, width=16,
                           freeze_encoder=True, use_text=True, use_temporal=True)
        prediction, state = model(sample["video"].unsqueeze(0), sample["text_emb"].unsqueeze(0))
        assert prediction.shape == (1,2,1,64,64)
        assert state.shape[0] == 1
        loss = segmentation_loss(prediction, sample["mask"].unsqueeze(0))
        loss.backward()
        assert torch.isfinite(loss)
        assert model.head[-1].weight.grad is not None


def test_visual_only_and_empty_masks():
    with tempfile.TemporaryDirectory() as tmp:
        make_dataset(tmp, image_size=48, n_train=1, n_val=1, nframes=3)
        ds = MeViS(tmp, "valid_u", image_size=48, clip_length=2, visual_only=True, full_video=True)
        sample = ds[0]
        model = UniVisionX(encoder="toy", pretrained=False, width=16,
                           use_text=False, use_temporal=False)
        out, h = model(sample["video"].unsqueeze(0))
        assert out.shape == (1,3,1,48,48)
        assert h is None
        assert mask_iou(np.zeros((4,4)), np.zeros((4,4))) == 1
        assert boundary_f(np.zeros((4,4)), np.zeros((4,4))) == 1
