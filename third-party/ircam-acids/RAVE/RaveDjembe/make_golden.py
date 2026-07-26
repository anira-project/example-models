"""Generate golden test vectors from the explicit-state wrappers.

The proof chain: test_rebuild.py shows the eager rebuild equals the stock
TorchScript export bit-exactly (RNG-synced); test_stateful_eager.py shows the
wrappers equal the eager model bit-exactly. The goldens are therefore
generated from the WRAPPERS (deterministic mean-encode, explicit fill input),
so runtime-specific parity tests (ONNX Runtime, TFLite, ExecuTorch) can run in
their own venvs without the .ts file or the rave source.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "rave"))

from stateful_rave import RATIO, build_eager_model, StatefulForward

HERE = Path(__file__).resolve().parent
BLOCK = 8 * RATIO      # 8 latent frames per call — the export default
N_BLOCKS = 8

torch.set_grad_enabled(False)


def make_audio(seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, 1, N_BLOCKS * BLOCK, generator=g) * 0.3).clamp(-1, 1)


def make_fills(shape, seed):
    torch.manual_seed(seed)
    return [torch.randn(shape) for _ in range(N_BLOCKS)]


def main():
    model, meta = build_eager_model()
    fwd = StatefulForward(model, meta, BLOCK)
    enc, dec = fwd.enc, fwd.dec
    n_frames = BLOCK // RATIO

    audio = make_audio(1234)
    audio_b = make_audio(2)
    fills = make_fills(fwd.fill_shape, 42)
    fills_b = make_fills(fwd.fill_shape, 200)
    fills_dec = make_fills(fwd.fill_shape, 7)

    def blocks(x):
        for i in range(N_BLOCKS):
            yield i, x[..., i * BLOCK:(i + 1) * BLOCK]

    state = torch.zeros(1, fwd.registry.size)
    fwd_ref = []
    for i, b in blocks(audio):
        y, state = fwd(b, state, fills[i])
        fwd_ref.append(y)
    fwd_ref = torch.cat(fwd_ref, -1)

    state = torch.zeros(1, fwd.registry.size)
    fwd_ref_b = []
    for i, b in blocks(audio_b):
        y, state = fwd(b, state, fills_b[i])
        fwd_ref_b.append(y)
    fwd_ref_b = torch.cat(fwd_ref_b, -1)

    state = torch.zeros(1, enc.registry.size)
    z_ref = []
    for i, b in blocks(audio):
        z, state = enc(b, state)
        z_ref.append(z)
    z_ref = torch.cat(z_ref, -1)

    state = torch.zeros(1, dec.registry.size)
    dec_ref = []
    for i in range(N_BLOCKS):
        z = z_ref[..., i * n_frames:(i + 1) * n_frames]
        y, state = dec(z, state, fills_dec[i])
        dec_ref.append(y)
    dec_ref = torch.cat(dec_ref, -1)

    out = HERE / "golden"
    out.mkdir(exist_ok=True)
    np.savez_compressed(
        out / "golden.npz",
        audio=audio.numpy(),
        audio_b=audio_b.numpy(),
        fill_fwd=np.stack([f.numpy() for f in fills]),
        fill_fwd_b=np.stack([f.numpy() for f in fills_b]),
        fill_dec=np.stack([f.numpy() for f in fills_dec]),
        forward_ref=fwd_ref.numpy(),
        forward_ref_b=fwd_ref_b.numpy(),
        latent_ref=z_ref.numpy(),
        decoder_ref=dec_ref.numpy(),
        block_size=np.int64(BLOCK),
        n_blocks=np.int64(N_BLOCKS),
        n_frames=np.int64(n_frames),
    )
    print(f"wrote {out / 'golden.npz'}"
          f" (audio {tuple(audio.shape)}, latent {tuple(z_ref.shape)})")


if __name__ == "__main__":
    main()
