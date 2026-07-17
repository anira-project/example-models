"""Generate golden test vectors from the TorchScript reference model.

Saves all inputs (audio blocks, noise blocks, latents) and reference outputs
to export/golden/golden.npz so runtime-specific parity tests (TFLite,
ExecuTorch) can run in their own venvs without loading the .ts file.

RNG note: the reference consumes torch.rand internally for the decoder noise;
we seed it and generate the identical noise sequence for the explicit
`noise_in` inputs (same trick as test_onnxruntime.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TS_PATH = HERE / "Djembe_deterministic.ts"
META = json.loads((HERE / "models" / "rave_meta.json").read_text())
BLOCK = META["block_size"]
NOISE_SHAPE = tuple(META["noise_shape"])
N_BLOCKS = 16

torch.set_grad_enabled(False)


def fresh_ts():
    ts = torch.jit.load(str(TS_PATH), map_location="cpu")
    for name, buf in ts.named_buffers():
        if name.endswith(".pad") or name.endswith(".cache"):
            buf.zero_()
    return ts


def noise_seq(seed):
    torch.manual_seed(seed)
    return np.stack(
        [(torch.rand(NOISE_SHAPE) * 2 - 1).numpy() for _ in range(N_BLOCKS)])


def main():
    g = torch.Generator().manual_seed(1234)
    audio = (torch.randn(1, 1, N_BLOCKS * BLOCK, generator=g) * 0.3).clamp(-1, 1)
    blocks = [audio[..., i * BLOCK:(i + 1) * BLOCK] for i in range(N_BLOCKS)]

    # forward reference (seed 42)
    ts = fresh_ts()
    torch.manual_seed(42)
    fwd_ref = torch.cat([ts(b) for b in blocks], -1)

    # encoder reference
    ts = fresh_ts()
    z_ref = torch.cat([ts.encode(b) for b in blocks], -1)

    # decoder reference on those latents (seed 7)
    n_frames = META["latent_frames_per_block"]
    ts = fresh_ts()
    torch.manual_seed(7)
    dec_ref = torch.cat(
        [ts.decode(z_ref[..., i * n_frames:(i + 1) * n_frames])
         for i in range(N_BLOCKS)], -1)

    # second, different stream for the state-isolation (interleaving) test
    g = torch.Generator().manual_seed(2)
    audio_b = (torch.randn(1, 1, N_BLOCKS * BLOCK, generator=g) * 0.3).clamp(-1, 1)
    blocks_b = [audio_b[..., i * BLOCK:(i + 1) * BLOCK] for i in range(N_BLOCKS)]
    ts = fresh_ts()
    torch.manual_seed(200)
    fwd_ref_b = torch.cat([ts(b) for b in blocks_b], -1)

    out = HERE / "golden"
    out.mkdir(exist_ok=True)
    np.savez_compressed(
        out / "golden.npz",
        audio=audio.numpy(),
        audio_b=audio_b.numpy(),
        noise_fwd_b=noise_seq(200),
        forward_ref_b=fwd_ref_b.numpy(),
        noise_fwd=noise_seq(42),
        noise_dec=noise_seq(7),
        forward_ref=fwd_ref.numpy(),
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
