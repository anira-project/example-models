"""Step 1 check: eager rebuild (with original cached-conv buffers) == .ts model.

Runs the same chunked audio through the TorchScript reference and the rebuilt
eager model (both starting from zeroed caches, RNG synced) and compares.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "rave"))

from stateful_rave import TS_PATH, RATIO, build_eager_model

torch.set_grad_enabled(False)


def zero_caches_ts(m):
    for name, buf in m.named_buffers():
        if name.endswith(".pad") or name.endswith(".cache"):
            buf.zero_()


def zero_caches_eager(m):
    for name, buf in m.named_buffers():
        if name.endswith(".pad") or name.endswith(".cache"):
            buf.zero_()


def eager_forward(model, meta, x):
    """Replicates VariationalScriptedRAVE.forward (deterministic variant)."""
    mb = model.pqmf(x)
    z = model.encoder(mb)
    mean, _ = torch.chunk(z, 2, 1)
    z = mean - model.latent_mean.unsqueeze(-1)
    z = F.conv1d(z, model.latent_pca.unsqueeze(-1))
    z = z[:, : meta["latent_size"]]

    zeros = torch.zeros(1, meta["full_latent_size"] - z.shape[1], z.shape[-1])
    z = torch.cat([z, zeros], 1)
    z = F.conv1d(z, model.latent_pca.t().unsqueeze(-1))
    z = z + model.latent_mean.unsqueeze(-1)
    y = model.decoder(z)
    audio = model.pqmf.inverse(y)
    return audio


def main():
    n_blocks = 8
    block = RATIO  # 2048
    torch.manual_seed(1234)
    audio = (torch.randn(1, 1, n_blocks * block) * 0.3).clamp(-1, 1)

    # reference
    ts = torch.jit.load(str(TS_PATH), map_location="cpu")
    zero_caches_ts(ts)
    torch.manual_seed(42)
    ref = []
    for i in range(n_blocks):
        ref.append(ts(audio[..., i * block:(i + 1) * block]))
    ref = torch.cat(ref, -1)

    # eager rebuild
    model, meta = build_eager_model()
    zero_caches_eager(model)
    torch.manual_seed(42)
    out = []
    for i in range(n_blocks):
        out.append(eager_forward(model, meta, audio[..., i * block:(i + 1) * block]))
    out = torch.cat(out, -1)

    diff = (ref - out).abs()
    print(f"ref rms      : {ref.pow(2).mean().sqrt():.6f}")
    print(f"max abs diff : {diff.max():.3e}")
    print(f"mean abs diff: {diff.mean():.3e}")
    assert diff.max() < 1e-5, "REBUILD MISMATCH"
    print("OK: eager rebuild matches TorchScript reference")


if __name__ == "__main__":
    main()
