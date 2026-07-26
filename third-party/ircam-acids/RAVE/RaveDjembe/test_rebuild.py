"""Step 1 check: eager rebuild (with original cached-conv buffers) == .ts model.

The stock export is stochastic: encode samples mean + eps * std
(randn_like) and decode fills the truncated latent dims with randn. With a
synced seed the eager chain draws the SAME numbers in the SAME order, so
parity here is exact — proving weights, structure, causal padding and the
latent glue all match.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "rave"))

from stateful_rave import TS_PATH, RATIO, build_eager_model

torch.set_grad_enabled(False)


def zero_caches(m):
    for name, buf in m.named_buffers():
        if name.endswith(".pad") or name.endswith(".cache"):
            buf.zero_()


def eager_forward_stochastic(model, meta, x):
    """Replicates VariationalScriptedRAVE.forward including its RNG draws."""
    mb = model.pqmf(x)
    z = model.encoder(mb)
    mean, scale = torch.chunk(z, 2, 1)
    std = torch.nn.functional.softplus(scale) + 1e-4
    z = mean + torch.randn_like(mean) * std          # draw 1 (encode)
    z = z - model.latent_mean.unsqueeze(-1)
    z = F.conv1d(z, model.latent_pca.unsqueeze(-1))
    z = z[:, : meta["latent_size"]]

    fill = torch.randn(1, meta["full_latent_size"] - z.shape[1],
                       z.shape[-1])                  # draw 2 (decode)
    z = torch.cat([z, fill], 1)
    z = F.conv1d(z, model.latent_pca.t().unsqueeze(-1))
    z = z + model.latent_mean.unsqueeze(-1)
    y = model.decoder(z)
    return model.pqmf.inverse(y)


def main():
    n_blocks = 8
    block = 8 * RATIO
    g = torch.Generator().manual_seed(1234)
    audio = (torch.randn(1, 1, n_blocks * block, generator=g) * 0.3).clamp(-1, 1)

    ts = torch.jit.load(str(TS_PATH), map_location="cpu")
    zero_caches(ts)
    torch.manual_seed(42)
    ref = torch.cat(
        [ts(audio[..., i * block:(i + 1) * block]) for i in range(n_blocks)], -1)

    model, meta = build_eager_model()
    zero_caches(model)
    torch.manual_seed(42)
    out = torch.cat(
        [eager_forward_stochastic(model, meta,
                                  audio[..., i * block:(i + 1) * block])
         for i in range(n_blocks)], -1)

    diff = (ref - out).abs()
    print(f"ref rms      : {ref.pow(2).mean().sqrt():.6f}")
    print(f"max abs diff : {diff.max():.3e}")
    print(f"mean abs diff: {diff.mean():.3e}")
    assert diff.max() < 1e-5, "REBUILD MISMATCH"
    print("OK: eager rebuild matches TorchScript reference (RNG-synced)")


if __name__ == "__main__":
    main()
