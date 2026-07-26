"""Step 2 check: explicit-state wrappers (eager) == .ts model.

The wrappers carry ALL memory in an explicit flat state tensor and are
deterministic: the encoder emits the variational mean, the decoder takes the
truncated-dims prior sample as an explicit `fill` input. Parity vs the stock
(stochastic) TorchScript model is exact when the RNG draws are replicated:
eps for encode is forced to zero by comparing against a mean-path eager
reference proven exact in test_rebuild, and the fills are seeded and passed
explicitly. We verify:
  1. forward parity vs the RNG-synced eager reference
  2. encode parity (mean path)
  3. decode parity (given fills)
  4. statelessness: two interleaved streams through the SAME module instance
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "rave"))

from stateful_rave import (RATIO, build_eager_model,
                           StatefulEncoder, StatefulDecoder, StatefulForward)

torch.set_grad_enabled(False)
BLOCK = 8 * RATIO


def zero_caches(m):
    for name, buf in m.named_buffers():
        if name.endswith(".pad") or name.endswith(".cache"):
            buf.zero_()


def make_audio(n_blocks, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, 1, n_blocks * BLOCK, generator=g) * 0.3).clamp(-1, 1)


def eager_reference(model, meta, audio, fills):
    """Mean-encode + explicit-fill decode, stateful via internal caches."""
    zero_caches(model)
    out = []
    n_blocks = audio.shape[-1] // BLOCK
    for i in range(n_blocks):
        x = audio[..., i * BLOCK:(i + 1) * BLOCK]
        mb = model.pqmf(x)
        z = model.encoder(mb)
        mean, _ = torch.chunk(z, 2, 1)
        z = mean - model.latent_mean.unsqueeze(-1)
        z = F.conv1d(z, model.latent_pca.unsqueeze(-1))
        z = z[:, : meta["latent_size"]]
        z = torch.cat([z, fills[i]], 1)
        z = F.conv1d(z, model.latent_pca.t().unsqueeze(-1))
        z = z + model.latent_mean.unsqueeze(-1)
        y = model.decoder(z)
        out.append(model.pqmf.inverse(y))
    return torch.cat(out, -1)


def run_wrapper_forward(fwd, audio, fills, state=None):
    state = torch.zeros(1, fwd.registry.size) if state is None else state
    out = []
    for i in range(audio.shape[-1] // BLOCK):
        y, state = fwd(audio[..., i * BLOCK:(i + 1) * BLOCK], state, fills[i])
        out.append(y)
    return torch.cat(out, -1), state


def report(name, a, b, tol):
    diff = (a - b).abs().max().item()
    status = "OK " if diff < tol else "FAIL"
    print(f"[{status}] {name:<42} max diff {diff:.3e} (tol {tol:g})")
    assert diff < tol, name


def make_fills(shape, n_blocks, seed):
    torch.manual_seed(seed)
    return [torch.randn(shape) for _ in range(n_blocks)]


def main():
    n_blocks = 12
    audio = make_audio(n_blocks, seed=1234)
    audio_b = make_audio(n_blocks, seed=2)

    # ---- eager references FIRST: patch_stateful() (triggered by wrapper
    # construction) globally replaces the cached-conv forwards, after which
    # the plain eager model can no longer run.
    model_ref, meta = build_eager_model()
    n_frames = BLOCK // RATIO
    fill_shape = (1, meta["full_latent_size"] - meta["latent_size"], n_frames)
    fills = make_fills(fill_shape, n_blocks, seed=42)
    fills_b = make_fills(fill_shape, n_blocks, seed=7)

    ref = eager_reference(model_ref, meta, audio, fills)
    ref_b = eager_reference(model_ref, meta, audio_b, fills_b)

    zero_caches(model_ref)
    z_ref = []
    for i in range(n_blocks):
        x = audio[..., i * BLOCK:(i + 1) * BLOCK]
        z = model_ref.encoder(model_ref.pqmf(x))
        mean, _ = torch.chunk(z, 2, 1)
        z = mean - model_ref.latent_mean.unsqueeze(-1)
        z = F.conv1d(z, model_ref.latent_pca.unsqueeze(-1))
        z_ref.append(z[:, : meta["latent_size"]])
    z_ref = torch.cat(z_ref, -1)

    zero_caches(model_ref)
    y_ref = []
    for i in range(n_blocks):
        z = z_ref[..., i * n_frames:(i + 1) * n_frames]
        zf = torch.cat([z, fills[i]], 1)
        zf = F.conv1d(zf, model_ref.latent_pca.t().unsqueeze(-1))
        zf = zf + model_ref.latent_mean.unsqueeze(-1)
        y_ref.append(model_ref.pqmf.inverse(model_ref.decoder(zf)))
    y_ref = torch.cat(y_ref, -1)

    # ---- wrappers (patches the primitives globally)
    model, meta2 = build_eager_model()
    fwd = StatefulForward(model, meta2, BLOCK)

    out, _ = run_wrapper_forward(fwd, audio, fills)
    report("forward: wrapper vs eager reference", ref, out, 1e-5)

    enc = fwd.enc
    state = torch.zeros(1, enc.registry.size)
    z_out = []
    for i in range(n_blocks):
        z, state = enc(audio[..., i * BLOCK:(i + 1) * BLOCK], state)
        z_out.append(z)
    report("encoder: wrapper vs eager reference", z_ref,
           torch.cat(z_out, -1), 1e-5)

    dec = fwd.dec
    state = torch.zeros(1, dec.registry.size)
    y_out = []
    for i in range(n_blocks):
        y, state = dec(z_ref[..., i * n_frames:(i + 1) * n_frames], state,
                       fills[i])
        y_out.append(y)
    report("decoder: wrapper vs eager reference", y_ref,
           torch.cat(y_out, -1), 1e-5)

    sa = torch.zeros(1, fwd.registry.size)
    sb = torch.zeros(1, fwd.registry.size)
    out_a, out_b = [], []
    for i in range(n_blocks):
        ya, sa = fwd(audio[..., i * BLOCK:(i + 1) * BLOCK], sa, fills[i])
        yb, sb = fwd(audio_b[..., i * BLOCK:(i + 1) * BLOCK], sb, fills_b[i])
        out_a.append(ya)
        out_b.append(yb)
    report("interleaved stream A", ref, torch.cat(out_a, -1), 1e-5)
    report("interleaved stream B", ref_b, torch.cat(out_b, -1), 1e-5)
    print("ALL OK")


if __name__ == "__main__":
    main()
