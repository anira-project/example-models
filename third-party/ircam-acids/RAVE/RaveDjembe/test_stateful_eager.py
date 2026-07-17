"""Step 2 check: explicit-state wrappers (eager) == .ts model.

The wrappers carry ALL memory in an explicit flat state tensor. We verify:
  1. forward parity vs the TorchScript reference (RNG-synced noise)
  2. encode parity
  3. decode parity
  4. statelessness: two interleaved streams through the SAME module instance,
     each with its own state vector, match two dedicated reference runs.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "rave"))

from stateful_rave import (TS_PATH, RATIO, build_eager_model,
                           StatefulEncoder, StatefulDecoder, StatefulForward)

torch.set_grad_enabled(False)
BLOCK = RATIO  # 2048


def fresh_ts():
    ts = torch.jit.load(str(TS_PATH), map_location="cpu")
    for name, buf in ts.named_buffers():
        if name.endswith(".pad") or name.endswith(".cache"):
            buf.zero_()
    return ts


def make_audio(n_blocks, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, 1, n_blocks * BLOCK, generator=g) * 0.3).clamp(-1, 1)


def run_ts_forward(ts, audio, seed):
    torch.manual_seed(seed)
    out = []
    for i in range(audio.shape[-1] // BLOCK):
        out.append(ts(audio[..., i * BLOCK:(i + 1) * BLOCK]))
    return torch.cat(out, -1)


def run_wrapper_forward(fwd, audio, seed, state=None):
    torch.manual_seed(seed)
    state = torch.zeros(1, fwd.registry.size) if state is None else state
    out = []
    for i in range(audio.shape[-1] // BLOCK):
        noise = torch.rand(fwd.noise_shape) * 2 - 1
        y, state = fwd(audio[..., i * BLOCK:(i + 1) * BLOCK], state, noise)
        out.append(y)
    return torch.cat(out, -1), state


def report(name, a, b, tol):
    diff = (a - b).abs().max().item()
    status = "OK " if diff < tol else "FAIL"
    print(f"[{status}] {name:<42} max diff {diff:.3e} (tol {tol:g})")
    assert diff < tol, name


def main():
    n_blocks = 12
    audio = make_audio(n_blocks, seed=1234)

    model, meta = build_eager_model()
    fwd = StatefulForward(model, meta, BLOCK)
    enc = fwd.enc
    dec = fwd.dec
    print(f"state sizes: forward={fwd.registry.size}, "
          f"encoder={enc.registry.size}, decoder={dec.registry.size}")

    # 1. forward parity
    ref = run_ts_forward(fresh_ts(), audio, seed=42)
    out, _ = run_wrapper_forward(fwd, audio, seed=42)
    report("forward vs TorchScript", ref, out, 1e-5)

    # 2. encode parity
    ts = fresh_ts()
    z_ref = torch.cat(
        [ts.encode(audio[..., i * BLOCK:(i + 1) * BLOCK])
         for i in range(n_blocks)], -1)
    state = torch.zeros(1, enc.registry.size)
    z_out = []
    for i in range(n_blocks):
        z, state = enc(audio[..., i * BLOCK:(i + 1) * BLOCK], state)
        z_out.append(z)
    z_out = torch.cat(z_out, -1)
    report("encode vs TorchScript", z_ref, z_out, 1e-5)

    # 3. decode parity (feed the reference latents)
    ts = fresh_ts()
    torch.manual_seed(7)
    y_ref = torch.cat(
        [ts.decode(z_ref[..., i:i + 1]) for i in range(n_blocks)], -1)
    torch.manual_seed(7)
    state = torch.zeros(1, dec.registry.size)
    y_out = []
    for i in range(n_blocks):
        noise = torch.rand(dec.noise_shape) * 2 - 1
        y, state = dec(z_ref[..., i:i + 1], state, noise)
        y_out.append(y)
    y_out = torch.cat(y_out, -1)
    report("decode vs TorchScript", y_ref, y_out, 1e-5)

    # 4. statelessness: interleave two independent streams through ONE module
    audio_a = make_audio(n_blocks, seed=1)
    audio_b = make_audio(n_blocks, seed=2)
    ref_a = run_ts_forward(fresh_ts(), audio_a, seed=100)
    ref_b = run_ts_forward(fresh_ts(), audio_b, seed=200)

    # pre-generate noise sequences with the same seeds
    torch.manual_seed(100)
    noise_a = [torch.rand(fwd.noise_shape) * 2 - 1 for _ in range(n_blocks)]
    torch.manual_seed(200)
    noise_b = [torch.rand(fwd.noise_shape) * 2 - 1 for _ in range(n_blocks)]

    sa = torch.zeros(1, fwd.registry.size)
    sb = torch.zeros(1, fwd.registry.size)
    out_a, out_b = [], []
    for i in range(n_blocks):
        ya, sa = fwd(audio_a[..., i * BLOCK:(i + 1) * BLOCK], sa, noise_a[i])
        yb, sb = fwd(audio_b[..., i * BLOCK:(i + 1) * BLOCK], sb, noise_b[i])
        out_a.append(ya)
        out_b.append(yb)
    report("interleaved stream A", ref_a, torch.cat(out_a, -1), 1e-5)
    report("interleaved stream B", ref_b, torch.cat(out_b, -1), 1e-5)

    print("ALL OK")


if __name__ == "__main__":
    main()
