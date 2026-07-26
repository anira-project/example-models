"""Export the explicit-state RAVE wrappers to ExecuTorch (.pte, XNNPACK).

Run inside venv-et (ExecuTorch's pinned torch). Same interface as the ONNX
models: state (and the decoder latent fill) are explicit inputs/outputs.

Usage: python export_executorch.py [--block 2048] [--outdir executorch]
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "rave"))

from stateful_rave import RATIO, build_eager_model, StatefulForward

torch.set_grad_enabled(False)


def lower(module, example_inputs, path):
    ep = torch.export.export(module, example_inputs, strict=False)
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner)
    prog = to_edge_transform_and_lower(
        ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    path.write_bytes(prog.buffer)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=8 * RATIO,
                    help="audio samples per call (multiple of 128; default 1024 = 8 latent frames, amortizes per-call overhead)")
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parent / "models")
    a = ap.parse_args()
    assert a.block % RATIO == 0
    a.outdir.mkdir(parents=True, exist_ok=True)

    model, meta = build_eager_model()
    fwd = StatefulForward(model, meta, a.block)
    enc, dec = fwd.enc, fwd.dec
    n_frames = a.block // RATIO

    audio = torch.zeros(1, 1, a.block)
    latent = torch.zeros(1, meta["latent_size"], n_frames)
    fill = torch.zeros(*fwd.fill_shape)

    lower(enc, (audio, torch.zeros(1, enc.registry.size)),
          a.outdir / "rave_encoder.pte")
    lower(dec, (latent, torch.zeros(1, dec.registry.size), fill),
          a.outdir / "rave_decoder.pte")
    lower(fwd, (audio, torch.zeros(1, fwd.registry.size), fill),
          a.outdir / "rave_forward.pte")


if __name__ == "__main__":
    main()
