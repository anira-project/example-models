"""Export the explicit-state RAVE wrappers to LiteRT (.tflite).

Run inside venv-tflite (litert-torch + its pinned torch). Same interface as
the ONNX/ExecuTorch exports: state (and decoder noise) are explicit
inputs/outputs; feed zeros initially and pass state_out back in.

Usage: python export_tflite.py [--block 2048] [--outdir tflite]
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "rave"))

from stateful_rave import RATIO, build_eager_model, StatefulForward

torch.set_grad_enabled(False)


def convert(module, example_inputs, path):
    import litert_torch
    m = litert_torch.convert(module.eval(), example_inputs)
    m.export(str(path))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=RATIO)
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
    noise = torch.zeros(*fwd.noise_shape)

    convert(enc, (audio, torch.zeros(1, enc.registry.size)),
            a.outdir / "rave_encoder.tflite")
    convert(dec, (latent, torch.zeros(1, dec.registry.size), noise),
            a.outdir / "rave_decoder.tflite")
    convert(fwd, (audio, torch.zeros(1, fwd.registry.size), noise),
            a.outdir / "rave_forward.tflite")


if __name__ == "__main__":
    main()
