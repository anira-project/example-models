"""Export the explicit-state RAVE model to ONNX (encoder / decoder / forward).

All models are stateless: every call takes `state_in` and returns `state_out`.
Feed zeros as the initial state; pass the returned state into the next call.
The decoder additionally takes `fill_in` — the prior sample for the
PCA-truncated latent dimensions (zeros = deterministic mean-of-prior;
N(0, 1) noise reproduces the stock model's stochastic decode).

Usage:  python export_onnx.py [--block 2048] [--outdir onnx]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "rave"))

from stateful_rave import (RATIO, build_eager_model,
                           StatefulEncoder, StatefulDecoder, StatefulForward)

torch.set_grad_enabled(False)


def export(module, args, input_names, output_names, path):
    torch.onnx.export(
        module,
        args,
        str(path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=RATIO,
                    help="audio samples per call (multiple of 128)")
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parent / "models")
    a = ap.parse_args()
    assert a.block % RATIO == 0, "block size must be a multiple of 128"
    a.outdir.mkdir(parents=True, exist_ok=True)

    model, meta = build_eager_model()
    fwd = StatefulForward(model, meta, a.block)
    enc, dec = fwd.enc, fwd.dec
    n_frames = a.block // RATIO

    audio = torch.zeros(1, 1, a.block)
    latent = torch.zeros(1, meta["latent_size"], n_frames)
    fill = torch.zeros(*fwd.fill_shape)

    export(enc, (audio, torch.zeros(1, enc.registry.size)),
           ["audio_in", "state_in"], ["latent_out", "state_out"],
           a.outdir / "rave_encoder.onnx")
    export(dec, (latent, torch.zeros(1, dec.registry.size), fill),
           ["latent_in", "state_in", "fill_in"], ["audio_out", "state_out"],
           a.outdir / "rave_decoder.onnx")
    export(fwd, (audio, torch.zeros(1, fwd.registry.size), fill),
           ["audio_in", "state_in", "fill_in"], ["audio_out", "state_out"],
           a.outdir / "rave_forward.onnx")

    info = {
        "sample_rate": meta["sr"],
        "block_size": a.block,
        "latent_frames_per_block": n_frames,
        "latent_size": meta["latent_size"],
        "fill_shape": list(fwd.fill_shape),
        "fill_note": "prior sample for the truncated latent dims; "
                     "zeros = deterministic, N(0,1) = stock behavior",
        "state_sizes": {
            "encoder": enc.registry.size,
            "decoder": dec.registry.size,
            "forward": fwd.registry.size,
        },
        "state_layout": {
            "encoder": enc.registry.spec(),
            "decoder": dec.registry.spec(),
            "forward": fwd.registry.spec(),
        },
    }
    (a.outdir / "rave_meta.json").write_text(json.dumps(info, indent=2))
    print(f"wrote {a.outdir / 'rave_meta.json'}")


if __name__ == "__main__":
    main()
