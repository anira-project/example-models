"""LiteRT/TFLite parity harness: golden PyTorch reference vs .tflite.

Runs block-by-block with explicit state feedback against the golden vectors
from make_golden.py (original TorchScript model). Run inside venv-tflite.

Input/output tensors are matched by shape (audio / state / noise / latent all
have distinct shapes), so converter-assigned names don't matter.
"""
import json
import time
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter

HERE = Path(__file__).resolve().parent
META = json.loads((HERE / "models" / "rave_meta.json").read_text())
G = np.load(HERE / "golden" / "golden.npz")
BLOCK = int(G["block_size"])
N_BLOCKS = int(G["n_blocks"])
N_FRAMES = int(G["n_frames"])
TOL = 1e-5


class Model:
    """Shape-matched wrapper around a tflite signature runner."""

    def __init__(self, name):
        self.interp = Interpreter(model_path=str(HERE / "models" / name))
        self.runner = self.interp.get_signature_runner()
        self.in_names = {tuple(d["shape"]): n
                         for n, d in self.runner.get_input_details().items()}
        self.out_names = {tuple(d["shape"]): n
                          for n, d in self.runner.get_output_details().items()}

    def __call__(self, tensors):
        """tensors: list of np arrays; returns dict shape->array."""
        feed = {self.in_names[t.shape]: t for t in tensors}
        out = self.runner(**feed)
        return {tuple(v.shape): v for v in out.values()}


def report(name, ref, out, tol=TOL):
    ref, out = np.asarray(ref), np.asarray(out)
    diff = np.abs(ref - out).max()
    scaled = diff / (1.0 + np.abs(ref).max())
    status = "OK " if scaled < tol else "FAIL"
    print(f"[{status}] {name:<46} max diff {diff:.3e} "
          f"(scaled {scaled:.3e}, tol {tol:g})")
    assert scaled < tol, name


def audio_block(arr, i):
    return np.ascontiguousarray(arr[..., i * BLOCK:(i + 1) * BLOCK],
                                dtype=np.float32)


def run_forward(m, audio, fills, state_size):
    state = np.zeros((1, state_size), np.float32)
    audio_shape = (1, 1, BLOCK)
    out = []
    for i in range(N_BLOCKS):
        r = m([audio_block(audio, i), state, fills[i]])
        out.append(r[audio_shape])
        state = r[(1, state_size)]
    return np.concatenate(out, -1), state


def main():
    sizes = META["state_sizes"]
    m_fwd = Model("rave_forward.tflite")
    m_enc = Model("rave_encoder.tflite")
    m_dec = Model("rave_decoder.tflite")

    # forward parity
    out, _ = run_forward(m_fwd, G["audio"], G["fill_fwd"], sizes["forward"])
    report("forward: litert vs TorchScript", G["forward_ref"], out)

    # encoder parity
    state = np.zeros((1, sizes["encoder"]), np.float32)
    z_out = []
    for i in range(N_BLOCKS):
        r = m_enc([audio_block(G["audio"], i), state])
        z_out.append(r[(1, META["latent_size"], N_FRAMES)])
        state = r[(1, sizes["encoder"])]
    z_out = np.concatenate(z_out, -1)
    report("encoder: litert vs TorchScript", G["latent_ref"], z_out)

    # decoder parity
    state = np.zeros((1, sizes["decoder"]), np.float32)
    y_out = []
    for i in range(N_BLOCKS):
        z = np.ascontiguousarray(
            G["latent_ref"][..., i * N_FRAMES:(i + 1) * N_FRAMES])
        r = m_dec([z, state, G["fill_dec"][i]])
        y_out.append(r[(1, 1, BLOCK)])
        state = r[(1, sizes["decoder"])]
    report("decoder: litert vs TorchScript", G["decoder_ref"],
           np.concatenate(y_out, -1))

    # state isolation: two different streams interleaved through ONE interpreter
    sa = np.zeros((1, sizes["forward"]), np.float32)
    sb = np.zeros((1, sizes["forward"]), np.float32)
    out_a, out_b = [], []
    for i in range(N_BLOCKS):
        ra = m_fwd([audio_block(G["audio"], i), sa, G["fill_fwd"][i]])
        rb = m_fwd([audio_block(G["audio_b"], i), sb, G["fill_fwd_b"][i]])
        out_a.append(ra[(1, 1, BLOCK)])
        sa = ra[(1, sizes["forward"])]
        out_b.append(rb[(1, 1, BLOCK)])
        sb = rb[(1, sizes["forward"])]
    report("interleaved stream A (one interpreter)", G["forward_ref"],
           np.concatenate(out_a, -1))
    report("interleaved stream B (one interpreter)", G["forward_ref_b"],
           np.concatenate(out_b, -1))

    # benchmark
    args = [audio_block(G["audio"], 0),
            np.zeros((1, sizes["forward"]), np.float32), G["fill_fwd"][0]]
    for _ in range(5):
        m_fwd(args)
    t0 = time.perf_counter()
    n_iter = 50
    for _ in range(n_iter):
        m_fwd(args)
    dt = (time.perf_counter() - t0) / n_iter * 1000
    budget = BLOCK / META["sample_rate"] * 1000
    print(f"\nbenchmark: {dt:.2f} ms per {BLOCK}-sample block "
          f"(realtime budget {budget:.2f} ms, {budget / dt:.1f}x realtime)")
    print("ALL OK")


if __name__ == "__main__":
    main()
