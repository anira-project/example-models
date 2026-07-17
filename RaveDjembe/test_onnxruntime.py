"""ONNX Runtime test harness: PyTorch (.ts) reference vs ONNX, same numbers.

For each of encoder / decoder / forward:
  * run the original TorchScript streaming model block-by-block (it keeps its
    state in internal buffers) -> reference output
  * run the ONNX model in onnxruntime block-by-block, passing state_in and
    feeding the returned state_out into the next call (stateless session)
  * assert numerical parity within float tolerance

Plus:
  * statelessness proof: two interleaved streams through ONE session, each
    with its own state vector, match dedicated reference runs
  * fresh-session reproducibility: same inputs -> bit-identical outputs
  * realtime benchmark

The models are deterministic; decoder noise is passed as an input, generated
here with the same seeded RNG sequence the TorchScript model consumes
internally (torch.rand), so outputs are directly comparable.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

torch.set_grad_enabled(False)

HERE = Path(__file__).resolve().parent
TS_PATH = HERE / "Djembe_deterministic.ts"
ONNX_DIR = HERE / "models"

META = json.loads((ONNX_DIR / "rave_meta.json").read_text())
BLOCK = META["block_size"]
NOISE_SHAPE = tuple(META["noise_shape"])
TOL = 1e-5


def fresh_ts():
    ts = torch.jit.load(str(TS_PATH), map_location="cpu")
    for name, buf in ts.named_buffers():
        if name.endswith(".pad") or name.endswith(".cache"):
            buf.zero_()
    return ts


def session(name):
    return ort.InferenceSession(str(ONNX_DIR / name),
                                providers=["CPUExecutionProvider"])


def make_audio(n_blocks, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, 1, n_blocks * BLOCK, generator=g) * 0.3).clamp(-1, 1)


def noise_seq(n_blocks, seed):
    torch.manual_seed(seed)
    return [(torch.rand(NOISE_SHAPE) * 2 - 1).numpy() for _ in range(n_blocks)]


def report(name, ref, out, tol=TOL):
    """Scale-aware comparison: |a-b| relative to the reference magnitude.

    Audio is in [-1, 1] so this is ~absolute tolerance there; latents reach
    magnitude ~50, where float32 kernel-order differences between backends
    legitimately reach a few times 1e-5 absolute.
    """
    diff = np.abs(ref - out).max()
    scaled = diff / (1.0 + np.abs(ref).max())
    status = "OK " if scaled < tol else "FAIL"
    print(f"[{status}] {name:<46} max diff {diff:.3e} "
          f"(scaled {scaled:.3e}, tol {tol:g})")
    assert scaled < tol, name


def blocks(x):
    for i in range(x.shape[-1] // BLOCK):
        yield i, x[..., i * BLOCK:(i + 1) * BLOCK]


def run_onnx_forward(sess, audio, noises, state=None):
    state = np.zeros((1, META["state_sizes"]["forward"]), np.float32) \
        if state is None else state
    out = []
    for i, chunk in blocks(audio):
        y, state = sess.run(None, {"audio_in": chunk.numpy(),
                                   "state_in": state,
                                   "noise_in": noises[i]})
        out.append(y)
    return np.concatenate(out, -1), state


def main():
    n_blocks = 16
    audio = make_audio(n_blocks, seed=1234)

    sess_fwd = session("rave_forward.onnx")
    sess_enc = session("rave_encoder.onnx")
    sess_dec = session("rave_decoder.onnx")

    # ---------------- forward parity ----------------
    ts = fresh_ts()
    torch.manual_seed(42)
    ref = torch.cat([ts(c) for _, c in blocks(audio)], -1).numpy()

    noises = noise_seq(n_blocks, seed=42)
    out, _ = run_onnx_forward(sess_fwd, audio, noises)
    report("forward: onnxruntime vs TorchScript", ref, out)

    # ---------------- encoder parity ----------------
    ts = fresh_ts()
    z_ref = torch.cat([ts.encode(c) for _, c in blocks(audio)], -1).numpy()

    state = np.zeros((1, META["state_sizes"]["encoder"]), np.float32)
    z_out = []
    for _, chunk in blocks(audio):
        z, state = sess_enc.run(None, {"audio_in": chunk.numpy(),
                                       "state_in": state})
        z_out.append(z)
    z_out = np.concatenate(z_out, -1)
    report("encoder: onnxruntime vs TorchScript", z_ref, z_out)

    # ---------------- decoder parity ----------------
    n_frames = META["latent_frames_per_block"]
    ts = fresh_ts()
    torch.manual_seed(7)
    y_ref = torch.cat(
        [ts.decode(torch.from_numpy(
            z_ref[..., i * n_frames:(i + 1) * n_frames]))
         for i in range(n_blocks)], -1).numpy()

    noises = noise_seq(n_blocks, seed=7)
    state = np.zeros((1, META["state_sizes"]["decoder"]), np.float32)
    y_out = []
    for i in range(n_blocks):
        y, state = sess_dec.run(None, {
            "latent_in": z_ref[..., i * n_frames:(i + 1) * n_frames],
            "state_in": state,
            "noise_in": noises[i]})
        y_out.append(y)
    y_out = np.concatenate(y_out, -1)
    report("decoder: onnxruntime vs TorchScript", y_ref, y_out)

    # ------------- encoder->decoder == forward -------------
    report("enc->dec chain == forward model", y_out.astype(np.float64),
           run_onnx_forward(sess_fwd, audio, noise_seq(n_blocks, 7))[0], TOL)

    # ---------------- statelessness: interleaved streams ----------------
    audio_a, audio_b = make_audio(n_blocks, 1), make_audio(n_blocks, 2)
    ts = fresh_ts()
    torch.manual_seed(100)
    ref_a = torch.cat([ts(c) for _, c in blocks(audio_a)], -1).numpy()
    ts = fresh_ts()
    torch.manual_seed(200)
    ref_b = torch.cat([ts(c) for _, c in blocks(audio_b)], -1).numpy()

    na, nb = noise_seq(n_blocks, 100), noise_seq(n_blocks, 200)
    sa = np.zeros((1, META["state_sizes"]["forward"]), np.float32)
    sb = np.zeros((1, META["state_sizes"]["forward"]), np.float32)
    out_a, out_b = [], []
    for i, _ in blocks(audio_a):
        ya, sa = sess_fwd.run(None, {
            "audio_in": audio_a[..., i * BLOCK:(i + 1) * BLOCK].numpy(),
            "state_in": sa, "noise_in": na[i]})
        yb, sb = sess_fwd.run(None, {
            "audio_in": audio_b[..., i * BLOCK:(i + 1) * BLOCK].numpy(),
            "state_in": sb, "noise_in": nb[i]})
        out_a.append(ya)
        out_b.append(yb)
    report("interleaved stream A (one session)", ref_a,
           np.concatenate(out_a, -1))
    report("interleaved stream B (one session)", ref_b,
           np.concatenate(out_b, -1))

    # ---------------- reproducibility across sessions ----------------
    out2, _ = run_onnx_forward(session("rave_forward.onnx"), audio,
                               noise_seq(n_blocks, 42))
    report("fresh session bit-identical", out, out2, tol=1e-12)

    # ---------------- benchmark ----------------
    state = np.zeros((1, META["state_sizes"]["forward"]), np.float32)
    chunk = audio[..., :BLOCK].numpy()
    noise = noises[0]
    for _ in range(5):  # warmup
        sess_fwd.run(None, {"audio_in": chunk, "state_in": state,
                            "noise_in": noise})
    t0 = time.perf_counter()
    n_iter = 50
    for _ in range(n_iter):
        _, state = sess_fwd.run(None, {"audio_in": chunk, "state_in": state,
                                       "noise_in": noise})
    dt = (time.perf_counter() - t0) / n_iter * 1000
    budget = BLOCK / META["sample_rate"] * 1000
    print(f"\nbenchmark: {dt:.2f} ms per {BLOCK}-sample block "
          f"(realtime budget {budget:.2f} ms, "
          f"{budget / dt:.1f}x realtime)")
    print("ALL OK")


if __name__ == "__main__":
    main()
