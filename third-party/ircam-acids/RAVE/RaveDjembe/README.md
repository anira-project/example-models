# RaveDjembe — RAVE with explicit state (stateless streaming)

The RAVE model itself was developed, modified and trained by
[Michael Witte](https://github.com/michaelwitte); the original architecture
and training code come from the IRCAM
[acids-ircam/RAVE](https://github.com/acids-ircam/RAVE) repository.

Port of `Djembe_streaming.ts` (RAVE v2.3.1 TorchScript streaming export,
v1 architecture, variational encoder, causal low-latency configuration;
checkpoint drop `djembe_2_80d99cfa85`) to **ONNX Runtime, LiteRT/TFLite and
ExecuTorch**, with all streaming state passed in/out as an explicit tensor,
so stateless runtimes can run block-by-block inference. All backends are
parity-verified against the original TorchScript model (see the proof chain
below).

## Model facts

| | |
|---|---|
| architecture | RAVE v1 (PQMF 16 bands, Encoder + Generator), causal convolutions |
| sample rate | 44100, mono |
| compression ratio | 128 samples per latent frame (16x lower latency than the previous Djembe) |
| export block size | 1024 samples = 8 latent frames per call (amortizes per-call overhead ~3-4x; re-export with `--block` for any multiple of 128) |
| latent size | 2 (PCA-truncated from 16) |
| determinism | encoder emits the variational mean; the decoder's truncated-dims prior sample is the explicit `fill_in` input (zeros = deterministic) |

Hyperparameters recovered from the weights and the scripted module tree:
`capacity=64`, `ratios=[2,2,2,1]`, `n_band=16` (129-tap PQMF prototype),
`full_latent=16`, no noise generator (`use_noise=False`),
`ResidualStack dilations [[3,1],[9,1],[27,1],[36,1]]`, all convs bias-free,
causal padding (`cached_conv.get_padding.mode = "causal"` — every branch
alignment delay is zero, which is where much of the latency drop comes from).

## Interface (identical across `.onnx` / `.tflite` / `.pte`)

All tensors float32, batch 1, fixed block size 1024 samples (8 latent
frames per call; frame k of a block covers samples [k*128, (k+1)*128)):

```
rave_encoder : audio_in [1,1,1024], state_in [1,7136]                     -> latent_out [1,2,8], state_out
rave_decoder : latent_in [1,2,8],   state_in [1,154464], fill_in [1,14,8] -> audio_out [1,1,1024], state_out
rave_forward : audio_in [1,1,1024], state_in [1,161600], fill_in [1,14,8] -> audio_out [1,1,1024], state_out
```

Streaming protocol:

1. start every stream with `state_in = zeros`
2. each call: pass the previous call's `state_out` as `state_in`
3. `fill_in`: the prior sample for the 14 PCA-truncated latent dimensions.
   Zeros give the deterministic mean-of-prior decode; N(0, 1) noise per call
   reproduces the stock model's stochastic decode texture.

`models/rave_meta.json` documents the exact state layout (ordered
`{name, channels, length}` chunks) per model. Multiple independent streams
can share one session/interpreter — state is fully external.

## Proof chain

The stock export is stochastic (variational sampling in `encode`, prior
noise in `decode`), so parity is proven in two exact hops:

1. `test_rebuild.py` — the eager rebuild equals the TorchScript export
   **bit-exactly** with synced RNG draws.
2. `test_stateful_eager.py` — the explicit-state wrappers equal the eager
   model **bit-exactly** (mean-encode, explicit fill).

`make_golden.py` then generates golden vectors from the wrappers, and
`test_onnxruntime.py` / `test_tflite.py` / `test_executorch.py` verify each
runtime against them (block-by-block, state feedback, interleaved-stream
statelessness, reproducibility, realtime benchmark).

## How the state was made explicit

The streaming TorchScript model keeps memory in two `cached_conv` primitives:

* `CachedPadding1d` — causal-conv left context and delay lines
* `CachedConvTranspose1d` — overlap-add tails of the decoder upsamplers

`stateful_rave.py` rebuilds the model in eager PyTorch from the RAVE repo
source, loads the `.ts` weights (bit-exact), and re-implements those
primitives to read/write slices of one flat state vector. Portability
rewrites: `torch.rand_like` → external `noise_in`; `rfft/irfft` → exact DFT
matmuls; PQMF sign flips → constant masks. Resulting op set: Conv,
ConvTranspose, MatMul, elementwise, Slice/Concat/Reshape — no FFT, no
random, no dynamic shapes.

## Results (macOS arm64, CPU; scaled = diff / (1 + max|ref|))

| check (vs TorchScript reference) | ONNX Runtime | LiteRT | ExecuTorch |
|---|---|---|---|
| forward, 8 chained 1024-blocks | 8.9e-06 | 1.0e-05 | 5.9e-06 |
| encoder (latents \|z\|≈50) | 4.2e-06 | 4.2e-06 | 5.9e-06 |
| decoder | 3.5e-06 | 3.1e-06 | 3.1e-06 |
| 2 interleaved streams, 1 session | ≤8.9e-06 | ≤1.0e-05 | ≤6.3e-06 |
| ms / 1024-sample block, 1 thread (budget 23.2) | 6.1 (3.8×) | 8.2 (2.8×) | 8.3 (2.8×) |

## Files and licensing

`Djembe_streaming.ts` is the source TorchScript export this port is built
from; `../rave_funk_drum.ts` is a second custom-trained RAVE model kept
alongside for experimentation (not yet ported — the same pipeline applies if
its architecture matches). The RAVE-trained models and the exports derived
from them are licensed CC BY-NC 4.0, see [../LICENSE](../LICENSE). The
export/test code in this directory is covered by the repository license.

## Regenerating / testing

The environment is the repository's uv project (`pyproject.toml` at the
repository root; `uv run` finds it from this directory). Its base
dependencies already pin torch to the release the ExecuTorch wheels are built
against, so one environment with the `rave` dependency group covers the ONNX
and ExecuTorch exports, the golden vectors and the parity tests of all three
runtimes. Only the LiteRT export needs `litert-torch` (formerly
ai-edge-torch), in the separate `rave-tflite-export` group: it depends on
`litert-converter`, which ships wheels for Linux x86_64 and macOS arm64 only.
Regeneration additionally needs the RAVE source cloned into this directory:

```bash
cd third-party/ircam-acids/RAVE/RaveDjembe
git clone https://github.com/acids-ircam/RAVE rave   # v2.3.1-era blocks.py/pqmf.py

uv run --group rave python test_rebuild.py          # eager rebuild == .ts (bit-exact)
uv run --group rave python test_stateful_eager.py   # explicit-state wrappers == .ts
uv run --group rave python export_onnx.py           # writes models/*.onnx + rave_meta.json
uv run --group rave python test_onnxruntime.py
uv run --group rave python make_golden.py           # golden/golden.npz
uv run --group rave python export_executorch.py     # writes models/*.pte (XNNPACK)
uv run --group rave python test_executorch.py

# LiteRT export: Linux x86_64 or macOS arm64 only
uv run --group rave --group rave-tflite-export python export_tflite.py  # writes models/*.tflite
uv run --group rave python test_tflite.py
```

Running only the parity tests (not regenerating) needs neither the RAVE
clone nor `make_golden.py` — the committed `golden/golden.npz` and models
suffice for `test_onnxruntime.py` / `test_executorch.py` / `test_tflite.py`.
