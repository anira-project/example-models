# RaveDjembe — RAVE with explicit state (stateless streaming)

Port of `Djembe_deterministic.ts` (RAVE v2.3.1 TorchScript streaming export,
v1 architecture, variational encoder) to **ONNX Runtime, LiteRT/TFLite and
ExecuTorch**, with all streaming state passed in/out as an explicit tensor,
so stateless runtimes can run block-by-block inference. All three backends
are parity-verified against the original TorchScript model.

## Model facts

| | |
|---|---|
| architecture | RAVE v1 (PQMF 16 bands, Encoder + Generator with loudness/noise branches) |
| sample rate | 44100, mono |
| compression ratio | 2048 samples per latent frame |
| latent size | 4 (PCA-truncated from 16; zero-filled on decode) |
| determinism | latent path deterministic; decoder noise is an explicit input |

Hyperparameters recovered from the weights: `capacity=32`, `ratios=[4,4,4,2]`,
`n_band=16`, `full_latent=16`, noise generator `ratios=[4,4,4], bands=5`,
`ResidualStack dilations [[1,1],[3,1],[5,1]]`, all convs bias-free (v1.gin).

## Interface (identical across `.onnx` / `.tflite` / `.pte`)

All tensors float32, batch 1, fixed block size 2048 samples:

```
rave_encoder : audio_in [1,1,2048],  state_in [1,5888]                        -> latent_out [1,4,1], state_out
rave_decoder : latent_in [1,4,1],    state_in [1,24384], noise_in [1,2,16,64] -> audio_out [1,1,2048], state_out
rave_forward : audio_in [1,1,2048],  state_in [1,30272], noise_in [1,2,16,64] -> audio_out [1,1,2048], state_out
```

Streaming protocol:

1. start every stream with `state_in = zeros`
2. each call: pass the previous call's `state_out` as `state_in`
3. `noise_in`: uniform random in [-1, 1] per call (any RNG; it adds the
   model's noise texture, ~-34 dB below the signal). Zeros disable the noise
   branch — output is then fully deterministic from audio alone. The tensor
   holds exactly one value per output sample.

`models/rave_meta.json` documents the exact state layout (ordered
`{name, channels, length}` chunks) per model. Multiple independent streams
can share one session/interpreter — state is fully external.

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
| forward, 16 chained blocks | 8.7e-08 | 9.5e-08 | 1.3e-07 |
| encoder (latents \|z\|≈50, rel err) | 1.5e-06 | 1.4e-06 | 1.5e-06 |
| decoder | 2.9e-08 | 9.4e-08 | 8.1e-08 |
| 2 interleaved streams, 1 session | ≤1.4e-07 | ≤9.5e-08 | ≤2.1e-07 |
| ms / 2048-sample block (budget 46.4) | 0.82 (56×) | 1.14 (41×) | 4.06 (11×) |

## Regenerating / testing

Each toolchain pins its own torch, so three venvs. Regeneration additionally
needs the RAVE source cloned into this directory:

```bash
cd RaveDjembe
git clone https://github.com/acids-ircam/RAVE rave   # v2.3.1-era blocks.py/pqmf.py

# ONNX + golden vectors — any recent python
python3 -m venv venv
./venv/bin/pip install torch onnx onnxruntime "cached-conv>=2.5.0" \
    gin-config einops numpy scipy
./venv/bin/python test_rebuild.py          # eager rebuild == .ts (bit-exact)
./venv/bin/python test_stateful_eager.py   # explicit-state wrappers == .ts
./venv/bin/python export_onnx.py           # writes models/*.onnx + rave_meta.json
./venv/bin/python test_onnxruntime.py
./venv/bin/python make_golden.py           # golden/golden.npz for the other venvs

# ExecuTorch — python 3.10-3.12; torch pinned to match the executorch wheel ABI
python3.10 -m venv venv-et
./venv-et/bin/pip install executorch "torch==2.12.*" "cached-conv>=2.5.0" \
    gin-config einops scipy
./venv-et/bin/python export_executorch.py  # writes models/*.pte (XNNPACK)
./venv-et/bin/python test_executorch.py

# LiteRT — python 3.10-3.12 (litert-torch, formerly ai-edge-torch)
python3.10 -m venv venv-tflite
./venv-tflite/bin/pip install litert-torch "cached-conv>=2.5.0" gin-config \
    einops scipy
./venv-tflite/bin/python export_tflite.py  # writes models/*.tflite
./venv-tflite/bin/python test_tflite.py
```

Running only the parity tests (not regenerating) needs neither the RAVE
clone nor `make_golden.py` — the committed `golden/golden.npz` and models
suffice for `test_onnxruntime.py` / `test_executorch.py` / `test_tflite.py`.
