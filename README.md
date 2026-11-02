# example-models

This repository contains example models for the [anira](https://github.com/anira-project/anira) library, designed for validation and demonstration purposes.

## Content

- **`SimpleGainNetwork`**: This folder contains an example model, designed for multichannel audio inference. It includes:
  - **Inputs**:
    - **Audio data**: A multichannel input array representing the audio signal. Models are generated for `1` (mono), `2` (stereo), `4` and `16` channels.
    - **Gain parameter**: A parameter to apply gain (amplification) to the input audio.
  - **Outputs**:
    - **Processed audio**: The input audio after applying the gain.
    - **Peak gain**: The maximum absolute gain measured across all channels.
  - **Formats**: Each channel count is exported to TorchScript (`.pt`), ONNX (`.onnx`), TFLite (`.tflite`), and ExecuTorch (`.pte`). The build script cross-checks that all four runtimes produce consistent outputs.

- **`StatefulAccumulatorNetwork`**: This folder contains an example model with **explicit state**, designed to validate state passing: the state is an ordinary input tensor and an ordinary output tensor, and the caller feeds `state_out` back as the next `state_in`. The model itself keeps nothing between two calls, so the same model runs in every runtime (none of them needs a hidden state) and the feedback is observable from outside. It has no weights and no randomness. It includes:
  - **Inputs**, in this order:
    - **`state_in`**: float32 `[1, C, 2]`. Per channel `c`, `state_in[0, c, 0]` is the running sum and `state_in[0, c, 1]` a block counter. It starts as zeros.
    - **`data`**: float32 `[1, C, N]`, the audio block. `N` is dynamic. Models are generated for `C = 1` (mono) and `C = 2` (stereo).
  - **Outputs**, in this order:
    - **`processed_data`**: float32 `[1, C, N]`, with `processed_data[0, c, n] = data[0, c, n] + state_in[0, c, 0]`.
    - **`state_out`**: float32 `[1, C, 2]`, with `state_out[0, c, 0] = state_in[0, c, 0] + sum_n data[0, c, n]` and `state_out[0, c, 1] = state_in[0, c, 1] + 1`.
  - **State position**: The state is deliberately the *first* input and the *last* output, because state tensors may sit anywhere in a model's tensor order.
  - **Expectation**: With the state fed back, the output of block `k` is its input plus the sum of every sample of the blocks `0..k-1` of the same channel, and the counter of `state_out` reads `k + 1`. Without feedback (zero state at every block) the output equals the input. With small integer-valued inputs every value is exact in float32, so both hold bit-exactly, and a stale or zeroed state is visible in the samples.
  - **Tensor order per format**: TorchScript, ONNX and ExecuTorch keep the order above; the ONNX tensor names are exactly `state_in`, `data`, `processed_data`, `state_out`. The converted TFLite files keep it on the input side only. By position (`get_input_details()` / `get_output_details()`, which is what a consumer that binds TFLite tensors by position sees, as anira does), mono and stereo alike:
    - input 0: `state_in` (`serving_default_args_0:0`, `[1, C, 2]`), input 1: `data` (`serving_default_args_0_1:0`, `[1, C, -1]`)
    - output 0: `state_out` (`PartitionedCall:1`, `[1, C, 2]`), output 1: `processed_data` (`PartitionedCall:0`, `[1, C, -1]`)
    - The `serving_default` signature keeps the declared mapping: `args_0` = `state_in`, `args_0_1` = `data`, `output_0` = `processed_data`, `output_1` = `state_out`.
  - **Formats**: Each channel count is exported to TorchScript (`.pt`), ONNX (`.onnx`), TFLite (`.tflite`), and ExecuTorch (`.pte`). The build script prints the tensor order of the ONNX and TFLite files, then runs the saved TorchScript trace and every export with `N = 10` and `N = 64` over three consecutive blocks, with the state fed back and without, and asserts the expectation above with exact equality.

- **`TypedGainNetwork`**: This folder contains the `SimpleGainNetwork` tensor set at other dtypes than float32, designed to validate that model tensors of any dtype travel end to end through a runtime. The models compute in float32 and cast at the boundary (`processed_data = Cast(Cast(data, float32) * Cast(gain, float32), <dtype>)`), so every runtime needs only a Cast for the boundary dtype, and with small integer-valued inputs and a gain of 2 every output is exact. It includes:
  - **`typed_gain_network_<dtype>`**, `<dtype>` one of `int16`, `int8`, `float16`, mono:
    - **Inputs**: `data` `[1, 1, N]` (`N` dynamic) and `gain` `[1]` (`[1, 1, 1]` in the TFLite file), both of `<dtype>`.
    - **Outputs**: `processed_data` `[1, 1, N]` and `peak` `[1]` (the largest absolute sample), both of `<dtype>`.
    - **Formats**: TorchScript (`.pt`), ONNX (`.onnx`), TFLite (`.tflite`, no quantisation parameters on any tensor) and ExecuTorch (`.pte`). The TFLite files order their tensors as `SimpleGainNetwork`'s: input 0 is `data` (`serving_default_args_0:0`), input 1 `gain`.
  - **`typed_gain_network_<dtype>_unplanned.pte`**: the same method at a static `[1, 1, 512]` window with its inputs and outputs left out of ExecuTorch's memory plan (`MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=False)`), so a runtime binds its caller's memory to them instead of copying: `Method::set_input` aliases the input, `Method::set_output_data_ptr` takes the output buffer. The window is static because `set_output_data_ptr` takes a buffer of the output's full capacity, which a dynamic axis would make its upper bound (65536 samples).
  - **`dtype_matrix_<dtype>.onnx`**, for the ten dtypes `float32`, `float64`, `float16`, `bfloat16`, `int8`, `uint8`, `int16`, `int32`, `int64`, `bool`: one input `data` `[1, 1, N]` and one output `processed_data` of that dtype, `processed_data = Cast(Cast(data, float32) * 2, <dtype>)` (a `bool` input comes back unchanged), written with `onnx.helper` (opset 17, IR version 8).
  - **`quantized_gain_network_int8.tflite`**: a fully integer TFLite model, `y = 0.5 * x` on `[1, 1, 64]`, whose input and output are `int8` tensors with a scale and a zero point (quantised I/O, for a consumer's refusal of it).
  - **Checks**: the build script runs every file in its runtime and asserts exact equality with the expectation (the unplanned `.pte` at its 512-sample window, the quantised model by its tensor types, its parameters and one inference within a quantisation step).

- **`RaveDjembe`**: A RAVE neural audio codec (v1 architecture, 44.1 kHz mono, trained on djembe) ported from its TorchScript streaming export to **ONNX Runtime, LiteRT/TFLite and ExecuTorch** with all streaming state passed as explicit input/output tensors, so stateless runtimes can run real-time block-by-block inference. Ships separate encoder / decoder / forward models per format, parity tests against golden reference vectors, and the full export pipeline. See [`third-party/ircam-acids/RAVE/RaveDjembe/README.md`](third-party/ircam-acids/RAVE/RaveDjembe/README.md).

## Regenerating the models

The Python environment is managed with [uv](https://docs.astral.sh/uv/): `pyproject.toml` declares it, `uv.lock` pins every package and `.python-version` selects Python 3.12. Create it with:

```bash
uv sync
```

This installs a CPU build of torch, onnx, onnxruntime, onnxscript, tensorflow and executorch. torch and executorch are pinned as a pair (torch 2.14 with executorch 1.5), because the ExecuTorch wheels are built against one torch release without declaring it. `uv run` syncs the environment on its own, so the `uv sync` step is optional. The `RaveDjembe` scripts need the `rave` dependency group on top, see its README.

To (re)generate the `SimpleGainNetwork` models:

```bash
uv run python SimpleGainNetwork/simple_gain_network.py
```

By default this builds the `(1, 2, 4, 16)` channel set; edit the `channel_counts` argument to `main()` to change it.

To (re)generate the `StatefulAccumulatorNetwork` models:

```bash
uv run python StatefulAccumulatorNetwork/stateful_accumulator_network.py
```

By default this builds the `(1, 2)` channel set. The `.pte` export is skipped when `executorch` is not installed. The committed files were produced and verified with Python 3.12.14 on Linux aarch64, with torch 2.14.0 (CPU build), onnx 1.23.0, onnxruntime 1.30.0, onnxscript 0.7.2, tensorflow 2.21.0 (its bundled TFLite converter and interpreter), executorch 1.5.0 and numpy 2.5.3.

To (re)generate the `TypedGainNetwork` models:

```bash
uv run python TypedGainNetwork/typed_gain_network.py
```

The `.pte` exports are skipped when `executorch` is not installed. TensorFlow's graph optimizer, which the quantised conversion runs, fails on a machine whose CUDA libraries are present but do not load; run the script with `CUDA_VISIBLE_DEVICES=-1` there. The committed files were produced and verified with Python 3.12 on Linux x86_64, with torch 2.14.0 (CPU build), onnx 1.23.0, onnxruntime 1.30.0, onnxscript 0.7.2, tensorflow 2.21.0, executorch 1.5.0 and numpy 2.5.3, and each loads and runs exactly in anira's pinned runtimes (ONNX Runtime 1.26.0, TFLite 2.17.0, LibTorch 2.12.0, ExecuTorch 1.3.1).

