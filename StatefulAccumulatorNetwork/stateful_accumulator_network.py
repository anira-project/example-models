import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import os
from typing import Callable, List, Tuple

try:
    from executorch.exir import to_edge
    from torch.export import export as torch_export

    _HAS_EXECUTORCH = True
except ImportError:
    _HAS_EXECUTORCH = False

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# state[0, c, 0] is the running sum of channel c, state[0, c, 1] the block counter.
STATE_SIZE = 2

# Every export is verified with these block sizes, over this many consecutive blocks.
TEST_BLOCK_SIZES = (10, 64)
NUM_TEST_BLOCKS = 3

InferenceFn = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def channel_suffix(channels: int) -> str:
    """Map a channel count to a file suffix (mono/stereo for 1/2, NchN otherwise)."""
    if channels == 1:
        return "mono"
    if channels == 2:
        return "stereo"
    return f"{channels}ch"


# StatefulAccumulatorNetwork (PyTorch)
# The state is an ordinary input and an ordinary output: the model keeps nothing
# between calls, the caller feeds state_out back as the next state_in. The state
# is deliberately the FIRST input and the LAST output.
class StatefulAccumulatorNetwork(nn.Module):
    def __init__(self):
        super(StatefulAccumulatorNetwork, self).__init__()

    def forward(self, state_in, data) -> Tuple[torch.Tensor, torch.Tensor]:
        running_sum = state_in[:, :, 0:1]
        block_count = state_in[:, :, 1:2]
        processed_data = data + running_sum
        new_sum = running_sum + torch.sum(data, dim=2, keepdim=True)
        new_count = block_count + 1.0
        state_out = torch.cat((new_sum, new_count), dim=2)
        return processed_data, state_out


# StatefulAccumulatorNetwork (TensorFlow)
class StatefulAccumulatorNetworkTF(tf.keras.Model):
    def call(self, inputs):
        state_in, data = inputs
        running_sum = state_in[:, :, 0:1]
        block_count = state_in[:, :, 1:2]
        processed_data = data + running_sum
        new_sum = running_sum + tf.reduce_sum(data, axis=2, keepdims=True)
        new_count = block_count + 1.0
        state_out = tf.concat([new_sum, new_count], axis=2)
        return processed_data, state_out


def export_torchscript_model(
    net: nn.Module, state_in: torch.Tensor, data: torch.Tensor, channels: int
) -> str:
    suffix = channel_suffix(channels)
    filepath = os.path.join(MODEL_DIR, f"stateful_accumulator_network_{suffix}.pt")
    scripted_net = torch.jit.trace(net, (state_in, data))
    scripted_net.save(filepath)
    print(f"Saved TorchScript model to {filepath}")
    return filepath


def export_onnx_model(
    net: nn.Module, state_in: torch.Tensor, data: torch.Tensor, channels: int
) -> str:
    suffix = channel_suffix(channels)
    filepath = os.path.join(MODEL_DIR, f"stateful_accumulator_network_{suffix}.onnx")
    # Use the legacy (TorchScript) exporter so we get a single self-contained
    # opset-11 .onnx file (the dynamo exporter forces opset 18 and emits an
    # external-data sidecar).
    torch.onnx.export(
        net,
        (state_in, data),
        filepath,
        input_names=["state_in", "data"],
        output_names=["processed_data", "state_out"],
        dynamic_axes={"data": {2: "dynamic"}, "processed_data": {2: "dynamic"}},
        opset_version=11,
        dynamo=False,
    )
    print(f"Saved ONNX model to {filepath}")
    return filepath


def export_executorch_model(
    net: nn.Module, state_in: torch.Tensor, data: torch.Tensor, channels: int
) -> str:
    suffix = channel_suffix(channels)
    filepath = os.path.join(MODEL_DIR, f"stateful_accumulator_network_{suffix}.pte")

    # Dynamic sample dimension (axis 2) so the .pte matches the ONNX/TFLite models.
    # The state is static.
    sample_dim = torch.export.Dim("samples", min=1, max=1 << 16)
    dynamic_shapes = {"state_in": None, "data": {2: sample_dim}}

    exported = torch_export(net, (state_in, data), dynamic_shapes=dynamic_shapes)
    edge_program = to_edge(exported)
    executorch_program = edge_program.to_executorch()

    with open(filepath, "wb") as f:
        f.write(executorch_program.buffer)
    print(f"Saved ExecuTorch model to {filepath}")
    return filepath


def convert_tf_to_tflite(model: tf.keras.Model, channels: int) -> str:
    suffix = channel_suffix(channels)
    tflite_model_path = os.path.join(
        MODEL_DIR, f"stateful_accumulator_network_{suffix}.tflite"
    )

    state_shape = tf.TensorSpec([1, channels, STATE_SIZE], dtype=tf.float32)
    data_shape = tf.TensorSpec([1, channels, None], dtype=tf.float32)

    concrete_func = tf.function(model).get_concrete_function([state_shape, data_shape])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"Model converted to TFLite and saved as {tflite_model_path}")
    return tflite_model_path


def run_pytorch_inference(
    net: nn.Module, state_in: np.ndarray, data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        processed_data, state_out = net(
            torch.from_numpy(state_in), torch.from_numpy(data)
        )
    return processed_data.numpy(), state_out.numpy()


def run_torchscript_inference(
    torchscript_model_path: str, state_in: np.ndarray, data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    # Load the saved file, so the trace that is committed is the one that is checked.
    scripted_net = torch.jit.load(torchscript_model_path)
    return run_pytorch_inference(scripted_net, state_in, data)


def run_onnx_inference(
    onnx_model_path: str, state_in: np.ndarray, data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    session = ort.InferenceSession(onnx_model_path)
    outputs = session.run(
        ["processed_data", "state_out"], {"state_in": state_in, "data": data}
    )
    return np.asarray(outputs[0]), np.asarray(outputs[1])


def run_tflite_inference(
    tflite_model_path: str, state_in: np.ndarray, data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    # TFLite does not guarantee that input/output ordering matches the order the
    # tensors were declared, so identify each tensor by its shape rather than by
    # position. Only the data input has a dynamic (-1) sample axis; the state
    # input is the other.
    data_in = next(d for d in input_details if -1 in list(d["shape_signature"]))
    state_in_details = next(d for d in input_details if d["index"] != data_in["index"])

    interpreter.resize_tensor_input(data_in["index"], data.shape)
    interpreter.resize_tensor_input(state_in_details["index"], state_in.shape)
    interpreter.allocate_tensors()  # Re-allocate tensors after resizing

    interpreter.set_tensor(data_in["index"], data)
    interpreter.set_tensor(state_in_details["index"], state_in)

    interpreter.invoke()

    # Re-read details after allocation. processed_data carries the dynamic sample
    # axis (-1); the state is the static output.
    output_details = interpreter.get_output_details()
    data_out = next(d for d in output_details if -1 in list(d["shape_signature"]))
    state_out_details = next(
        d for d in output_details if d["index"] != data_out["index"]
    )

    processed_data = interpreter.get_tensor(data_out["index"])
    state_out = interpreter.get_tensor(state_out_details["index"])
    return processed_data, state_out


def run_executorch_inference(
    pte_model_path: str, state_in: np.ndarray, data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(pte_model_path)
    method = program.load_method("forward")
    outputs = method.execute([torch.from_numpy(state_in), torch.from_numpy(data)])
    # Copy: the returned tensors live in the method's memory.
    return outputs[0].numpy().copy(), outputs[1].numpy().copy()


def print_onnx_tensor_order(onnx_model_path: str) -> None:
    session = ort.InferenceSession(onnx_model_path)
    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    assert input_names == ["state_in", "data"], (
        f"Unexpected ONNX input order: {input_names}"
    )
    assert output_names == ["processed_data", "state_out"], (
        f"Unexpected ONNX output order: {output_names}"
    )
    for position, tensor in enumerate(session.get_inputs()):
        print(f"ONNX input {position}: name={tensor.name}, shape={tensor.shape}")
    for position, tensor in enumerate(session.get_outputs()):
        print(f"ONNX output {position}: name={tensor.name}, shape={tensor.shape}")


def print_tflite_tensor_order(tflite_model_path: str) -> None:
    # A consumer that binds TFLite tensors by position needs this order: it is
    # the converter's, not necessarily the declared one.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    for kind, details, roles in (
        ("input", interpreter.get_input_details(), ("data", "state_in")),
        ("output", interpreter.get_output_details(), ("processed_data", "state_out")),
    ):
        for position, d in enumerate(details):
            shape_signature = [int(s) for s in d["shape_signature"]]
            role = roles[0] if -1 in shape_signature else roles[1]
            print(
                f"TFLite {kind} {position}: {role}, name={d['name']}, "
                f"index={d['index']}, shape_signature={shape_signature}"
            )
    print(f"TFLite signatures: {interpreter.get_signature_list()}")


def make_test_blocks(channels: int, num_samples: int) -> List[np.ndarray]:
    """Small integer-valued blocks, different per block and per channel: every value stays exact in float32."""
    base = np.arange(num_samples, dtype=np.int64) % 5 - 1
    blocks = []
    for k in range(NUM_TEST_BLOCKS):
        block = np.stack([base * (c + 1) + k for c in range(channels)])
        blocks.append(block[np.newaxis, :, :])
    return blocks


def expected_outputs(
    blocks: List[np.ndarray], k: int, feedback: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed form for block k, computed in integers. With the state fed back the output is the input
    plus the sum of every sample of blocks 0..k-1; without feedback (zero state) it is the input."""
    first = 0 if feedback else k
    sum_before = sum(
        (b.sum(axis=2, keepdims=True) for b in blocks[first:k]),
        np.zeros_like(blocks[k][:, :, 0:1]),
    )
    sum_through = sum_before + blocks[k].sum(axis=2, keepdims=True)
    block_count = np.full_like(sum_through, k + 1 - first)
    processed_data = blocks[k] + sum_before
    state_out = np.concatenate([sum_through, block_count], axis=2)
    return processed_data.astype(np.float32), state_out.astype(np.float32)


def check_exact(output: np.ndarray, expected: np.ndarray, name: str = "") -> None:
    assert output.dtype == np.float32, f"{name} is {output.dtype}, not float32"
    assert np.array_equal(output, expected), f"Mismatch in {name}"


def check_closed_form(
    run_inference: InferenceFn, channels: int, name: str = ""
) -> None:
    for num_samples in TEST_BLOCK_SIZES:
        blocks = make_test_blocks(channels, num_samples)
        zero_state = np.zeros((1, channels, STATE_SIZE), dtype=np.float32)

        # State fed back between the blocks, as the consumer does.
        state = zero_state
        for k, block in enumerate(blocks):
            processed_data, state = run_inference(state, block.astype(np.float32))
            expected_data, expected_state = expected_outputs(blocks, k, feedback=True)
            check_exact(
                processed_data,
                expected_data,
                name=f"{name} processed_data (N={num_samples}, block {k}, feedback)",
            )
            check_exact(
                state,
                expected_state,
                name=f"{name} state_out (N={num_samples}, block {k}, feedback)",
            )

        # Zero state at every block: the output is the input.
        for k, block in enumerate(blocks):
            processed_data, state_out = run_inference(
                zero_state, block.astype(np.float32)
            )
            expected_data, expected_state = expected_outputs(blocks, k, feedback=False)
            check_exact(
                processed_data,
                expected_data,
                name=f"{name} processed_data (N={num_samples}, block {k}, no feedback)",
            )
            check_exact(
                state_out,
                expected_state,
                name=f"{name} state_out (N={num_samples}, block {k}, no feedback)",
            )

        print(
            f"{name} (N={num_samples}): {NUM_TEST_BLOCKS} blocks match the closed form exactly, "
            f"with state feedback and without. Final state with feedback: {state.tolist()}"
        )


def build_models_for_channels(channels: int) -> None:
    print(
        f"\n=== Building StatefulAccumulatorNetwork for {channels} channel(s) ({channel_suffix(channels)}) ==="
    )

    # Create PyTorch model and convert to LibTorch and OnnxRuntime
    net = StatefulAccumulatorNetwork()
    state_in = torch.zeros(1, channels, STATE_SIZE)
    data = torch.randn(1, channels, 10)
    torchscript_model_path = export_torchscript_model(net, state_in, data, channels)
    onnx_model_path = export_onnx_model(net, state_in, data, channels)

    # Export ExecuTorch (.pte)
    pte_model_path = None
    if _HAS_EXECUTORCH:
        pte_model_path = export_executorch_model(net, state_in, data, channels)
    else:
        print(
            "ExecuTorch not installed; skipping .pte export. Install with: pip install executorch"
        )

    # Create TensorFlow model and convert to TFLite
    tf_model = StatefulAccumulatorNetworkTF()
    tf_model([tf.constant(state_in.numpy()), tf.constant(data.numpy())])
    tflite_model_path = convert_tf_to_tflite(tf_model, channels)

    # Tensor order of the exports
    print_onnx_tensor_order(onnx_model_path)
    print_tflite_tensor_order(tflite_model_path)

    # Run every export over a sequence of blocks and check it against the closed form
    check_closed_form(
        lambda s, d: run_pytorch_inference(net, s, d), channels, name="PyTorch"
    )
    check_closed_form(
        lambda s, d: run_torchscript_inference(torchscript_model_path, s, d),
        channels,
        name="TorchScript",
    )
    check_closed_form(
        lambda s, d: run_onnx_inference(onnx_model_path, s, d), channels, name="ONNX"
    )
    check_closed_form(
        lambda s, d: run_tflite_inference(tflite_model_path, s, d),
        channels,
        name="TFLite",
    )

    if pte_model_path is not None:
        check_closed_form(
            lambda s, d: run_executorch_inference(pte_model_path, s, d),
            channels,
            name="ExecuTorch",
        )

    print(
        f"All tests passed for {channels} channel(s)! "
        f"PyTorch, TorchScript, ONNX, TFLite{', and ExecuTorch' if pte_model_path else ''} outputs match the closed form."
    )


def main(channel_counts=(1, 2)):
    for channels in channel_counts:
        build_models_for_channels(channels)


if __name__ == "__main__":
    main()
