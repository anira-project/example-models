import os
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import torch
from torch import nn

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

# The model of each block of the sequence that switches between the two models:
# False is the accumulator, True its negated twin.
SWITCH_SEQUENCE = (False, False, True, True, False)

InferenceFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class BuiltModel(NamedTuple):
    """The exports of one model: how to run each of them, and the tensors each file reports."""

    runners: dict[str, InferenceFn]
    tensors: dict[str, list[str]]


def channel_suffix(channels: int) -> str:
    """Map a channel count to a file suffix (mono/stereo for 1/2, NchN otherwise)."""
    if channels == 1:
        return "mono"
    if channels == 2:
        return "stereo"
    return f"{channels}ch"


def model_path(channels: int, negated: bool, extension: str) -> str:
    """stateful_accumulator_network_<suffix>.<extension>, the negated twin with _negated before the suffix."""
    twin = "_negated" if negated else ""
    name = f"stateful_accumulator_network{twin}_{channel_suffix(channels)}.{extension}"
    return os.path.join(MODEL_DIR, name)


# StatefulAccumulatorNetwork (PyTorch)
# The state is an ordinary input and an ordinary output: the model keeps nothing
# between calls, the caller feeds state_out back as the next state_in. The state
# is deliberately the FIRST input and the LAST output.
#
# The negated twin is a second model of the identical tensor set that differs in
# one rule: it subtracts the running sum from the block where the accumulator
# adds it. Its state_out follows the accumulator's rule, so either model continues
# from the other's state, and the sign of the running sum in processed_data tells
# which of the two ran a block.
class StatefulAccumulatorNetwork(nn.Module):
    def __init__(self, negated: bool = False):
        super().__init__()
        self.negated = negated

    def forward(self, state_in, data) -> tuple[torch.Tensor, torch.Tensor]:
        running_sum = state_in[:, :, 0:1]
        block_count = state_in[:, :, 1:2]
        if self.negated:
            processed_data = data - running_sum
        else:
            processed_data = data + running_sum
        new_sum = running_sum + torch.sum(data, dim=2, keepdim=True)
        new_count = block_count + 1.0
        state_out = torch.cat((new_sum, new_count), dim=2)
        return processed_data, state_out


# StatefulAccumulatorNetwork (TensorFlow)
class StatefulAccumulatorNetworkTF(tf.keras.Model):
    def __init__(self, negated: bool = False):
        super().__init__()
        self.negated = negated

    def call(self, inputs):
        state_in, data = inputs
        running_sum = state_in[:, :, 0:1]
        block_count = state_in[:, :, 1:2]
        if self.negated:
            processed_data = data - running_sum
        else:
            processed_data = data + running_sum
        new_sum = running_sum + tf.reduce_sum(data, axis=2, keepdims=True)
        new_count = block_count + 1.0
        state_out = tf.concat([new_sum, new_count], axis=2)
        return processed_data, state_out


def export_torchscript_model(
    net: nn.Module,
    state_in: torch.Tensor,
    data: torch.Tensor,
    channels: int,
    negated: bool,
) -> str:
    filepath = model_path(channels, negated, "pt")
    scripted_net = torch.jit.trace(net, (state_in, data))
    scripted_net.save(filepath)
    print(f"Saved TorchScript model to {filepath}")
    return filepath


def export_onnx_model(
    net: nn.Module,
    state_in: torch.Tensor,
    data: torch.Tensor,
    channels: int,
    negated: bool,
) -> str:
    filepath = model_path(channels, negated, "onnx")
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
    net: nn.Module,
    state_in: torch.Tensor,
    data: torch.Tensor,
    channels: int,
    negated: bool,
) -> str:
    filepath = model_path(channels, negated, "pte")

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


def convert_tf_to_tflite(model: tf.keras.Model, channels: int, negated: bool) -> str:
    tflite_model_path = model_path(channels, negated, "tflite")

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
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        processed_data, state_out = net(
            torch.from_numpy(state_in), torch.from_numpy(data)
        )
    return processed_data.numpy(), state_out.numpy()


def run_torchscript_inference(
    torchscript_model_path: str, state_in: np.ndarray, data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Load the saved file, so the trace that is committed is the one that is checked.
    scripted_net = torch.jit.load(torchscript_model_path)
    return run_pytorch_inference(scripted_net, state_in, data)


def run_onnx_inference(
    onnx_model_path: str, state_in: np.ndarray, data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    session = ort.InferenceSession(onnx_model_path)
    outputs = session.run(
        ["processed_data", "state_out"], {"state_in": state_in, "data": data}
    )
    return np.asarray(outputs[0]), np.asarray(outputs[1])


def run_tflite_inference(
    tflite_model_path: str, state_in: np.ndarray, data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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
) -> tuple[np.ndarray, np.ndarray]:
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(pte_model_path)
    method = program.load_method("forward")
    outputs = method.execute([torch.from_numpy(state_in), torch.from_numpy(data)])
    # Copy: the returned tensors live in the method's memory.
    return outputs[0].numpy().copy(), outputs[1].numpy().copy()


# The tensors of a file as its runtime reports them, one line per tensor. The
# negated twin has to report the lines of the accumulator in every format: that
# is what makes the two models variants of one form.
def torchscript_tensors(torchscript_model_path: str) -> list[str]:
    scripted_net = torch.jit.load(torchscript_model_path)
    schema = scripted_net.forward.schema
    lines = [
        f"TorchScript input {position}: name={argument.name}, type={argument.type}"
        for position, argument in enumerate(schema.arguments[1:])  # without self
    ]
    lines += [
        f"TorchScript output {position}: type={returned.type}"
        for position, returned in enumerate(schema.returns)
    ]
    return lines


def onnx_tensors(onnx_model_path: str) -> list[str]:
    session = ort.InferenceSession(onnx_model_path)
    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    assert input_names == ["state_in", "data"], (
        f"Unexpected ONNX input order: {input_names}"
    )
    assert output_names == ["processed_data", "state_out"], (
        f"Unexpected ONNX output order: {output_names}"
    )
    lines = [
        f"ONNX input {position}: name={tensor.name}, shape={tensor.shape}, type={tensor.type}"
        for position, tensor in enumerate(session.get_inputs())
    ]
    lines += [
        f"ONNX output {position}: name={tensor.name}, shape={tensor.shape}, type={tensor.type}"
        for position, tensor in enumerate(session.get_outputs())
    ]
    return lines


def tflite_tensors(tflite_model_path: str) -> list[str]:
    # A consumer that binds TFLite tensors by position needs this order: it is
    # the converter's, not necessarily the declared one.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    lines = []
    for kind, details, roles in (
        ("input", interpreter.get_input_details(), ("data", "state_in")),
        ("output", interpreter.get_output_details(), ("processed_data", "state_out")),
    ):
        for position, d in enumerate(details):
            shape_signature = [int(s) for s in d["shape_signature"]]
            role = roles[0] if -1 in shape_signature else roles[1]
            lines.append(
                f"TFLite {kind} {position}: {role}, name={d['name']}, "
                f"shape_signature={shape_signature}, dtype={np.dtype(d['dtype']).name}"
            )
    signatures = interpreter.get_signature_list()
    lines.append(f"TFLite signatures: {signatures}")
    for signature_key in signatures:
        runner = interpreter.get_signature_runner(signature_key)
        for kind, details in (
            ("input", runner.get_input_details()),
            ("output", runner.get_output_details()),
        ):
            for key, d in details.items():
                lines.append(
                    f"TFLite signature {signature_key} {kind} {key}: name={d['name']}"
                )
    return lines


def executorch_tensors(pte_model_path: str) -> list[str]:
    from executorch.runtime import Runtime

    program = Runtime.get().load_program(pte_model_path)
    assert program.method_names == {"forward"}, (
        f"Unexpected methods: {program.method_names}"
    )
    metadata = program.load_method("forward").metadata
    # A dynamic axis reads as its upper bound here.
    lines = [
        f"ExecuTorch input {position}: {metadata.input_tensor_meta(position)}"
        for position in range(metadata.num_inputs())
    ]
    lines += [
        f"ExecuTorch output {position}: {metadata.output_tensor_meta(position)}"
        for position in range(metadata.num_outputs())
    ]
    return lines


def make_test_blocks(
    channels: int, num_samples: int, num_blocks: int = NUM_TEST_BLOCKS
) -> list[np.ndarray]:
    """Small integer-valued blocks, different per block and per channel: every value stays exact in float32."""
    base = np.arange(num_samples, dtype=np.int64) % 5 - 1
    blocks = []
    for k in range(num_blocks):
        block = np.stack([base * (c + 1) + k for c in range(channels)])
        blocks.append(block[np.newaxis, :, :])
    return blocks


def expected_outputs(
    blocks: list[np.ndarray], k: int, first: int, negated: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Closed form for block k, computed in integers, of a sequence whose state was zero at block
    `first` and fed back since. The output is the input plus (the accumulator) or minus (its negated
    twin) the sum of every sample of blocks first..k-1, so with first = k (zero state) it is the input.
    state_out is the same for both models: it does not tell which of them ran the blocks first..k."""
    sum_before = sum(
        (b.sum(axis=2, keepdims=True) for b in blocks[first:k]),
        np.zeros_like(blocks[k][:, :, 0:1]),
    )
    sum_through = sum_before + blocks[k].sum(axis=2, keepdims=True)
    block_count = np.full_like(sum_through, k + 1 - first)
    processed_data = blocks[k] - sum_before if negated else blocks[k] + sum_before
    state_out = np.concatenate([sum_through, block_count], axis=2)
    return processed_data.astype(np.float32), state_out.astype(np.float32)


def check_exact(output: np.ndarray, expected: np.ndarray, name: str = "") -> None:
    assert output.dtype == np.float32, f"{name} is {output.dtype}, not float32"
    assert np.array_equal(output, expected), f"Mismatch in {name}"


def check_closed_form(
    run_inference: InferenceFn, channels: int, negated: bool, name: str = ""
) -> None:
    for num_samples in TEST_BLOCK_SIZES:
        blocks = make_test_blocks(channels, num_samples)
        zero_state = np.zeros((1, channels, STATE_SIZE), dtype=np.float32)

        # State fed back between the blocks, as the consumer does.
        state = zero_state
        for k, block in enumerate(blocks):
            processed_data, state = run_inference(state, block.astype(np.float32))
            expected_data, expected_state = expected_outputs(blocks, k, 0, negated)
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
            expected_data, expected_state = expected_outputs(blocks, k, k, negated)
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


def check_identical_tensors(accumulator: BuiltModel, negated_twin: BuiltModel) -> None:
    """The negated twin reports the tensors of the accumulator in every format: names, order,
    shapes and dtypes."""
    assert accumulator.tensors.keys() == negated_twin.tensors.keys(), (
        "The negated twin has other formats than the accumulator"
    )
    for file_format, lines in accumulator.tensors.items():
        assert negated_twin.tensors[file_format] == lines, (
            f"The {file_format} tensors of the negated twin differ from the accumulator's: "
            f"{negated_twin.tensors[file_format]} != {lines}"
        )
    print(
        f"The negated twin reports the tensors of the accumulator in "
        f"{', '.join(accumulator.tensors)}: one form, two models."
    )


def check_model_switch(
    accumulator: BuiltModel, negated_twin: BuiltModel, channels: int
) -> None:
    """Change the model between two blocks of one sequence, as a host does that selects another
    variant while audio runs: the model that takes over either continues from the state_out of
    the model before it ("keep") or starts from zeros ("fresh"). Run in every runtime, with
    both files of that runtime."""
    models = ", ".join(
        "negated" if negated else "accumulator" for negated in SWITCH_SEQUENCE
    )
    for runtime in accumulator.runners:
        for num_samples in TEST_BLOCK_SIZES:
            blocks = make_test_blocks(channels, num_samples, len(SWITCH_SEQUENCE))
            zero_state = np.zeros((1, channels, STATE_SIZE), dtype=np.float32)
            final_states = {}

            for keep in (True, False):
                mode = "state kept" if keep else "fresh state"
                state = zero_state
                first = 0
                for k, block in enumerate(blocks):
                    negated = SWITCH_SEQUENCE[k]
                    if k > 0 and negated != SWITCH_SEQUENCE[k - 1] and not keep:
                        state = zero_state
                        first = k
                    model = negated_twin if negated else accumulator
                    processed_data, state = model.runners[runtime](
                        state, block.astype(np.float32)
                    )
                    expected_data, expected_state = expected_outputs(
                        blocks, k, first, negated
                    )
                    check_exact(
                        processed_data,
                        expected_data,
                        name=f"{runtime} processed_data (N={num_samples}, block {k}, {mode})",
                    )
                    check_exact(
                        state,
                        expected_state,
                        name=f"{runtime} state_out (N={num_samples}, block {k}, {mode})",
                    )
                final_states[keep] = state.tolist()

            print(
                f"{runtime} (N={num_samples}): the sequence {models} matches the closed form exactly, "
                f"with the state kept at each switch (final state {final_states[True]}) "
                f"and with a fresh state at each switch (final state {final_states[False]})."
            )


def build_models_for_channels(channels: int, negated: bool) -> BuiltModel:
    print(
        f"\n=== Building {'the negated twin of ' if negated else ''}StatefulAccumulatorNetwork "
        f"for {channels} channel(s) ({channel_suffix(channels)}) ==="
    )

    # Create PyTorch model and convert to LibTorch and OnnxRuntime
    net = StatefulAccumulatorNetwork(negated)
    state_in = torch.zeros(1, channels, STATE_SIZE)
    data = torch.randn(1, channels, 10)
    torchscript_model_path = export_torchscript_model(
        net, state_in, data, channels, negated
    )
    onnx_model_path = export_onnx_model(net, state_in, data, channels, negated)

    # Export ExecuTorch (.pte)
    pte_model_path = None
    if _HAS_EXECUTORCH:
        pte_model_path = export_executorch_model(net, state_in, data, channels, negated)
    else:
        print(
            "ExecuTorch not installed; skipping .pte export. Install with: pip install executorch"
        )

    # Create TensorFlow model and convert to TFLite
    tf_model = StatefulAccumulatorNetworkTF(negated)
    tf_model([tf.constant(state_in.numpy()), tf.constant(data.numpy())])
    tflite_model_path = convert_tf_to_tflite(tf_model, channels, negated)

    # Tensor order of the exports
    tensors = {
        "TorchScript": torchscript_tensors(torchscript_model_path),
        "ONNX": onnx_tensors(onnx_model_path),
        "TFLite": tflite_tensors(tflite_model_path),
    }
    if pte_model_path is not None:
        tensors["ExecuTorch"] = executorch_tensors(pte_model_path)
    for lines in tensors.values():
        for line in lines:
            print(line)

    # Run every export over a sequence of blocks and check it against the closed form
    runners: dict[str, InferenceFn] = {
        "PyTorch": lambda s, d: run_pytorch_inference(net, s, d),
        "TorchScript": lambda s, d: run_torchscript_inference(
            torchscript_model_path, s, d
        ),
        "ONNX": lambda s, d: run_onnx_inference(onnx_model_path, s, d),
        "TFLite": lambda s, d: run_tflite_inference(tflite_model_path, s, d),
    }
    if pte_model_path is not None:
        runners["ExecuTorch"] = lambda s, d: run_executorch_inference(
            pte_model_path, s, d
        )
    for runtime, run_inference in runners.items():
        check_closed_form(run_inference, channels, negated, name=runtime)

    print(
        f"All tests passed for {channels} channel(s)! "
        f"PyTorch, TorchScript, ONNX, TFLite{', and ExecuTorch' if pte_model_path else ''} outputs match the closed form."
    )
    return BuiltModel(runners, tensors)


def main(channel_counts=(1, 2)):
    # Every accumulator before the first twin: the converter names the inner tensors of a
    # TFLite file after the count of Keras models built before it, so this order rebuilds
    # the accumulator files as they were before the twin existed.
    accumulators = {
        channels: build_models_for_channels(channels, negated=False)
        for channels in channel_counts
    }
    negated_twins = {
        channels: build_models_for_channels(channels, negated=True)
        for channels in channel_counts
    }

    for channels in channel_counts:
        accumulator = accumulators[channels]
        negated_twin = negated_twins[channels]
        print(
            f"\n=== Switching between StatefulAccumulatorNetwork and its negated twin "
            f"for {channels} channel(s) ({channel_suffix(channels)}) ==="
        )
        check_identical_tensors(accumulator, negated_twin)
        check_model_switch(accumulator, negated_twin, channels)


if __name__ == "__main__":
    main()
