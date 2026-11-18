"""VariantGainNetwork: one architecture, three weights.

anira switches between the models of one handler while audio runs ("model variants"): the
variants have the identical stream form (the same tensors, names, order, shapes and dtypes) and
differ in what they compute. These three models are one Conv1d(1, 1, kernel_size=1, bias=False)
whose single weight is the only difference:

- variant_gain_network_half_mono.{pt,onnx,tflite,pte}:   processed_data =  0.5 * data
- variant_gain_network_double_mono.{pt,onnx,tflite,pte}: processed_data =  2.0 * data
- variant_gain_network_invert_mono.{pt,onnx,tflite,pte}: processed_data = -1.0 * data

One input data [1, 1, N] and one output processed_data [1, 1, N], both float32, N dynamic. Every
weight is a power of two or -1, so the product is exact in float32 for every sample and in every
runtime, and a consumer tells from any non-zero chunk which model produced it.

Every export is run in its runtime and checked against w * data with exact equality, and the
three variants of a format are asserted to report identical tensors.
"""

import os
from collections.abc import Callable

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

# The variants: the word in the file name and the one weight of the convolution.
VARIANTS: dict[str, float] = {"half": 0.5, "double": 2.0, "invert": -1.0}

# Every export is verified with these block sizes.
TEST_BLOCK_SIZES = (10, 64, 512)

InferenceFn = Callable[[np.ndarray], np.ndarray]


def model_path(variant: str, extension: str) -> str:
    return os.path.join(MODEL_DIR, f"variant_gain_network_{variant}_mono.{extension}")


# VariantGainNetwork (PyTorch)
# The architecture of every variant; the weight is all that a variant sets.
class VariantGainNetwork(nn.Module):
    def __init__(self, weight: float):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(weight)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.conv(data)


# VariantGainNetwork (TensorFlow)
class VariantGainNetworkTF(tf.keras.Model):
    def __init__(self, weight: float):
        # One name for every variant: the converter names the inner tensors of the file after
        # the model, and Keras would number the models it names itself.
        super().__init__(name="variant_gain_network")
        # The kernel is a constant, [width, in_channels, out_channels]: the converter leaves a
        # Keras layer's variable in the file as a resource variable that nothing initialises.
        self.kernel = tf.constant(weight, shape=[1, 1, 1], dtype=tf.float32)

    def call(self, data):
        # TensorFlow convolves [batch, time, channel]; the model's tensors are [batch, channel, time].
        channels_last = tf.transpose(data, [0, 2, 1])
        processed_data = tf.nn.conv1d(
            channels_last, self.kernel, stride=1, padding="VALID"
        )
        return tf.transpose(processed_data, [0, 2, 1])


def export_torchscript_model(net: nn.Module, data: torch.Tensor, variant: str) -> str:
    filepath = model_path(variant, "pt")
    scripted_net = torch.jit.trace(net, (data,))
    scripted_net.save(filepath)
    print(f"Saved TorchScript model to {filepath}")
    return filepath


def export_onnx_model(net: nn.Module, data: torch.Tensor, variant: str) -> str:
    filepath = model_path(variant, "onnx")
    # Use the legacy (TorchScript) exporter so we get a single self-contained
    # opset-11 .onnx file (the dynamo exporter forces opset 18 and emits an
    # external-data sidecar).
    torch.onnx.export(
        net,
        (data,),
        filepath,
        input_names=["data"],
        output_names=["processed_data"],
        dynamic_axes={"data": {2: "dynamic"}, "processed_data": {2: "dynamic"}},
        opset_version=11,
        dynamo=False,
    )
    print(f"Saved ONNX model to {filepath}")
    return filepath


def export_executorch_model(net: nn.Module, data: torch.Tensor, variant: str) -> str:
    filepath = model_path(variant, "pte")

    # Dynamic sample dimension (axis 2) so the .pte matches the ONNX/TFLite models.
    sample_dim = torch.export.Dim("samples", min=1, max=1 << 16)
    dynamic_shapes = {"data": {2: sample_dim}}

    exported = torch_export(net, (data,), dynamic_shapes=dynamic_shapes)
    edge_program = to_edge(exported)
    executorch_program = edge_program.to_executorch()

    with open(filepath, "wb") as f:
        f.write(executorch_program.buffer)
    print(f"Saved ExecuTorch model to {filepath}")
    return filepath


def convert_tf_to_tflite(model: tf.keras.Model, variant: str) -> str:
    tflite_model_path = model_path(variant, "tflite")

    data_shape = tf.TensorSpec([1, 1, None], dtype=tf.float32)

    concrete_func = tf.function(model).get_concrete_function(data_shape)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"Model converted to TFLite and saved as {tflite_model_path}")
    return tflite_model_path


def run_pytorch_inference(net: nn.Module, data: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        processed_data = net(torch.from_numpy(data))
    return processed_data.numpy()


def run_torchscript_inference(
    torchscript_model_path: str, data: np.ndarray
) -> np.ndarray:
    # Load the saved file, so the trace that is committed is the one that is checked.
    scripted_net = torch.jit.load(torchscript_model_path)
    return run_pytorch_inference(scripted_net, data)


def run_onnx_inference(onnx_model_path: str, data: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(onnx_model_path)
    outputs = session.run(["processed_data"], {"data": data})
    return np.asarray(outputs[0])


def run_tflite_inference(tflite_model_path: str, data: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    data_in = interpreter.get_input_details()[0]

    interpreter.resize_tensor_input(data_in["index"], data.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(data_in["index"], data)

    interpreter.invoke()

    data_out = interpreter.get_output_details()[0]
    return interpreter.get_tensor(data_out["index"])


def run_executorch_inference(pte_model_path: str, data: np.ndarray) -> np.ndarray:
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(pte_model_path)
    method = program.load_method("forward")
    outputs = method.execute([torch.from_numpy(data)])
    # Copy: the returned tensor lives in the method's memory.
    return outputs[0].numpy().copy()


# The tensors of a file as its runtime reports them, one line per tensor. The lines of the
# three variants of a format have to be identical: that is what makes them variants.
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
    lines += [
        f"TorchScript parameter: name={name}, shape={list(parameter.shape)}, dtype={parameter.dtype}"
        for name, parameter in scripted_net.named_parameters()
    ]
    return lines


def onnx_tensors(onnx_model_path: str) -> list[str]:
    session = ort.InferenceSession(onnx_model_path)
    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    assert input_names == ["data"], f"Unexpected ONNX inputs: {input_names}"
    assert output_names == ["processed_data"], (
        f"Unexpected ONNX outputs: {output_names}"
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
    # A consumer that binds TFLite tensors by position or by name needs these: the names are
    # the converter's, not the declared ones.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    lines = []
    for kind, details in (
        ("input", interpreter.get_input_details()),
        ("output", interpreter.get_output_details()),
    ):
        for position, d in enumerate(details):
            shape_signature = [int(s) for s in d["shape_signature"]]
            scale, zero_point = d["quantization"]
            assert scale == 0.0 and zero_point == 0, (
                f"{d['name']} carries quantisation parameters"
            )
            lines.append(
                f"TFLite {kind} {position}: name={d['name']}, shape_signature={shape_signature}, "
                f"dtype={np.dtype(d['dtype']).name}"
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


def make_test_blocks(num_samples: int) -> list[np.ndarray]:
    """Two blocks: small integers, and fractions that are not exact binary fractions of the weights' scale."""
    integers = (np.arange(num_samples, dtype=np.int64) % 7 - 3).astype(np.float32)
    fractions = np.linspace(-1.0, 1.0, num_samples, dtype=np.float32)
    return [block.reshape(1, 1, num_samples) for block in (integers, fractions)]


def expected_output(weight: float, data: np.ndarray) -> np.ndarray:
    """Closed form w * data, computed in float64. The weights are a power of two or -1, so the product
    of a float32 sample is a float32 value again: nothing rounds, in whatever order a runtime computes."""
    product = np.float64(weight) * data.astype(np.float64)
    expected = product.astype(np.float32)
    assert np.array_equal(expected.astype(np.float64), product), (
        "the expectation is not exact in float32"
    )
    return expected


def check_exact(output: np.ndarray, expected: np.ndarray, name: str = "") -> None:
    assert output.dtype == np.float32, f"{name} is {output.dtype}, not float32"
    assert output.shape == expected.shape, (
        f"{name} is {output.shape}, not {expected.shape}"
    )
    assert np.array_equal(output, expected), f"Mismatch in {name}"


def check_closed_form(
    run_inference: InferenceFn, weight: float, name: str = ""
) -> None:
    for num_samples in TEST_BLOCK_SIZES:
        for k, block in enumerate(make_test_blocks(num_samples)):
            processed_data = run_inference(block)
            check_exact(
                processed_data,
                expected_output(weight, block),
                name=f"{name} processed_data (N={num_samples}, block {k})",
            )
    print(
        f"{name}: processed_data == {weight} * data exactly, for N in {TEST_BLOCK_SIZES} "
        f"with integer and fractional samples."
    )


def build_variant(variant: str, weight: float) -> dict[str, list[str]]:
    print(f"\n=== Building VariantGainNetwork '{variant}' (weight {weight}) ===")

    # Create PyTorch model and convert to LibTorch and OnnxRuntime
    net = VariantGainNetwork(weight)
    data = torch.zeros(1, 1, 10)
    torchscript_model_path = export_torchscript_model(net, data, variant)
    onnx_model_path = export_onnx_model(net, data, variant)

    # Export ExecuTorch (.pte)
    pte_model_path = None
    if _HAS_EXECUTORCH:
        pte_model_path = export_executorch_model(net, data, variant)
    else:
        print(
            "ExecuTorch not installed; skipping .pte export. Install with: pip install executorch"
        )

    # Create TensorFlow model and convert to TFLite
    tf_model = VariantGainNetworkTF(weight)
    tf_model(tf.constant(data.numpy()))
    tflite_model_path = convert_tf_to_tflite(tf_model, variant)

    # Tensors of the exports
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

    # Run every export and check it against the closed form
    check_closed_form(
        lambda d: run_pytorch_inference(net, d), weight, name=f"{variant} PyTorch"
    )
    check_closed_form(
        lambda d: run_torchscript_inference(torchscript_model_path, d),
        weight,
        name=f"{variant} TorchScript",
    )
    check_closed_form(
        lambda d: run_onnx_inference(onnx_model_path, d), weight, name=f"{variant} ONNX"
    )
    check_closed_form(
        lambda d: run_tflite_inference(tflite_model_path, d),
        weight,
        name=f"{variant} TFLite",
    )

    if pte_model_path is not None:
        check_closed_form(
            lambda d: run_executorch_inference(pte_model_path, d),
            weight,
            name=f"{variant} ExecuTorch",
        )

    print(
        f"All tests passed for '{variant}'! "
        f"PyTorch, TorchScript, ONNX, TFLite{', and ExecuTorch' if pte_model_path else ''} outputs match the closed form."
    )
    return tensors


def check_identical_tensors(
    tensors_per_variant: dict[str, dict[str, list[str]]],
) -> None:
    """The variants of one format report the same tensors: names, order, shapes and dtypes."""
    reference_variant = next(iter(tensors_per_variant))
    reference = tensors_per_variant[reference_variant]
    for variant, tensors in tensors_per_variant.items():
        assert tensors.keys() == reference.keys(), (
            f"'{variant}' has other formats than '{reference_variant}'"
        )
        for runtime, lines in tensors.items():
            assert lines == reference[runtime], (
                f"The {runtime} tensors of '{variant}' differ from those of '{reference_variant}': "
                f"{lines} != {reference[runtime]}"
            )
    print(
        f"\nThe variants {list(tensors_per_variant)} report identical tensors in "
        f"{', '.join(reference)}: one form, {len(tensors_per_variant)} models."
    )


def main() -> None:
    tensors_per_variant = {
        variant: build_variant(variant, weight) for variant, weight in VARIANTS.items()
    }
    check_identical_tensors(tensors_per_variant)


if __name__ == "__main__":
    main()
