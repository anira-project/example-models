"""TypedGainNetwork: the SimpleGainNetwork tensor set at other dtypes than float32.

anira validates that model tensors of any dtype travel end to end through its engines. These
models do the arithmetic in float32 and cast at the boundary, so every runtime only needs a Cast
for the boundary dtype (its kernels for the compute stay the float32 ones), and with small
integer-valued inputs and a power-of-two gain every output is exact:

- typed_gain_network_<dtype>.{pt,onnx,tflite,pte}, dtype in int16, int8, float16:
  inputs  data [1, 1, N] and gain [1] ([1, 1, 1] in the TFLite file), both of <dtype>;
  outputs processed_data = data * gain and peak = max |processed_data| ([1]), both of <dtype>.
- typed_gain_network_<dtype>_unplanned.pte: the same method at a static [1, 1, 512] window, with
  its inputs and outputs left out of ExecuTorch's memory plan, so a runtime binds its caller's
  memory to them (Method::set_input aliases, Method::set_output_data_ptr) instead of copying. The
  window is static because set_output_data_ptr takes a buffer of the output's capacity, which a
  dynamic axis would make its upper bound.
- dtype_matrix_<dtype>.onnx, for the ten dtypes anira names (float32, float64, float16, bfloat16,
  int8, uint8, int16, int32, int64, bool): one input data [1, 1, N] and one output
  processed_data of that dtype, processed_data = Cast(Cast(data, float32) * 2, <dtype>), written
  with onnx.helper (opset 17, IR version 8). A bool doubles to true for every true element.
- quantized_gain_network_int8.tflite: a fully integer TFLite model, y = 0.5 * x on [1, 1, 64],
  whose input and output are int8 tensors that carry a scale and a zero point. anira refuses
  quantised I/O at load; this file proves the refusal.

Every export is checked against the expectation with exact equality (the quantised model by its
tensor types and parameters).
"""
import ctypes
import os
from typing import Tuple

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import torch
import torch.nn as nn
from onnx import TensorProto, helper

try:
    from executorch.exir import ExecutorchBackendConfig, to_edge
    from executorch.exir.passes import MemoryPlanningPass
    from torch.export import export as torch_export
    _HAS_EXECUTORCH = True
except ImportError:
    _HAS_EXECUTORCH = False

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

GAIN = 2.0
# The unplanned-I/O .pte has a static window: the gain fixture's 512 samples.
UNPLANNED_SAMPLES = 512
# The dtypes of the four-format set: the torch dtype, the TensorFlow dtype, the numpy dtype.
TYPED = {
    "int16": (torch.int16, tf.int16, np.int16),
    "int8": (torch.int8, tf.int8, np.int8),
    "float16": (torch.float16, tf.float16, np.float16),
}
# The ten dtypes of the ONNX matrix: the ONNX element type, the numpy dtype (None: no numpy
# dtype, checked through its bytes).
MATRIX = {
    "float32": (TensorProto.FLOAT, np.float32),
    "float64": (TensorProto.DOUBLE, np.float64),
    "float16": (TensorProto.FLOAT16, np.float16),
    "bfloat16": (TensorProto.BFLOAT16, None),
    "int8": (TensorProto.INT8, np.int8),
    "uint8": (TensorProto.UINT8, np.uint8),
    "int16": (TensorProto.INT16, np.int16),
    "int32": (TensorProto.INT32, np.int32),
    "int64": (TensorProto.INT64, np.int64),
    "bool": (TensorProto.BOOL, np.bool_),
}


# TypedGainNetwork (PyTorch)
class TypedGainNetwork(nn.Module):
    def __init__(self, dtype):
        super(TypedGainNetwork, self).__init__()
        self.dtype = dtype

    def forward(self, data, gain) -> Tuple[torch.Tensor, torch.Tensor]:
        processed_data = data.to(torch.float32) * gain.to(torch.float32)
        peak = torch.max(torch.abs(processed_data)).view(1)
        return processed_data.to(self.dtype), peak.to(self.dtype)


# TypedGainNetwork (TensorFlow)
class TypedGainNetworkTF(tf.keras.Model):
    def __init__(self, dtype):
        super(TypedGainNetworkTF, self).__init__()
        self.out_dtype = dtype

    def call(self, inputs):
        data, gain = inputs
        processed_data = tf.cast(data, tf.float32) * tf.cast(gain, tf.float32)
        peak = tf.reshape(tf.reduce_max(tf.abs(processed_data)), [1])
        return tf.cast(processed_data, self.out_dtype), tf.cast(peak, self.out_dtype)


def path_of(name):
    return os.path.join(MODEL_DIR, name)


def example_inputs(word, samples=10):
    torch_dtype, _, np_dtype = TYPED[word]
    # Small integers, so that data * gain and its peak are exact in every dtype of the set.
    values = np.resize(np.array([-8, -3, -1, 0, 1, 2, 3, 5, 7, 8], dtype=np.float32), samples)
    data = torch.from_numpy(values.reshape(1, 1, samples).astype(np_dtype))
    gain = torch.tensor([GAIN]).to(torch_dtype)
    return data, gain


def expected(word, samples=10):
    _, _, np_dtype = TYPED[word]
    data, gain = example_inputs(word, samples)
    processed = data.numpy().astype(np.float32) * GAIN
    return processed.astype(np_dtype), np.array([np.max(np.abs(processed))]).astype(np_dtype)


def check_exact(name, got, want):
    got = np.asarray(got)
    assert got.dtype == want.dtype, f"{name}: dtype {got.dtype}, expected {want.dtype}"
    assert got.shape == want.shape, f"{name}: shape {got.shape}, expected {want.shape}"
    assert np.array_equal(got, want), f"{name}: {got} != {want}"
    print(f"{name}: exact")


def export_torchscript(net, data, gain, word):
    filepath = path_of(f"typed_gain_network_{word}.pt")
    torch.jit.trace(net, (data, gain)).save(filepath)
    print(f"Saved TorchScript model to {filepath}")
    return filepath


def export_onnx(net, data, gain, word):
    filepath = path_of(f"typed_gain_network_{word}.onnx")
    # The legacy (TorchScript) exporter, as SimpleGainNetwork: one self-contained opset-11 file.
    torch.onnx.export(
        net,
        (data, gain),
        filepath,
        input_names=["data", "gain"],
        output_names=["processed_data", "peak"],
        dynamic_axes={"data": {2: "dynamic"}, "processed_data": {2: "dynamic"}},
        opset_version=11,
        dynamo=False,
    )
    print(f"Saved ONNX model to {filepath}")
    return filepath


def export_executorch(net, data, gain, word, planned):
    suffix = "" if planned else "_unplanned"
    filepath = path_of(f"typed_gain_network_{word}{suffix}.pte")
    if planned:
        # Dynamic sample dimension (axis 2), as the ONNX and TFLite files.
        sample_dim = torch.export.Dim("samples", min=1, max=1 << 16)
        exported = torch_export(net, (data, gain), dynamic_shapes={"data": {2: sample_dim}, "gain": None})
        program = to_edge(exported).to_executorch()
    else:
        # A static [1, 1, UNPLANNED_SAMPLES]: a caller binds an output buffer of the tensor's
        # capacity (Method::set_output_data_ptr refuses a smaller one), and a dynamic axis would
        # make that capacity its upper bound.
        exported = torch_export(net, (data, gain))
        program = to_edge(exported).to_executorch(
            ExecutorchBackendConfig(
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=False)
            )
        )
    with open(filepath, "wb") as f:
        f.write(program.buffer)
    print(f"Saved ExecuTorch model to {filepath}")
    return filepath


def export_tflite(word):
    _, tf_dtype, _ = TYPED[word]
    filepath = path_of(f"typed_gain_network_{word}.tflite")
    model = TypedGainNetworkTF(tf_dtype)
    data_spec = tf.TensorSpec([1, 1, None], dtype=tf_dtype)
    gain_spec = tf.TensorSpec([1, 1, 1], dtype=tf_dtype)
    concrete_func = tf.function(model).get_concrete_function([data_spec, gain_spec])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
    with open(filepath, "wb") as f:
        f.write(converter.convert())
    print(f"Saved TFLite model to {filepath}")
    return filepath


def run_onnx(path, feeds, names):
    session = ort.InferenceSession(path)
    return session.run(names, feeds)


def run_tflite_gain(path, data, gain):
    interpreter = tf.lite.Interpreter(model_path=path)
    inputs = interpreter.get_input_details()
    # As SimpleGainNetwork: the data input is the one with the dynamic sample axis.
    data_in = next(d for d in inputs if -1 in list(d["shape_signature"]))
    gain_in = next(d for d in inputs if d["index"] != data_in["index"])
    interpreter.resize_tensor_input(data_in["index"], data.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(data_in["index"], data)
    interpreter.set_tensor(gain_in["index"], gain.reshape(1, 1, 1))
    interpreter.invoke()
    outputs = interpreter.get_output_details()
    data_out = next(d for d in outputs if -1 in list(d["shape_signature"]))
    peak_out = next(d for d in outputs if d["index"] != data_out["index"])
    for d in inputs + outputs:
        scale, zero_point = d["quantization"]
        assert scale == 0.0 and zero_point == 0, f"{path}: {d['name']} carries quantisation parameters"
    return interpreter.get_tensor(data_out["index"]), interpreter.get_tensor(peak_out["index"])


def run_executorch(path, data, gain):
    from executorch.runtime import Runtime
    method = Runtime.get().load_program(path).load_method("forward")
    outputs = method.execute([data, gain])
    return outputs[0].numpy(), outputs[1].numpy()


def build_typed(word):
    print(f"\n=== TypedGainNetwork, {word} ===")
    torch_dtype, _, _ = TYPED[word]
    net = TypedGainNetwork(torch_dtype)
    data, gain = example_inputs(word)
    want_data, want_peak = expected(word)

    with torch.no_grad():
        got = net(data, gain)
    check_exact(f"{word} PyTorch processed_data", got[0].numpy(), want_data)
    check_exact(f"{word} PyTorch peak", got[1].numpy(), want_peak)

    pt = export_torchscript(net, data, gain, word)
    loaded = torch.jit.load(pt)
    with torch.no_grad():
        got = loaded(data, gain)
    check_exact(f"{word} TorchScript processed_data", got[0].numpy(), want_data)
    check_exact(f"{word} TorchScript peak", got[1].numpy(), want_peak)

    onnx_path = export_onnx(net, data, gain, word)
    got = run_onnx(onnx_path, {"data": data.numpy(), "gain": gain.numpy()}, ["processed_data", "peak"])
    check_exact(f"{word} ONNX processed_data", got[0], want_data)
    check_exact(f"{word} ONNX peak", got[1], want_peak)

    tflite_path = export_tflite(word)
    got = run_tflite_gain(tflite_path, data.numpy(), gain.numpy())
    check_exact(f"{word} TFLite processed_data", got[0], want_data)
    check_exact(f"{word} TFLite peak", got[1], want_peak)

    if not _HAS_EXECUTORCH:
        print("ExecuTorch not installed; skipping the .pte exports. Install with: pip install executorch")
        return
    pte = export_executorch(net, data, gain, word, planned=True)
    got = run_executorch(pte, data, gain)
    check_exact(f"{word} ExecuTorch processed_data", got[0], want_data)
    check_exact(f"{word} ExecuTorch peak", got[1], want_peak)

    data, gain = example_inputs(word, UNPLANNED_SAMPLES)
    want_data, want_peak = expected(word, UNPLANNED_SAMPLES)
    pte = export_executorch(net, data, gain, word, planned=False)
    got = run_executorch(pte, data, gain)
    check_exact(f"{word} ExecuTorch (unplanned I/O) processed_data", got[0], want_data)
    check_exact(f"{word} ExecuTorch (unplanned I/O) peak", got[1], want_peak)


def bfloat16_bits(values):
    """The bfloat16 bit patterns of float32 values that bfloat16 holds exactly."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    assert np.all(bits & 0xFFFF == 0), "not exact in bfloat16"
    return (bits >> 16).astype(np.uint16)


def build_matrix(word):
    elem_type, np_dtype = MATRIX[word]
    filepath = path_of(f"dtype_matrix_{word}.onnx")
    two = helper.make_tensor("two", TensorProto.FLOAT, [], [2.0])
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["data"], ["as_float"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["as_float", "two"], ["doubled"]),
            helper.make_node("Cast", ["doubled"], ["processed_data"], to=elem_type),
        ],
        f"dtype_matrix_{word}",
        [helper.make_tensor_value_info("data", elem_type, [1, 1, "N"])],
        [helper.make_tensor_value_info("processed_data", elem_type, [1, 1, "N"])],
        initializer=[two],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8,
                              producer_name="anira example-models")
    onnx.checker.check_model(model)
    onnx.save(model, filepath)
    print(f"Saved ONNX dtype matrix model to {filepath}")

    values = np.array([[[0, 1, 2, 3, 5, 7]]], dtype=np.float32)
    if word == "bool":
        data = np.array([[[False, True, False, True, True, False]]])
        want = data.copy()
    elif np_dtype is None:
        # bfloat16 has no numpy dtype: fed and read through an OrtValue of its bytes.
        session = ort.InferenceSession(filepath)
        data = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(bfloat16_bits(values), TensorProto.BFLOAT16)
        out = session.run_with_ort_values(["processed_data"], {"data": data})[0]
        count = int(np.prod(out.shape()))
        got = np.ctypeslib.as_array(ctypes.cast(out.data_ptr(), ctypes.POINTER(ctypes.c_uint16)), (count,))
        check_exact(f"matrix {word}", got.reshape(out.shape()).copy(), bfloat16_bits(values * 2))
        return
    else:
        data = values.astype(np_dtype)
        want = (values * 2).astype(np_dtype)
    got = run_onnx(filepath, {"data": data}, ["processed_data"])[0]
    check_exact(f"matrix {word}", got, want)


def build_quantized():
    print("\n=== quantized_gain_network_int8 (fully integer TFLite) ===")
    filepath = path_of("quantized_gain_network_int8.tflite")

    @tf.function(input_signature=[tf.TensorSpec([1, 1, 64], dtype=tf.float32)])
    def half(x):
        return x * 0.5

    def representative_dataset():
        rng = np.random.default_rng(0)
        for _ in range(64):
            yield [rng.uniform(-1.0, 1.0, size=(1, 1, 64)).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_concrete_functions([half.get_concrete_function()], half)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    with open(filepath, "wb") as f:
        f.write(converter.convert())
    print(f"Saved quantised TFLite model to {filepath}")

    interpreter = tf.lite.Interpreter(model_path=filepath)
    interpreter.allocate_tensors()
    for d in interpreter.get_input_details() + interpreter.get_output_details():
        scale, zero_point = d["quantization"]
        assert d["dtype"] == np.int8, f"{d['name']}: dtype {d['dtype']}"
        assert scale != 0.0, f"{d['name']}: no quantisation parameters"
        print(f"{d['name']}: int8, scale {scale}, zero point {zero_point}")
    # One inference: y = 0.5 x within one quantisation step of the input's scale.
    x = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(1, 1, 64)
    d_in, d_out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    s_in, z_in = d_in["quantization"]
    s_out, z_out = d_out["quantization"]
    q = np.clip(np.round(x / s_in) + z_in, -128, 127).astype(np.int8)
    interpreter.set_tensor(d_in["index"], q)
    interpreter.invoke()
    y = (interpreter.get_tensor(d_out["index"]).astype(np.float32) - z_out) * s_out
    assert np.max(np.abs(y - 0.5 * x)) <= s_in + s_out, "quantised model does not halve its input"
    print("quantised model: y = 0.5 x within one step")


def main():
    for word in TYPED:
        build_typed(word)
    print("\n=== dtype matrix (ONNX) ===")
    for word in MATRIX:
        build_matrix(word)
    build_quantized()
    print("\nAll TypedGainNetwork models exported and checked.")


if __name__ == "__main__":
    main()
