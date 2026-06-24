import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import os
from typing import Tuple

try:
    from executorch.exir import to_edge
    from torch.export import export as torch_export
    _HAS_EXECUTORCH = True
except ImportError:
    _HAS_EXECUTORCH = False

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def channel_suffix(channels):
    """Map a channel count to a file suffix (mono/stereo for 1/2, NchN otherwise)."""
    if channels == 1:
        return "mono"
    if channels == 2:
        return "stereo"
    return f"{channels}ch"


# SimpleGainNetwork (PyTorch)
class SimpleGainNetwork(nn.Module):
    def __init__(self):
        super(SimpleGainNetwork, self).__init__()

    def forward(self, data, gain) -> Tuple[torch.Tensor, torch.Tensor]:
        processed_data = data * gain
        peak = torch.max(torch.abs(processed_data))
        peak = peak.view(1)
        return processed_data, peak


# SimpleGainNetwork (TensorFlow)
class SimpleGainNetworkTF(tf.keras.Model):
    def call(self, inputs):
        data, gain = inputs
        processed_data = data * gain
        peak = tf.reduce_max(tf.abs(processed_data))
        peak = tf.reshape(peak, [1])
        return processed_data, peak


def run_pytorch_inference(net, data, gain):
    with torch.no_grad():
        return net(data, gain)


def export_torchscript_model(net, data, gain, channels):
    suffix = channel_suffix(channels)
    filepath = os.path.join(MODEL_DIR, f"simple_gain_network_{suffix}.pt")
    scripted_net = torch.jit.trace(net, (data, gain))
    scripted_net.save(filepath)
    print(f"Saved TorchScript model to {filepath}")
    return filepath


def export_onnx_model(net, data, gain, channels):
    suffix = channel_suffix(channels)
    filepath = os.path.join(MODEL_DIR, f"simple_gain_network_{suffix}.onnx")
    # Use the legacy (TorchScript) exporter so we get a single self-contained
    # opset-11 .onnx file (the dynamo exporter forces opset 18 and emits an
    # external-data sidecar).
    torch.onnx.export(
        net,
        (data, gain),
        filepath,
        input_names=["data", "gain"],
        output_names=["processed_data", "peak"],
        dynamic_axes={"data": {2: "dynamic"}},
        opset_version=11,
        dynamo=False
    )
    print(f"Saved ONNX model to {filepath}")
    return filepath


def export_executorch_model(net, data, gain, channels):
    suffix = channel_suffix(channels)
    filepath = os.path.join(MODEL_DIR, f"simple_gain_network_{suffix}.pte")

    # Dynamic sample dimension (axis 2) so the .pte matches the ONNX/TFLite models.
    sample_dim = torch.export.Dim("samples", min=1, max=1 << 16)
    dynamic_shapes = {"data": {2: sample_dim}, "gain": None}

    exported = torch_export(net, (data, gain), dynamic_shapes=dynamic_shapes)
    edge_program = to_edge(exported)
    executorch_program = edge_program.to_executorch()

    with open(filepath, "wb") as f:
        f.write(executorch_program.buffer)
    print(f"Saved ExecuTorch model to {filepath}")
    return filepath


def convert_tf_to_tflite(model, channels):
    suffix = channel_suffix(channels)
    tflite_model_path = os.path.join(MODEL_DIR, f"simple_gain_network_{suffix}.tflite")

    data_shape = tf.TensorSpec([1, channels, None], dtype=tf.float32)
    gain_shape = tf.TensorSpec([1, 1, 1], dtype=tf.float32)

    concrete_func = tf.function(model).get_concrete_function([data_shape, gain_shape])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"Model converted to TFLite and saved as {tflite_model_path}")
    return tflite_model_path


def run_onnx_inference(onnx_model_path, data, gain):
    session = ort.InferenceSession(onnx_model_path)
    data_numpy, gain_numpy = data.numpy(), gain.numpy()
    outputs = session.run(["processed_data", "peak"], {"data": data_numpy, "gain": gain_numpy})
    return outputs[0], outputs[1]


def run_tflite_inference(tflite_model_path, data, gain):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # TFLite does not guarantee that input/output ordering matches the order the
    # tensors were declared, so identify each tensor by its shape rather than by
    # position. Only the data input has a dynamic (-1) sample axis; the gain
    # input is the other. The peak output is the single-element tensor.
    data_in = next(d for d in input_details if -1 in list(d['shape_signature']))
    gain_in = next(d for d in input_details if d['index'] != data_in['index'])

    interpreter.resize_tensor_input(data_in['index'], data.shape)
    interpreter.resize_tensor_input(gain_in['index'], gain.shape)
    interpreter.allocate_tensors()  # Re-allocate tensors after resizing

    interpreter.set_tensor(data_in['index'], data.numpy())
    interpreter.set_tensor(gain_in['index'], gain.numpy())

    interpreter.invoke()

    # Re-read details after allocation. processed_data carries the dynamic sample
    # axis (-1); the peak is the static single-element output.
    output_details = interpreter.get_output_details()
    data_out = next(d for d in output_details if -1 in list(d['shape_signature']))
    peak_out = next(d for d in output_details if d['index'] != data_out['index'])

    processed_data = interpreter.get_tensor(data_out['index'])
    peak_value = interpreter.get_tensor(peak_out['index'])
    return processed_data, peak_value


def run_executorch_inference(pte_model_path, data, gain):
    from executorch.runtime import Runtime
    runtime = Runtime.get()
    program = runtime.load_program(pte_model_path)
    method = program.load_method("forward")
    outputs = method.execute([data, gain])
    return outputs[0].numpy(), outputs[1].numpy()


def check_consistency(output1, output2, tolerance=1e-6, name=""):
    assert np.allclose(output1, output2, atol=tolerance), f"Mismatch in {name} outputs"
    print(f"{name} outputs are consistent.")


def build_models_for_channels(channels):
    print(f"\n=== Building SimpleGainNetwork for {channels} channel(s) ({channel_suffix(channels)}) ===")

    # Create PyTorch model and convert to LibTorch and OnnxRuntime
    net = SimpleGainNetwork()
    data = torch.randn(1, channels, 10)
    gain = torch.tensor([1.5])
    torchscript_model_path = export_torchscript_model(net, data, gain, channels)
    onnx_model_path = export_onnx_model(net, data, gain, channels)

    # Export ExecuTorch (.pte) for future on-device runtime support
    pte_model_path = None
    if _HAS_EXECUTORCH:
        pte_model_path = export_executorch_model(net, data, gain, channels)
    else:
        print("ExecuTorch not installed; skipping .pte export. Install with: pip install executorch")

    # Create TensorFlow model and convert to TFLite
    tf_model = SimpleGainNetworkTF()
    tf_model((tf.constant(data.numpy()), tf.constant(gain.numpy())))
    tflite_model_path = convert_tf_to_tflite(tf_model, channels)

    # Run inferences
    output_data_pytorch, output_peak_pytorch = run_pytorch_inference(net, data, gain)
    processed_data_onnx, peak_value_onnx = run_onnx_inference(onnx_model_path, data, gain)
    processed_data_tflite, peak_value_tflite = run_tflite_inference(tflite_model_path, data, gain)

    # Check consistency
    check_consistency(output_data_pytorch.numpy(), processed_data_onnx, name="PyTorch vs ONNX (Processed Data)")
    check_consistency(output_peak_pytorch.numpy(), peak_value_onnx, name="PyTorch vs ONNX (Peak)")
    check_consistency(processed_data_onnx, processed_data_tflite, name="ONNX vs TFLite (Processed Data)")
    check_consistency(peak_value_onnx, peak_value_tflite, name="ONNX vs TFLite (Peak)")

    if pte_model_path is not None:
        processed_data_pte, peak_value_pte = run_executorch_inference(pte_model_path, data, gain)
        check_consistency(output_data_pytorch.numpy(), processed_data_pte, name="PyTorch vs ExecuTorch (Processed Data)")
        check_consistency(output_peak_pytorch.numpy(), peak_value_pte, name="PyTorch vs ExecuTorch (Peak)")

    print(f"All tests passed for {channels} channel(s)! "
          f"PyTorch, TorchScript, ONNX, TFLite{', and ExecuTorch' if pte_model_path else ''} outputs are consistent.")


def main(channel_counts=(1, 2, 4, 16)):
    for channels in channel_counts:
        build_models_for_channels(channels)


if __name__ == "__main__":
    main()
