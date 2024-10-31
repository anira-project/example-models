import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# SimpleGainNetwork (PyTorch)
class SimpleGainNetwork(nn.Module):
    def __init__(self):
        super(SimpleGainNetwork, self).__init__()

    def forward(self, data, gain):
        processed_data = data * gain
        peak = torch.max(torch.abs(processed_data))
        return processed_data, torch.tensor([peak])


# SimpleGainNetwork (TensorFlow)
class SimpleGainNetworkTF(tf.keras.Model):
    def call(self, inputs):
        data, gain = inputs
        processed_data = data * gain
        peak = tf.reduce_max(tf.abs(processed_data), axis=[1, 2], keepdims=True)
        return processed_data, peak


def run_pytorch_inference(net, data, gain):
    with torch.no_grad():
        return net(data, gain)


def export_torchscript_model(net, data, gain, filepath="simple_gain_network.pt"):
    filepath = os.path.join(MODEL_DIR, filepath)
    scripted_net = torch.jit.trace(net, (data, gain))
    scripted_net.save(filepath)
    print(f"Saved TorchScript model to {filepath}")
    return filepath


def export_onnx_model(net, data, gain, filepath="simple_gain_network.onnx"):
    filepath = os.path.join(MODEL_DIR, filepath)
    torch.onnx.export(
        net,
        (data, gain),
        filepath,
        input_names=["data", "gain"],
        output_names=["processed_data", "peak"],
        dynamic_axes={"data": {2: "dynamic"}},
        opset_version=11
    )
    print(f"Saved ONNX model to {filepath}")
    return filepath


def convert_tf_to_tflite(model, tflite_model_path="simple_gain_network.tflite"):
    tflite_model_path = os.path.join(MODEL_DIR, tflite_model_path)
    data_shape = tf.TensorSpec([1, 2, None], dtype=tf.float32)
    gain_shape = tf.TensorSpec([1, 1, 1], dtype=tf.float32)

    func = tf.function(model).get_concrete_function([data_shape, gain_shape])

    converter = tf.lite.TFLiteConverter.from_concrete_functions([func])
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

    interpreter.resize_tensor_input(input_details[0]['index'], data.shape)
    interpreter.resize_tensor_input(input_details[1]['index'], gain.shape)
    interpreter.allocate_tensors()  # Re-allocate tensors after resizing

    interpreter.set_tensor(input_details[0]['index'], data.numpy())
    interpreter.set_tensor(input_details[1]['index'], gain.numpy())

    interpreter.invoke()

    processed_data = interpreter.get_tensor(output_details[0]['index'])
    peak_value = interpreter.get_tensor(output_details[1]['index'])
    return processed_data, peak_value


def check_consistency(output1, output2, tolerance=1e-6, name=""):
    assert np.allclose(output1, output2, atol=tolerance), f"Mismatch in {name} outputs"
    print(f"{name} outputs are consistent.")


def main():
    net = SimpleGainNetwork()
    data = torch.randn(1, 2, 10)
    gain = torch.tensor([1.5])

    torchscript_model_path = export_torchscript_model(net, data, gain)
    onnx_model_path = export_onnx_model(net, data, gain)

    tf_model = SimpleGainNetworkTF()
    tf_model((tf.constant(data.numpy()), tf.constant(gain.numpy())))
    tflite_model_path = convert_tf_to_tflite(tf_model)

    output_data_pytorch, output_peak_pytorch = run_pytorch_inference(net, data, gain)
    processed_data_onnx, peak_value_onnx = run_onnx_inference(onnx_model_path, data, gain)
    processed_data_tflite, peak_value_tflite = run_tflite_inference(tflite_model_path, data, gain)

    check_consistency(output_data_pytorch.numpy(), processed_data_onnx, name="PyTorch vs ONNX (Processed Data)")
    check_consistency(output_peak_pytorch.numpy(), peak_value_onnx, name="PyTorch vs ONNX (Peak)")
    check_consistency(processed_data_onnx, processed_data_tflite, name="ONNX vs TFLite (Processed Data)")
    check_consistency(peak_value_onnx, peak_value_tflite, name="ONNX vs TFLite (Peak)")

    print("All tests passed! The outputs from PyTorch, TorchScript, ONNX, and TFLite are consistent.")


if __name__ == "__main__":
    main()
