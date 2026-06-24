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

To (re)generate the models:

```bash
pip install -r requirements.txt
python SimpleGainNetwork/simple_gain_network.py
```

By default this builds the `(1, 2, 4, 16)` channel set; edit the `channel_counts` argument to `main()` to change it.

