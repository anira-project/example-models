"""Rebuild the Djembe RAVE v1 (streaming) model in eager PyTorch with EXPLICIT state.

The original ``Djembe_deterministic.ts`` is a TorchScript export of RAVE v2.3.1
(`scripts/export.py --streaming`), variational encoder, v1 architecture:

    pqmf   : CachedPQMF(attenuation=100, n_band=16)
    encoder: VariationalEncoder(Encoder(data_size=16, capacity=32, latent_size=16,
                                        ratios=[4,4,4,2], n_out=2))
    decoder: Generator(latent_size=16, capacity=32, data_size=16,
                       ratios=[4,4,4,2], loud_stride=1, use_noise=True)
    latent truncated to 4 dims via PCA (deterministic: mean latent, zero-fill).

Streaming state lives in two primitive module types from ``cached_conv``:
  * CachedPadding1d      -- left-context of every causal conv / delay line
  * CachedConvTranspose1d -- overlap-add tail of the decoder upsamplers

This module rebuilds the model from the repo source, loads the weights from the
.ts file, and re-implements those primitives so that ALL state is passed in and
returned as one flat float32 tensor -> stateless inference (ONNX / TFLite /
ExecuTorch friendly).  The decoder's noise generator is made deterministic by
taking the uniform noise as an explicit input, and its FFT convolution is
replaced by exact DFT matmuls (portable to runtimes without FFT ops).
"""

import functools
import importlib
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)

HERE = Path(__file__).resolve().parent
REPO = HERE / "rave"  # git clone https://github.com/acids-ircam/RAVE rave
TS_PATH = HERE / "Djembe_deterministic.ts"

# Architecture constants (derived from the .ts state dict, see README)
N_BAND = 16
CAPACITY = 32
FULL_LATENT = 16
RATIOS = [4, 4, 4, 2]
NOISE_RATIOS = [4, 4, 4]
NOISE_BANDS = 5
RATIO = 2048  # total samples per latent frame (16 * 4*4*4*2)

_cc = None
_blocks = None
_pqmf_mod = None


def _import_rave_modules():
    """Import rave.blocks / rave.pqmf from the repo without its heavy deps."""
    global _cc, _blocks, _pqmf_mod
    if _blocks is not None:
        return _cc, _blocks, _pqmf_mod

    import cached_conv as cc
    cc.use_cached_conv(True)

    # v1.gin: cc.Conv1d.bias = False / cc.ConvTranspose1d.bias = False
    _Conv, _ConvT = cc.Conv1d, cc.ConvTranspose1d

    def Conv1d(*a, **k):
        k.setdefault("bias", False)
        return _Conv(*a, **k)

    def ConvTranspose1d(*a, **k):
        k.setdefault("bias", False)
        return _ConvT(*a, **k)

    cc.Conv1d = Conv1d
    cc.ConvTranspose1d = ConvTranspose1d

    # Stub the `rave` package so we can import blocks/pqmf without
    # rave/__init__.py and rave/core.py (which need lightning, librosa, lmdb...)
    pkg = types.ModuleType("rave")
    pkg.__path__ = [str(REPO / "rave")]
    sys.modules["rave"] = pkg

    core = types.ModuleType("rave.core")
    core.mod_sigmoid = lambda x: 2 * torch.sigmoid(x) ** 2.3 + 1e-7

    import torch.fft as fft

    def amp_to_impulse_response(amp, target_size):
        amp = torch.stack([amp, torch.zeros_like(amp)], -1)
        amp = torch.view_as_complex(amp)
        amp = fft.irfft(amp)
        filter_size = amp.shape[-1]
        amp = torch.roll(amp, filter_size // 2, -1)
        win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)
        amp = amp * win
        amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
        amp = torch.roll(amp, -filter_size // 2, -1)
        return amp

    def fft_convolve(signal, kernel):
        signal = nn.functional.pad(signal, (0, signal.shape[-1]))
        kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))
        output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
        output = output[..., output.shape[-1] // 2:]
        return output

    core.amp_to_impulse_response = amp_to_impulse_response
    core.fft_convolve = fft_convolve
    sys.modules["rave.core"] = core

    # blocks.py imports torchaudio only for Spectrogram (unused by this model);
    # stub it if torchaudio is missing or ABI-broken in this venv.
    try:
        import torchaudio  # noqa: F401
    except Exception:
        ta = types.ModuleType("torchaudio")
        ta_t = types.ModuleType("torchaudio.transforms")
        ta_t.Spectrogram = object
        ta.transforms = ta_t
        sys.modules["torchaudio"] = ta
        sys.modules["torchaudio.transforms"] = ta_t

    import scipy.signal
    if not hasattr(scipy.signal, "kaiser"):  # removed in modern scipy
        from scipy.signal.windows import kaiser as _kaiser
        scipy.signal.kaiser = _kaiser

    blocks = importlib.import_module("rave.blocks")
    pqmf_mod = importlib.import_module("rave.pqmf")

    # The scipy-based filter design is broken on modern scipy AND pointless
    # here: the real filters are loaded from the .ts export afterwards
    # (_fix_pqmf_filters). Return a dummy prototype of the right length.
    pqmf_mod.get_prototype = lambda atten, M, N=None: np.zeros(377)

    # v1.gin bindings normally supplied by gin
    blocks.ResidualStack = functools.partial(
        blocks.ResidualStack, kernel_sizes=[3],
        dilations_list=[[1, 1], [3, 1], [5, 1]])
    blocks.NoiseGenerator = functools.partial(
        blocks.NoiseGenerator, ratios=NOISE_RATIOS, noise_bands=NOISE_BANDS)

    _cc, _blocks, _pqmf_mod = cc, blocks, pqmf_mod
    return cc, blocks, pqmf_mod


class EagerRAVE(nn.Module):
    """Container matching the .ts state-dict layout (pqmf / encoder / decoder)."""

    def __init__(self):
        super().__init__()
        cc, blocks, pqmf_mod = _import_rave_modules()
        self.pqmf = pqmf_mod.CachedPQMF(attenuation=100, n_band=N_BAND)
        enc = functools.partial(
            blocks.Encoder, data_size=N_BAND, capacity=CAPACITY,
            latent_size=FULL_LATENT, ratios=RATIOS, n_out=2,
            sample_norm=False, repeat_layers=1)
        self.encoder = blocks.VariationalEncoder(enc)
        self.decoder = blocks.Generator(
            latent_size=FULL_LATENT, capacity=CAPACITY, data_size=N_BAND,
            ratios=RATIOS, loud_stride=1, use_noise=True)
        self.register_buffer("latent_pca", torch.eye(FULL_LATENT))
        self.register_buffer("latent_mean", torch.zeros(FULL_LATENT))
        self.register_buffer("fidelity", torch.zeros(FULL_LATENT))


def _fix_pqmf_filters(pqmf, sd, cc):
    """Rebuild PQMF filters/convs from the .ts weights (independent of scipy)."""
    hk = sd["pqmf.hk"].clone()
    h = sd["pqmf.h"].clone()
    pqmf.register_buffer("hk", hk)
    pqmf.register_buffer("h", h)

    from rave.pqmf import make_odd
    from einops import rearrange

    hkf = make_odd(hk).unsqueeze(1)                       # (16, 1, 513)
    hki = hk.flip(-1)
    hki = rearrange(hki, "c (t m) -> m c t", m=hk.shape[0])
    hki = make_odd(hki)                                   # (16, 16, 33)

    pqmf.forward_conv = cc.Conv1d(
        hkf.shape[1], hkf.shape[0], hkf.shape[2],
        padding=cc.get_padding(hkf.shape[-1]),
        stride=hkf.shape[0], bias=False)
    pqmf.forward_conv.weight.data.copy_(hkf)

    pqmf.inverse_conv = cc.Conv1d(
        hki.shape[1], hki.shape[0], hki.shape[-1],
        padding=cc.get_padding(hki.shape[-1]), bias=False)
    pqmf.inverse_conv.weight.data.copy_(hki)


def build_eager_model(ts_path=TS_PATH):
    """Build the eager model and load weights from the TorchScript export.

    Returns (model, meta) where meta carries latent sizes etc.
    """
    cc, blocks, pqmf_mod = _import_rave_modules()

    ts = torch.jit.load(str(ts_path), map_location="cpu")
    sd = ts.state_dict()

    model = EagerRAVE()
    model.eval()

    _fix_pqmf_filters(model.pqmf, sd, cc)

    # Warmup pass with the ORIGINAL cached-conv forwards so every lazy cache
    # buffer (CachedPadding1d.pad / CachedConvTranspose1d.cache) is created.
    x = torch.zeros(1, 1, 4 * RATIO)
    mb = model.pqmf(x)
    z = model.encoder(mb)
    y = model.decoder(torch.zeros(1, FULL_LATENT, 4))
    model.pqmf.inverse(y)

    keep = {}
    for k, v in sd.items():
        head = k.split(".")[0]
        if head in ("pqmf", "encoder", "decoder") or k in (
                "latent_pca", "latent_mean", "fidelity"):
            keep[k] = v
    missing, unexpected = model.load_state_dict(keep, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert not missing, f"missing keys: {missing}"

    # `warmed_up` is a tensor buffer used as `if self.warmed_up:` — fine under
    # jit tracing (constant) but a data-dependent guard for torch.export.
    # Replace with a plain python bool after loading.
    for mod in (model.encoder, model.decoder):
        mod.warmed_up = bool(mod._buffers.pop("warmed_up").item())

    meta = {
        "sr": int(ts.sr),
        "latent_size": int(ts.latent_size),           # 4 (exported dims)
        "full_latent_size": int(ts.full_latent_size), # 16
        "ratio": RATIO,
        "n_band": N_BAND,
    }
    return model, meta


# ---------------------------------------------------------------------------
# Explicit-state re-implementations of the cached_conv primitives
# ---------------------------------------------------------------------------

def _padding_forward(self, x):
    if self.padding == 0:
        return x
    y = torch.cat([self._state_in, x], -1)
    n = y.shape[-1]
    self._state_out = y[..., n - self.padding:]
    if self.crop:
        y = y[..., : n - self.padding]
    return y


def _convT_forward(self, x):
    y = F.conv_transpose1d(
        x, self.weight, None, self.stride, 0,
        self.output_padding, self.groups, self.dilation)
    p = 2 * self.padding[0]
    head = y[..., :p] + self._state_in
    y = torch.cat([head, y[..., p:]], -1)
    n = y.shape[-1]
    self._state_out = y[..., n - p:]
    y = y[..., : n - p]
    if self.bias is not None:
        y = y + self.bias.unsqueeze(-1)
    return y


def _make_dft_matrices(bands, target):
    """Exact real-DFT matrices (float64 -> float32) for the noise generator."""
    fs = 2 * (bands - 1)          # irfft length of the amplitude spectrum (8)
    N = 2 * target                # fft_convolve length (128)
    Nh = N // 2 + 1

    k = np.arange(bands)
    t = np.arange(fs)
    w = np.ones(bands); w[1:-1] = 2.0
    irfft_small = (w[:, None] * np.cos(2 * np.pi * np.outer(k, t) / fs)) / fs

    tN = np.arange(N)
    kN = np.arange(Nh)
    ang = 2 * np.pi * np.outer(tN, kN) / N
    rfft_C = np.cos(ang)          # [N, Nh]  real part
    rfft_S = -np.sin(ang)         # [N, Nh]  imag part
    wN = np.ones(Nh); wN[1:-1] = 2.0
    irfft_C = (wN[:, None] * np.cos(2 * np.pi * np.outer(kN, tN) / N)) / N
    irfft_S = (-wN[:, None] * np.sin(2 * np.pi * np.outer(kN, tN) / N)) / N

    f32 = lambda a: torch.from_numpy(a).to(torch.float32)
    return (f32(irfft_small), f32(rfft_C), f32(rfft_S),
            f32(irfft_C), f32(irfft_S))


def _noise_forward(self, x):
    """NoiseGenerator.forward with external noise + matmul DFTs.

    self._noise_in: uniform noise in [-1, 1], shape [1, T, data_size, target]
    (same shape/order that `torch.rand_like(ir)` consumed in the original).
    """
    from rave.core import mod_sigmoid
    amp = mod_sigmoid(self.net(x) - 5)
    amp = amp.permute(0, 2, 1)
    amp = amp.reshape(amp.shape[0], amp.shape[1], self.data_size, -1)

    fs = 2 * (amp.shape[-1] - 1)
    target = self._target_int  # python int (torch.export-safe)

    # amp_to_impulse_response (irfft of a real spectrum -> matmul)
    ir = amp @ self._irfft_small
    ir = torch.roll(ir, fs // 2, -1) * self._hann
    ir = F.pad(ir, (0, target - fs))
    ir = torch.roll(ir, -(fs // 2), -1)

    noise = self._noise_in
    # fft_convolve(noise, ir) -> matmul DFT
    sig = F.pad(noise, (0, target))
    ker = F.pad(ir, (target, 0))
    sr = sig @ self._rfft_C
    si = sig @ self._rfft_S
    kr = ker @ self._rfft_C
    ki = ker @ self._rfft_S
    outr = sr * kr - si * ki
    outi = sr * ki + si * kr
    out = outr @ self._irfft_C + outi @ self._irfft_S
    out = out[..., target:]

    out = out.permute(0, 2, 1, 3)
    out = out.reshape(out.shape[0], out.shape[1], -1)
    return out


def _pqmf_forward(self, x):
    x = self.forward_conv(x)
    return x * self._fwd_mask


def _pqmf_inverse(self, x):
    x = x * self._inv_mask
    m = self.hk.shape[0]
    x = self.inverse_conv(x) * m
    x = torch.index_select(x, 1, self._rev_idx)   # flip(1), ONNX-friendly
    x = x.permute(0, 2, 1)
    x = x.reshape(x.shape[0], x.shape[1], -1, m).permute(0, 2, 1, 3)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    return x


def _reverse_half_mask(channels, length):
    mask = torch.ones(1, channels, length)
    mask[..., 1::2, ::2] = -1
    return mask


_PATCHED = False


def patch_stateful():
    """Replace cached_conv / NoiseGenerator / CachedPQMF forwards globally."""
    global _PATCHED
    if _PATCHED:
        return
    cc, blocks, pqmf_mod = _import_rave_modules()
    from cached_conv import convs
    convs.CachedPadding1d.forward = _padding_forward
    convs.CachedConvTranspose1d.forward = _convT_forward
    # NoiseGenerator is wrapped in functools.partial; patch the real class
    ng = blocks.NoiseGenerator.func if isinstance(
        blocks.NoiseGenerator, functools.partial) else blocks.NoiseGenerator
    ng.forward = _noise_forward
    pqmf_mod.CachedPQMF.forward = _pqmf_forward
    pqmf_mod.CachedPQMF.inverse = _pqmf_inverse
    _PATCHED = True


# ---------------------------------------------------------------------------
# State registry: maps one flat state vector <-> all stateful submodules
# ---------------------------------------------------------------------------

class StateRegistry:
    def __init__(self, roots):
        """roots: list of (prefix, module) whose subtrees are traversed."""
        from cached_conv import convs
        self.entries = []  # (name, module, channels, length)
        for prefix, root in roots:
            for name, m in root.named_modules():
                full = f"{prefix}.{name}" if name else prefix
                if isinstance(m, convs.CachedPadding1d) and m.padding > 0:
                    self.entries.append((full, m, m.pad.shape[1], int(m.padding)))
                elif isinstance(m, convs.CachedConvTranspose1d):
                    L = 2 * m.padding[0]
                    self.entries.append((full, m, m.cache.shape[1], L))
        self.size = sum(c * l for _, _, c, l in self.entries)

    def scatter(self, state):
        offset = 0
        for _, m, c, l in self.entries:
            n = c * l
            m._state_in = state[:, offset:offset + n].reshape(1, c, l)
            offset += n

    def gather(self):
        return torch.cat(
            [m._state_out.reshape(1, -1) for _, m, _, _ in self.entries], -1)

    def spec(self):
        return [{"name": n, "channels": c, "length": l}
                for n, _, c, l in self.entries]


# ---------------------------------------------------------------------------
# Stateless (explicit-state) wrapper models
# ---------------------------------------------------------------------------

class StatefulEncoder(nn.Module):
    """audio [1,1,block], state [1,S] -> latent [1,4,block/2048], new_state."""

    def __init__(self, model, meta, block_size):
        super().__init__()
        patch_stateful()
        assert block_size % RATIO == 0
        self.block_size = block_size
        self.latent_size = meta["latent_size"]
        self.pqmf = model.pqmf
        self.encoder = model.encoder
        self.register_buffer("latent_pca", model.latent_pca.clone())
        self.register_buffer("latent_mean", model.latent_mean.clone())
        self.pqmf._fwd_mask = _reverse_half_mask(N_BAND, block_size // N_BAND)
        self.registry = StateRegistry([
            ("pqmf.forward_conv", model.pqmf.forward_conv),
            ("encoder", model.encoder),
        ])

    def forward(self, audio, state):
        self.registry.scatter(state)
        x = self.pqmf(audio)
        z = self.encoder(x)
        mean, _ = torch.chunk(z, 2, 1)
        z = mean - self.latent_mean.unsqueeze(-1)
        z = F.conv1d(z, self.latent_pca.unsqueeze(-1))
        z = z[:, : self.latent_size]
        return z, self.registry.gather()


class StatefulDecoder(nn.Module):
    """latent [1,4,n], state [1,S], noise [1,n*2,16,64] -> audio, new_state."""

    def __init__(self, model, meta, block_size):
        super().__init__()
        patch_stateful()
        assert block_size % RATIO == 0
        self.block_size = block_size
        self.n_frames = block_size // RATIO
        self.latent_size = meta["latent_size"]
        self.full_latent_size = meta["full_latent_size"]
        self.decoder = model.decoder
        self.pqmf = model.pqmf
        self.register_buffer("latent_pca", model.latent_pca.clone())
        self.register_buffer("latent_mean", model.latent_mean.clone())
        self.register_buffer(
            "zero_fill",
            torch.zeros(1, self.full_latent_size - self.latent_size,
                        self.n_frames))
        self.pqmf._inv_mask = _reverse_half_mask(N_BAND, block_size // N_BAND)
        self.pqmf._rev_idx = torch.arange(N_BAND - 1, -1, -1)
        self._setup_noise()
        self.registry = StateRegistry([
            ("decoder", model.decoder),
            ("pqmf.inverse_conv", model.pqmf.inverse_conv),
        ])

    def _setup_noise(self):
        ng = self.decoder.synth.branches[2]
        target = int(ng.target_size)
        ng._target_int = target
        (ng._irfft_small, ng._rfft_C, ng._rfft_S,
         ng._irfft_C, ng._irfft_S) = _make_dft_matrices(NOISE_BANDS, target)
        fs = 2 * (NOISE_BANDS - 1)
        ng._hann = torch.hann_window(fs, dtype=torch.float32)
        self.noise_gen = ng
        # noise input shape for one block
        self.noise_frames = self.block_size // N_BAND // target
        self.noise_shape = (1, self.noise_frames, N_BAND, target)

    def forward(self, latent, state, noise):
        self.registry.scatter(state)
        self.noise_gen._noise_in = noise
        z = torch.cat([latent, self.zero_fill], 1)
        z = F.conv1d(z, self.latent_pca.t().unsqueeze(-1))
        z = z + self.latent_mean.unsqueeze(-1)
        y = self.decoder(z)
        audio = self.pqmf.inverse(y)
        return audio, self.registry.gather()


class StatefulForward(nn.Module):
    """Full forward: audio, state, noise -> audio_out, new_state."""

    def __init__(self, model, meta, block_size):
        super().__init__()
        self.enc = StatefulEncoder(model, meta, block_size)
        self.dec = StatefulDecoder(model, meta, block_size)
        self.noise_shape = self.dec.noise_shape
        self.registry = StateRegistry([
            ("pqmf.forward_conv", model.pqmf.forward_conv),
            ("encoder", model.encoder),
            ("decoder", model.decoder),
            ("pqmf.inverse_conv", model.pqmf.inverse_conv),
        ])

    def forward(self, audio, state, noise):
        self.registry.scatter(state)
        self.dec.noise_gen._noise_in = noise
        x = self.enc.pqmf(audio)
        z = self.enc.encoder(x)
        mean, _ = torch.chunk(z, 2, 1)
        z = mean - self.enc.latent_mean.unsqueeze(-1)
        z = F.conv1d(z, self.enc.latent_pca.unsqueeze(-1))
        z = z[:, : self.enc.latent_size]

        z = torch.cat([z, self.dec.zero_fill], 1)
        z = F.conv1d(z, self.dec.latent_pca.t().unsqueeze(-1))
        z = z + self.dec.latent_mean.unsqueeze(-1)
        y = self.dec.decoder(z)
        audio_out = self.dec.pqmf.inverse(y)
        return audio_out, self.registry.gather()
