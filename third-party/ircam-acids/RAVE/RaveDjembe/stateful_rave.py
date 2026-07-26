"""Rebuild the Djembe-2 RAVE v1 (streaming) model in eager PyTorch with EXPLICIT state.

The original ``Djembe_streaming.ts`` (WeTransfer drop djembe_2_80d99cfa85) is a TorchScript export of
RAVE v2.3.1 (`scripts/export.py --streaming`), variational encoder, v1
architecture, low-latency configuration:

    pqmf   : CachedPQMF(n_band=16, 129-tap prototype)
    encoder: VariationalEncoder(Encoder(data_size=16, capacity=64, latent_size=16,
                                        ratios=[2,2,2,1], n_out=2))
    decoder: Generator(latent_size=16, capacity=64, data_size=16,
                       ratios=[2,2,2,1], loud_stride=1, use_noise=False)
    latent truncated to 2 dims via PCA (deterministic: mean latent, zero-fill).
    One latent frame per 128 samples (16x lower latency than the original
    Djembe's 2048).

Streaming state lives in two primitive module types from ``cached_conv``:
  * CachedPadding1d      -- left-context of every causal conv / delay line
  * CachedConvTranspose1d -- overlap-add tail of the decoder upsamplers

This module rebuilds the model from the repo source, loads the weights from the
.ts file, and re-implements those primitives so that ALL state is passed in and
returned as one flat float32 tensor -> stateless inference (ONNX / TFLite /
ExecuTorch friendly).  This model has no noise generator (use_noise=False).
The stock export is stochastic in two places — the variational encoder samples
``mean + eps * std`` and the decoder fills the truncated latent dims with
``randn`` (the VAE prior).  The stateless wrappers make both deterministic:
the encoder emits the mean, and the decoder takes the prior sample as an
explicit ``fill`` input (zeros = deterministic mean-of-prior; random in
N(0, 1) reproduces the stock behavior).
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
TS_PATH = HERE / "Djembe_streaming.ts"

# Architecture constants (derived from the .ts module tree, see README)
N_BAND = 16
CAPACITY = 64
FULL_LATENT = 16
RATIOS = [2, 2, 2, 1]
RATIO = 128  # total samples per latent frame (16 * 2*2*2*1)

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

    # This model was exported CAUSAL (gin: cached_conv.get_padding.mode =
    # "causal"): every conv pads left-only, so all branch-alignment and
    # downsampling delays are zero — visible in the .ts as 0-length cache
    # buffers. Bind the same mode or the rebuilt caches disagree in shape.
    _get_padding = cc.get_padding
    cc.get_padding = lambda *a, **k: _get_padding(*a, **{**k, "mode": "causal"})

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

    # gin bindings normally supplied by gin; dilations read from the .ts
    # module tree (CachedConv1d.dilation attributes).
    blocks.ResidualStack = functools.partial(
        blocks.ResidualStack, kernel_sizes=[3],
        dilations_list=[[3, 1], [9, 1], [27, 1], [36, 1]])

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
            ratios=RATIOS, loud_stride=1, use_noise=False)
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
        "latent_size": int(ts.latent_size),           # 2 (exported dims)
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
    """audio [1,1,block], state [1,S] -> latent [1,2,block/128], new_state.

    Deterministic: emits the variational mean (no sampling)."""

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
    """latent [1,2,n], state [1,S], fill [1,14,n] -> audio [1,1,n*128], new_state.

    ``fill`` is the prior sample for the PCA-truncated latent dimensions:
    zeros give the deterministic mean-of-prior decode; N(0, 1) noise
    reproduces the stock TorchScript behavior."""

    def __init__(self, model, meta, block_size):
        super().__init__()
        patch_stateful()
        assert block_size % RATIO == 0
        self.block_size = block_size
        self.n_frames = block_size // RATIO
        self.latent_size = meta["latent_size"]
        self.full_latent_size = meta["full_latent_size"]
        self.fill_shape = (1, self.full_latent_size - self.latent_size,
                           self.n_frames)
        self.decoder = model.decoder
        self.pqmf = model.pqmf
        self.register_buffer("latent_pca", model.latent_pca.clone())
        self.register_buffer("latent_mean", model.latent_mean.clone())
        self.pqmf._inv_mask = _reverse_half_mask(N_BAND, block_size // N_BAND)
        self.pqmf._rev_idx = torch.arange(N_BAND - 1, -1, -1)
        self.registry = StateRegistry([
            ("decoder", model.decoder),
            ("pqmf.inverse_conv", model.pqmf.inverse_conv),
        ])

    def forward(self, latent, state, fill):
        self.registry.scatter(state)
        z = torch.cat([latent, fill], 1)
        z = F.conv1d(z, self.latent_pca.t().unsqueeze(-1))
        z = z + self.latent_mean.unsqueeze(-1)
        y = self.decoder(z)
        audio = self.pqmf.inverse(y)
        return audio, self.registry.gather()


class StatefulForward(nn.Module):
    """Full forward: audio, state, fill -> audio_out, new_state."""

    def __init__(self, model, meta, block_size):
        super().__init__()
        self.enc = StatefulEncoder(model, meta, block_size)
        self.dec = StatefulDecoder(model, meta, block_size)
        self.fill_shape = self.dec.fill_shape
        self.registry = StateRegistry([
            ("pqmf.forward_conv", model.pqmf.forward_conv),
            ("encoder", model.encoder),
            ("decoder", model.decoder),
            ("pqmf.inverse_conv", model.pqmf.inverse_conv),
        ])

    def forward(self, audio, state, fill):
        self.registry.scatter(state)
        x = self.enc.pqmf(audio)
        z = self.enc.encoder(x)
        mean, _ = torch.chunk(z, 2, 1)
        z = mean - self.enc.latent_mean.unsqueeze(-1)
        z = F.conv1d(z, self.enc.latent_pca.unsqueeze(-1))
        z = z[:, : self.enc.latent_size]

        z = torch.cat([z, fill], 1)
        z = F.conv1d(z, self.dec.latent_pca.t().unsqueeze(-1))
        z = z + self.dec.latent_mean.unsqueeze(-1)
        y = self.dec.decoder(z)
        audio_out = self.dec.pqmf.inverse(y)
        return audio_out, self.registry.gather()
