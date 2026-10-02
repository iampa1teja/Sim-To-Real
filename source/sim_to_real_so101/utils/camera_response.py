"""Per-camera residual colour response applied to rendered RGB (sim -> real camera look).

Only for what the scene (materials, lights) cannot reproduce: vignetting, exposure/white balance,
tone curve, blur and sensor noise. Fitted separately per camera; parameters live in
real_setup.json under each camera's "response" key. Every field is optional (identity default).

Order of operations on an H x W x 3 uint8 frame:
  0) x = srgb_to_linear(rgb / 255)                          (IEC 61966-2-1)
  1) x *= clip(1 + k1 r^2 + k2 r^4, 0, 4)                   r = |(u, v) - (cx, cy)| / |((W-1)/2, (H-1)/2)|,
                                                            integer pixel centres, cx, cy default to the image centre
  2) x = gain * x + offset                                  per channel, linear light
  3) y = lut(linear_to_srgb(clip(x, 0, 1)))                 17 evenly spaced knots on [0, 1], shared or per channel
  4) y = gaussian_blur(y, blur_sigma_px)                    separable, radius ceil(3 sigma), reflect padding
  5) y += N(0, 1) * (a + b y)                               per pixel and channel; one seeded generator per camera
  6) out = round(clip(y, 0, 1) * 255)
"""
import math

import torch
import torch.nn.functional as F


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp_min(0.0031308) ** (1 / 2.4) - 0.055)


def _apply_lut(y, knots):
    """Piecewise-linear curve through len(knots) evenly spaced inputs on [0, 1]; knots: (C, K)."""
    k = knots.shape[-1] - 1
    t = y.clamp(0, 1) * k
    i = t.floor().clamp(max=k - 1).long()
    w = t - i
    lo = torch.stack([knots[c][i[..., c]] for c in range(3)], dim=-1)
    hi = torch.stack([knots[c][i[..., c] + 1] for c in range(3)], dim=-1)
    return lo + w * (hi - lo)


def _gaussian_blur(y, sigma):
    """y: (N, H, W, 3) float."""
    radius = int(math.ceil(3 * sigma))
    xs = torch.arange(-radius, radius + 1, device=y.device, dtype=y.dtype)
    kernel = torch.exp(-0.5 * (xs / sigma) ** 2)
    kernel = kernel / kernel.sum()
    img = y.permute(0, 3, 1, 2).reshape(-1, 1, y.shape[1], y.shape[2])
    img = F.pad(img, (radius, radius, 0, 0), mode="reflect")
    img = F.conv2d(img, kernel.view(1, 1, 1, -1))
    img = F.pad(img, (0, 0, radius, radius), mode="reflect")
    img = F.conv2d(img, kernel.view(1, 1, -1, 1))
    return img.reshape(y.shape[0], 3, y.shape[1], y.shape[2]).permute(0, 2, 3, 1)


class CameraResponse:
    """Applies one camera's fitted response to batches of uint8 RGB frames on the camera's device."""

    def __init__(self, params: dict, height: int, width: int, device):
        self.params = params
        self.device = device
        dtype = torch.float32
        vignette = params.get("vignette")
        self.vignette = None
        if vignette:
            cx = vignette.get("cx")
            cy = vignette.get("cy")
            cx = (width - 1) / 2 if cx is None else cx
            cy = (height - 1) / 2 if cy is None else cy
            v, u = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                                  torch.arange(width, device=device, dtype=dtype), indexing="ij")
            r2 = ((u - cx) ** 2 + (v - cy) ** 2) / (((width - 1) / 2) ** 2 + ((height - 1) / 2) ** 2)
            self.vignette = (1 + vignette.get("k1", 0.0) * r2 + vignette.get("k2", 0.0) * r2 ** 2).clamp(0, 4)[..., None]
        self.gain = torch.tensor(params.get("gain", [1.0, 1.0, 1.0]), device=device, dtype=dtype)
        self.offset = torch.tensor(params.get("offset", [0.0, 0.0, 0.0]), device=device, dtype=dtype)
        lut = params.get("lut")
        self.lut = None
        if lut:
            per_channel = lut.get("per_channel")
            knots = per_channel if per_channel else [lut["knots"]] * 3
            self.lut = torch.tensor(knots, device=device, dtype=dtype)
            if self.lut.shape != (3, 17) or (self.lut.diff(dim=-1) < 0).any():
                raise ValueError("response.lut needs 17 monotone non-decreasing knots per channel")
        self.blur_sigma = float(params.get("blur_sigma_px", 0.0))
        noise = params.get("noise") or {}
        self.noise_a = float(noise.get("a", 0.0))
        self.noise_b = float(noise.get("b", 0.0))
        self.generator = None
        if self.noise_a or self.noise_b:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(int(noise.get("seed", 0)))

    def __call__(self, rgb_uint8):
        """rgb_uint8: (N, H, W, 3) uint8 tensor -> (N, H, W, 3) uint8 tensor."""
        x = srgb_to_linear(rgb_uint8.float() / 255.0)
        if self.vignette is not None:
            x = x * self.vignette
        x = x * self.gain + self.offset
        y = linear_to_srgb(x.clamp(0, 1))
        if self.lut is not None:
            y = _apply_lut(y, self.lut)
        if self.blur_sigma > 0:
            y = _gaussian_blur(y, self.blur_sigma)
        if self.generator is not None:
            noise = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=self.generator)
            y = y + noise * (self.noise_a + self.noise_b * y)
        return (y.clamp(0, 1) * 255).round().to(torch.uint8)
