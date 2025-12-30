"""
CAMPPlus X-vector speaker encoder.

Extracts 192-dimensional speaker embeddings from audio.
Architecture based on 3D-Speaker (Alibaba DAMO Academy).
"""

from collections import OrderedDict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


def statistics_pooling(x: torch.Tensor, dim: int = -1, keepdim: bool = False, unbiased: bool = True, eps: float = 1e-2) -> torch.Tensor:
    """Statistics pooling: compute mean and std."""
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    """Statistics pooling layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return statistics_pooling(x)


def get_nonlinear(config_str: str, channels: int) -> nn.Sequential:
    """Build nonlinearity from config string."""
    nonlinear = nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", nn.ReLU(inplace=True))
        elif name == "prelu":
            nonlinear.add_module("prelu", nn.PReLU(channels))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError(f"Unexpected module: {name}")
    return nonlinear


class BasicResBlock(nn.Module):
    """Basic residual block for FCM."""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FCM(nn.Module):
    """Feature Calibration Module."""

    def __init__(self, block=BasicResBlock, num_blocks: List[int] = [2, 2], m_channels: int = 32, feat_dim: int = 80):
        super().__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)

        self.conv2 = nn.Conv2d(
            m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


class TDNNLayer(nn.Module):
    """Time-Delay Neural Network layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
    ):
        super().__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, f"Expect odd kernel size, got {kernel_size}"
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(nn.Module):
    """Context-Aware Mask Layer."""

    def __init__(
        self,
        bn_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        bias: bool,
        reduction: int = 2,
    ):
        super().__init__()
        self.linear_local = nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x: torch.Tensor, seg_len: int = 100, stype: str = "avg") -> torch.Tensor:
        if stype == "avg":
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError(f"Wrong segment pooling type: {stype}")
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., : x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):
    """CAM Dense TDNN Layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = False,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, f"Expect odd kernel size, got {kernel_size}"
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def bn_function(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear1(self.nonlinear1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):
    """CAM Dense TDNN Block."""

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = False,
    ):
        super().__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.add_module("tdnnd%d" % (i + 1), layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    """Transition layer for dimension reduction."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, config_str: str = "batchnorm-relu"):
        super().__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):
    """Dense layer with nonlinearity."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False, config_str: str = "batchnorm-relu"):
        super().__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMPPlus(nn.Module):
    """
    CAMPPlus speaker encoder.

    Extracts 192-dimensional speaker embeddings from 80-dim fbank features.
    Architecture from 3D-Speaker (Alibaba DAMO Academy).
    """

    def __init__(
        self,
        feat_dim: int = 80,
        embedding_size: int = 192,
        growth_rate: int = 32,
        bn_size: int = 4,
        init_channels: int = 128,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = True,
        output_level: str = "segment",
    ):
        super().__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.output_level = output_level

        self.xvector = nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    ),
                ]
            )
        )
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module("block%d" % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                "transit%d" % (i + 1),
                TransitLayer(channels, channels // 2, bias=False, config_str=config_str),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))

        if self.output_level == "segment":
            self.xvector.add_module("stats", StatsPool())
            self.xvector.add_module(
                "dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")
            )
        else:
            assert self.output_level == "frame", "`output_level` should be 'segment' or 'frame'."

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch, time, feat_dim)

        Returns:
            Speaker embeddings of shape (batch, embedding_size)
        """
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        if self.output_level == "frame":
            x = x.transpose(1, 2)
        return x

    @torch.inference_mode()
    def inference(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from waveform.

        Args:
            wav: Audio waveform at 16kHz

        Returns:
            Speaker embedding of shape (1, embedding_size)
        """
        # Compute fbank features
        fbank = self._compute_fbank(wav)

        # Get embedding
        embedding = self(fbank)

        return embedding

    def _compute_fbank(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Compute 80-dimensional fbank features.

        Args:
            wav: Audio at 16kHz

        Returns:
            Fbank features of shape (batch, time, 80)
        """
        try:
            import torchaudio.compliance.kaldi as kaldi

            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            fbank = kaldi.fbank(
                wav,
                num_mel_bins=80,
                sample_frequency=16000,
                frame_length=25,
                frame_shift=10,
            )

            # Normalize
            fbank = fbank - fbank.mean(dim=0, keepdim=True)

            return fbank.unsqueeze(0)

        except ImportError:
            # Fallback using librosa
            import numpy as np
            import librosa

            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()

            if wav.ndim == 2:
                wav = wav.squeeze(0)

            mel = librosa.feature.melspectrogram(
                y=wav,
                sr=16000,
                n_fft=512,
                hop_length=160,
                n_mels=80,
            )
            fbank = librosa.power_to_db(mel, ref=np.max).T

            # Normalize
            fbank = fbank - fbank.mean(axis=0, keepdims=True)

            return torch.from_numpy(fbank).float().unsqueeze(0)
