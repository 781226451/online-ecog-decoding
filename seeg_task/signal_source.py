"""信号采集接口。

实时解码模块的输入约定为 ``ndarray[n_channels, n_samples]``。真实场景下数据来自
SEEG 采集硬件 / LSL 流 / 录制文件；这里只定义接口 :class:`SignalSource`，并给出一个
:class:`SyntheticSource` 模拟桩，让整套范式可以脱离硬件先跑通。

接入真实数据时，只需实现 :class:`SignalSource.read_window` 即可，其余代码无需改动。
"""

from __future__ import annotations

import abc

import numpy as np


class SignalSource(abc.ABC):
    """采集接口。一次 ``read_window`` 返回一个解码窗口。"""

    def __init__(self, n_channels: int, window_samples: int) -> None:
        self.n_channels = n_channels
        self.window_samples = window_samples

    @abc.abstractmethod
    def read_window(self, true_label: int | None = None) -> np.ndarray:
        """读取一个采集窗口。

        Args:
            true_label: 当前试次的真实动作标签。真实采集忽略该参数；模拟桩用它
                合成“可分”的信号，从而让解码/在线学习有意义可观测。

        Returns:
            形状为 ``(n_channels, window_samples)`` 的 ``float`` 数组。
        """
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027 - 可选钩子，默认无操作
        """释放底层资源（关闭设备 / 断开流）。默认无操作。"""


class SyntheticSource(SignalSource):
    """模拟信号源。

    为每个类别预生成一组固定的“通道空间模式”，叠加噪声后输出，使得不同类别在统计上
    可区分——这样解码器（即便是桩）和在线训练才能体现出正确率的变化趋势。
    """

    def __init__(
        self,
        n_channels: int,
        window_samples: int,
        n_classes: int,
        noise_level: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(n_channels, window_samples)
        self.n_classes = n_classes
        self.noise_level = noise_level
        self._rng = rng if rng is not None else np.random.default_rng()
        # 每个类别一个固定的通道权重模式 (n_classes, n_channels)
        self._patterns = self._rng.standard_normal((n_classes, n_channels))

    def read_window(self, true_label: int | None = None) -> np.ndarray:
        noise = self._rng.standard_normal((self.n_channels, self.window_samples)) * self.noise_level
        if true_label is None:
            return noise
        # 类别相关的缓慢振荡，按通道模式加权叠加到噪声上。
        t = np.linspace(0.0, 1.0, self.window_samples, endpoint=False)
        freq = 5.0 + 3.0 * true_label
        base = np.sin(2.0 * np.pi * freq * t)  # (window_samples,)
        pattern = self._patterns[true_label][:, None]  # (n_channels, 1)
        signal = pattern * base[None, :]  # (n_channels, window_samples)
        return signal + noise
