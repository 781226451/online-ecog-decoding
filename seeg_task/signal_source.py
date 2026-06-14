"""信号采集接口。

信号源是**逐样本的流式数据源**：一次 :meth:`SignalSource.read_sample` 返回一列
``(n_channels, 1)`` 新采样。窗口化（把样本聚成 ``(n_channels, N)``）由上层
:class:`~seeg_task.buffer.BlockBuffer` 负责，因此信号源**不需要固定窗口长度**。

本模块定义统一接口 :class:`SignalSource`，并提供三种实现：

- :class:`LSLSource`       : 通过 Lab Streaming Layer (pylsl) 接收**外部实时数据**。
- :class:`DummySource`     : 纯随机噪声的 dummy 源，无需硬件即可跑通流程（无类别结构）。
- :class:`SyntheticSource` : 带类别可分结构的模拟源，可让解码/在线训练的正确率有可观测的提升。

用 :func:`create_source` 按 :class:`~seeg_task.config.ExperimentConfig` 选择具体实现。
接入其它真实设备时，只需实现 :meth:`SignalSource.read_sample` 即可，其余代码无需改动。
"""

from __future__ import annotations

import abc
import threading
from collections import deque

import numpy as np


class SignalSource(abc.ABC):
    """采集接口：逐样本流式数据源（不持有窗口长度）。"""

    def __init__(self, n_channels: int) -> None:
        self.n_channels = n_channels

    @abc.abstractmethod
    def read_sample(self, true_label: int | None = None) -> np.ndarray | None:
        """读取**单列**新采样。

        Args:
            true_label: 当前试次的真实动作标签（模拟桩用于合成可分信号；真实采集忽略）。

        Returns:
            形状为 ``(n_channels, 1)`` 的数组；若当前暂无新样本（如 LSL 非阻塞拉取为空）
            则返回 ``None``。
        """
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027 - 可选钩子，默认无操作
        """释放底层资源（关闭设备 / 断开流）。默认无操作。"""


class SyntheticSource(SignalSource):
    """模拟信号源。

    为每个类别预生成一组固定的“通道空间模式”，逐样本叠加噪声后输出，使得不同类别在统计上
    可区分——这样解码器（即便是桩）和在线训练才能体现出正确率的变化趋势。
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        noise_level: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(n_channels)
        self.n_classes = n_classes
        self.noise_level = noise_level
        self._rng = rng if rng is not None else np.random.default_rng()
        # 每个类别一个固定的通道权重模式 (n_classes, n_channels)
        self._patterns = self._rng.standard_normal((n_classes, n_channels))
        self._step = 0  # 采样步进计数（用于类别相关振荡相位）

    def read_sample(self, true_label: int | None = None) -> np.ndarray:
        noise = self._rng.standard_normal((self.n_channels, 1)) * self.noise_level
        if true_label is None:
            return noise
        self._step += 1
        osc = np.sin(2.0 * np.pi * (0.01 * (1 + true_label)) * self._step)
        pattern = self._patterns[true_label][:, None]  # (n_channels, 1)
        return pattern * osc + noise


class DummySource(SignalSource):
    """纯随机噪声的 dummy 数据源。

    不含任何类别信息，仅用于无硬件时验证整条流程（界面、解码调用、缓冲、训练编排）。
    注意：因数据与标签无关，在线训练不会提升正确率（正确率应在随机水平附近波动）。
    """

    def __init__(
        self,
        n_channels: int,
        noise_level: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(n_channels)
        self.noise_level = noise_level
        self._rng = rng if rng is not None else np.random.default_rng()

    def read_sample(self, true_label: int | None = None) -> np.ndarray:
        return self._rng.standard_normal((self.n_channels, 1)) * self.noise_level


class LSLSource(SignalSource):
    """通过 Lab Streaming Layer 接收外部实时数据。

    后台线程持续从 LSL inlet 拉取样本写入一个加锁的 FIFO 队列；:meth:`read_sample`
    在主线程逐列弹出最早未消费的样本，从而把采集与试次循环解耦。窗口化由上层
    :class:`~seeg_task.buffer.BlockBuffer` 负责，本类不持有窗口长度。

    依赖 ``pylsl``（惰性导入：仅在实例化本类时才需要安装）。
    """

    def __init__(
        self,
        n_channels: int,
        stream_name: str | None = None,
        stream_type: str = "EEG",
        resolve_timeout: float = 5.0,
        pull_timeout: float = 0.2,
        max_queue: int = 4096,
    ) -> None:
        super().__init__(n_channels)
        try:
            from pylsl import StreamInlet, resolve_byprop
        except ImportError as exc:  # noqa: TRY003
            raise ImportError(
                "使用 LSLSource 需要安装 pylsl：`uv add pylsl` 或 `pip install pylsl`"
            ) from exc

        # 优先按流名解析，否则按流类型（如 'EEG'）解析。
        if stream_name:
            streams = resolve_byprop("name", stream_name, timeout=resolve_timeout)
        else:
            streams = resolve_byprop("type", stream_type, timeout=resolve_timeout)
        if not streams:
            target = stream_name or f"type={stream_type}"
            raise RuntimeError(f"未解析到 LSL 流（{target}）；请确认数据源已启动")

        self._inlet = StreamInlet(streams[0])
        info = self._inlet.info()
        self._stream_channels = info.channel_count()
        if self._stream_channels != n_channels:
            raise ValueError(
                f"LSL 流通道数 {self._stream_channels} 与配置 n_channels {n_channels} 不一致；"
                f"请把 n_channels 设为 {self._stream_channels}（或选用通道数匹配的流）。"
            )
        srate = info.nominal_srate()
        print(f"[LSLSource] 已连接流 '{info.name()}' "
              f"(type={info.type()}, ch={self._stream_channels}, srate={srate})")

        self._pull_timeout = pull_timeout
        # FIFO 队列：后台线程入队、主线程逐列消费（maxlen 防止积压时无限增长）
        self._q: deque = deque(maxlen=max_queue)
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._pull_loop, name="lsl-puller", daemon=True)
        self._thread.start()

    def _pull_loop(self) -> None:
        while self._running:
            try:
                samples, _ts = self._inlet.pull_chunk(timeout=self._pull_timeout, max_samples=1024)
            except Exception as exc:  # noqa: BLE001 - 拉流异常不应使线程崩溃
                print(f"[LSLSource] pull_chunk 异常: {exc}")
                continue
            if samples:
                with self._lock:
                    self._q.extend(samples)

    def read_sample(self, true_label: int | None = None) -> np.ndarray | None:
        """从 FIFO 队列弹出一列最早未消费的样本，返回 ``(n_channels, 1)``；暂无新样本时返回 None。

        强制校验：收到的样本通道数必须等于配置 ``n_channels``，不一致直接抛 :class:`ValueError`。
        """
        with self._lock:
            frame = self._q.popleft() if self._q else None
        if frame is None:
            return None
        col = np.asarray(frame, dtype=np.float64).reshape(-1, 1)
        if col.shape[0] != self.n_channels:
            raise ValueError(
                f"收到样本通道数 {col.shape[0]} 与配置 n_channels {self.n_channels} 不一致"
            )
        return col

    def close(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            self._inlet.close_stream()
        except Exception:  # noqa: BLE001
            pass


def create_source(config, rng: np.random.Generator | None = None) -> SignalSource:
    """按 ``config.source_type`` 构造信号源。

    支持 ``"lsl"`` / ``"dummy"`` / ``"synthetic"``（大小写不敏感）。
    """
    kind = config.source_type.lower()
    if kind == "lsl":
        return LSLSource(
            config.n_channels,
            stream_name=config.lsl_stream_name,
            stream_type=config.lsl_stream_type,
            resolve_timeout=config.lsl_resolve_timeout,
        )
    if kind == "dummy":
        return DummySource(config.n_channels, rng=rng)
    if kind == "synthetic":
        return SyntheticSource(config.n_channels, config.n_classes, rng=rng)
    raise ValueError(f"未知 source_type: {config.source_type!r}（可选 lsl/dummy/synthetic）")
