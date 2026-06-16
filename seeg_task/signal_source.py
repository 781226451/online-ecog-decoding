"""信号采集接口。


信号源是**流式数据源**，对外只有三件事：

- :meth:`SignalSource.read`  : 一次性取走当前**全部可用**数据，返回 ``(n_channels, k)`` 或 ``None``。
- :meth:`SignalSource.flush` : 丢弃尚未消费的全部数据。
- :meth:`SignalSource.close` : 关闭并释放资源。

窗口化（把样本聚成 ``(n_channels, N)``）由上层 :class:`~seeg_task.buffer.BlockBuffer` 负责，
信号源不持有窗口长度。三种实现：

- :class:`LSLSource`       : 通过 Lab Streaming Layer (pylsl) 接收**外部实时数据**（后台线程入队，read 排空队列）。
- :class:`DummySource`     : 纯随机噪声模拟源。
- :class:`SyntheticSource` : 每通道一个频率的正弦 + 噪声的结构化模拟源。

数据**不与动作类别挂钩**——信号源只负责产数，不需要也不知道当前 trial 的标签。
模拟源按 ``sampling_rate`` **自计时**产数：``read()`` 返回「距上次读取以来按采样率应产生的样本」，
从而与真实流「取走已到达数据」语义一致，并把节流逻辑收敛进信号源内部。
用 :func:`create_source` 按 :class:`~seeg_task.config.ExperimentConfig` 选择实现。
"""

from __future__ import annotations

import abc
import threading
import time
from collections import deque

import numpy as np
from loguru import logger


class SignalSource(abc.ABC):
    """采集接口：read（取全部可用）/ flush（丢弃全部）/ close（关闭）。"""

    def __init__(self, n_channels: int) -> None:
        self.n_channels = n_channels

    @abc.abstractmethod
    def read(self) -> np.ndarray | None:
        """一次性获取当前**全部可用**数据。

        Returns:
            形状 ``(n_channels, k)``（``k>=1``）的数组；当前暂无数据时返回 ``None``。
        """
        raise NotImplementedError

    def flush(self) -> None:  # noqa: B027 - 默认无操作，带缓冲的源覆盖
        """丢弃尚未消费的全部数据（如进入采集阶段前清掉积压）。"""

    def close(self) -> None:  # noqa: B027 - 默认无操作
        """关闭并释放底层资源（设备 / 流）。"""


class _PacedSource(SignalSource):
    """按采样率自计时的模拟源基类：``read`` 返回距上次读取以来应产生的样本数。"""

    def __init__(self, n_channels: int, sampling_rate: float,
                 rng: np.random.Generator | None = None) -> None:
        super().__init__(n_channels)
        self.sampling_rate = sampling_rate
        self._rng = rng if rng is not None else np.random.default_rng()
        self._t0 = time.monotonic()
        self._emitted = 0

    def _take(self) -> int:
        """返回本次应产生的新样本数 k（>=0），并推进已产出计数。"""
        due = int((time.monotonic() - self._t0) * self.sampling_rate)
        k = max(0, due - self._emitted)
        self._emitted += k
        return k

    def flush(self) -> None:
        """重置计时基线：丢弃“应已积累但未读取”的样本。"""
        self._t0 = time.monotonic()
        self._emitted = 0


class SyntheticSource(_PacedSource):
    """结构化模拟源：每通道一个固定频率的正弦 + 噪声（与动作类别无关）。"""

    def __init__(self, n_channels: int, sampling_rate: float,
                 noise_level: float = 1.0, rng: np.random.Generator | None = None) -> None:
        super().__init__(n_channels, sampling_rate, rng)
        self.noise_level = noise_level
        self._freqs = self._rng.uniform(8.0, 30.0, size=n_channels)  # 每通道一个频率 (Hz)
        self._phase = 0  # 已产出样本数（用于相位连续）

    def read(self) -> np.ndarray | None:
        k = self._take()
        if k == 0:
            return None
        t = (self._phase + np.arange(k)) / self.sampling_rate  # (k,) 秒
        self._phase += k
        osc = np.sin(2.0 * np.pi * self._freqs[:, None] * t[None, :])  # (n_channels, k)
        return osc + self._rng.standard_normal((self.n_channels, k)) * self.noise_level


class DummySource(_PacedSource):
    """纯随机噪声模拟源（无类别结构）。

    仅用于无硬件时跑通流程；因数据与标签无关，在线训练不会提升正确率。
    """

    def __init__(self, n_channels: int, sampling_rate: float,
                 noise_level: float = 1.0, rng: np.random.Generator | None = None) -> None:
        super().__init__(n_channels, sampling_rate, rng)
        self.noise_level = noise_level

    def read(self) -> np.ndarray | None:
        k = self._take()
        if k == 0:
            return None
        return self._rng.standard_normal((self.n_channels, k)) * self.noise_level


class LSLSource(SignalSource):
    """通过 Lab Streaming Layer 接收外部实时数据。

    后台线程持续从 LSL inlet 拉取样本写入一个加锁的 FIFO 队列；:meth:`read` 在主线程
    一次性排空队列里现有的全部样本。依赖 ``pylsl``（惰性导入：仅实例化本类时才需要）。
    """

    def __init__(
        self,
        n_channels: int,
        stream_name: str | None = None,
        stream_type: str = "EEG",
        resolve_timeout: float = 5.0,
        pull_timeout: float = 0.2,
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
        if self._stream_channels < n_channels:
            raise ValueError(
                f"LSL 流通道数 {self._stream_channels} 小于配置 n_channels {n_channels}；"
                f"请把 n_channels 设为不超过 {self._stream_channels} 的值。"
            )
        if self._stream_channels > n_channels:
            logger.warning(
                "LSL 流通道数 {} > n_channels {}，将只取前 {} 个通道",
                self._stream_channels, n_channels, n_channels,
            )
        srate = info.nominal_srate()
        print(f"[LSLSource] 已连接流 '{info.name()}' "
              f"(type={info.type()}, ch={self._stream_channels}, srate={srate})")

        self._pull_timeout = pull_timeout
        self._q: deque = deque()  # 无界 FIFO：后台入队、主线程排空
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
                logger.debug("LSL pull_chunk: {} samples", len(samples))
                with self._lock:
                    self._q.extend(s[: self.n_channels] for s in samples)

    def read(self) -> np.ndarray | None:
        """一次性排空后台队列，返回 ``(n_channels, k)``；队列空时返回 None。

        强制校验通道数 == ``n_channels``，不一致直接抛 :class:`ValueError`。
        """
        with self._lock:
            if not self._q:
                return None
            frames = list(self._q)
            self._q.clear()
        chunk = np.asarray(frames, dtype=np.float64).T  # (stream_channels, k)
        chunk = chunk[: self.n_channels]               # 截取前 n_channels 个通道
        logger.debug("LSL read: {} frames", chunk.shape[1])
        return chunk

    def flush(self) -> None:
        with self._lock:
            self._q.clear()

    def close(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            self._inlet.close_stream()
        except Exception:  # noqa: BLE001
            pass


def create_source(config, rng: np.random.Generator | None = None) -> SignalSource:
    """按 ``config.source_type`` 构造信号源（``"lsl"`` / ``"dummy"`` / ``"synthetic"``）。"""
    kind = config.source_type.lower()
    if kind == "lsl":
        return LSLSource(
            config.n_channels,
            stream_name=config.lsl_stream_name,
            stream_type=config.lsl_stream_type,
            resolve_timeout=config.lsl_resolve_timeout,
        )
    if kind == "dummy":
        return DummySource(config.n_channels, config.sampling_rate, rng=rng)
    if kind == "synthetic":
        return SyntheticSource(config.n_channels, config.sampling_rate, rng=rng)
    raise ValueError(f"未知 source_type: {config.source_type!r}（可选 lsl/dummy/synthetic）")
