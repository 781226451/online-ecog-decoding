"""在线模型更新模块。

包含两部分：

- :class:`HistoryBuffer`  : 环形缓冲，保存最近若干个 ``(x, label)`` 历史样本。
- :class:`ModelTrainer`   : 用历史样本训练 / 更新模型，产出一个**新的** model 对象，
  随后由 :meth:`~seeg_task.decoder.Decoder.swap_model` 热替换。

I/O 契约（保持稳定）::

    ModelTrainer.train(samples: list[tuple[np.ndarray, int]]) -> LinearModel
        samples : [(x, label), ...]，x 形状 (n_channels, n_samples)
        返回值  : 可被 Decoder.swap_model 接收的新模型

桩实现用岭回归（一对多最小二乘，仅依赖 numpy）拟合 :func:`~seeg_task.decoder.extract_features`
提取的特征——足以让模拟数据上的正确率随训练上升、从而验证整条更新链路。
替换真实训练算法时，只要返回的对象实现 ``infer(features) -> logits`` 即可。
"""

from __future__ import annotations

import threading

import numpy as np

from .decoder import LinearModel, extract_features


class HistoryBuffer:
    """容量受限的历史样本缓冲（先进先出截断），线程安全。"""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity 必须为正")
        self.capacity = capacity
        self._items: list[tuple[np.ndarray, int]] = []
        self._lock = threading.Lock()

    def add(self, x: np.ndarray, label: int) -> None:
        with self._lock:
            self._items.append((np.asarray(x), int(label)))
            if len(self._items) > self.capacity:
                # 丢弃最旧的样本
                self._items = self._items[-self.capacity :]

    def recent(self, n: int) -> list[tuple[np.ndarray, int]]:
        """返回最近 ``n`` 个样本的快照（拷贝，供训练线程安全读取）。"""
        with self._lock:
            return list(self._items[-n:])

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class ModelTrainer:
    """离线训练桩：岭回归一对多分类器。"""

    def __init__(self, n_classes: int, n_features: int, ridge_lambda: float = 1.0) -> None:
        self.n_classes = n_classes
        self.n_features = n_features
        self.ridge_lambda = ridge_lambda

    def train(self, samples: list[tuple[np.ndarray, int]]) -> LinearModel | None:
        """用历史样本拟合一个新的 :class:`LinearModel`。

        Returns:
            新模型；若样本不足（少于 2 个不同类别）则返回 ``None`` 表示本次不更新。
        """
        if len(samples) < 2:
            return None

        feats = np.stack([extract_features(x) for x, _ in samples])  # (N, F)
        labels = np.asarray([lab for _, lab in samples], dtype=int)  # (N,)
        if len(np.unique(labels)) < 2:
            return None

        # 标准化特征（数值稳定 + 各通道量纲一致）。
        mu = feats.mean(axis=0)
        sigma = feats.std(axis=0) + 1e-8
        feats_n = (feats - mu) / sigma

        # 增广偏置项。
        n, f = feats_n.shape
        phi = np.hstack([feats_n, np.ones((n, 1))])  # (N, F+1)

        # one-hot 目标，{-1,+1} 编码更利于最小二乘分类。
        targets = -np.ones((n, self.n_classes))
        targets[np.arange(n), labels] = 1.0

        # 岭回归闭式解：beta = (PhiᵀPhi + λI)⁻¹ Phiᵀ T，不正则化偏置列。
        reg = self.ridge_lambda * np.eye(f + 1)
        reg[-1, -1] = 0.0
        beta = np.linalg.solve(phi.T @ phi + reg, phi.T @ targets)  # (F+1, n_classes)

        # 把“标准化 + 线性”折叠回作用于原始特征的等价权重：
        #   z = (feat - mu)/sigma ; logits = [z,1] @ beta
        #     = feat @ (W_z/sigma) + (b_z - sum(W_z*mu/sigma))
        w_z = beta[:f, :]            # (F, n_classes)
        b_z = beta[f, :]             # (n_classes,)
        w_raw = (w_z / sigma[:, None])               # (F, n_classes)
        b_raw = b_z - (w_z * (mu / sigma)[:, None]).sum(axis=0)  # (n_classes,)

        return LinearModel(weights=w_raw.T, bias=b_raw)
