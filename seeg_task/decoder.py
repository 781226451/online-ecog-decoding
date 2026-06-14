"""实时解码模块。

I/O 契约（保持稳定，便于替换为真实模型）::

    Decoder.predict(x: np.ndarray) -> np.ndarray
        x      : 形状 (n_channels, n_samples)
        返回值 : 形状 (n_classes,) 的概率分布 (>=0 且和为 1)，即 one-hot 概率分布

解码器内部持有一个 ``model`` 对象，并用锁保护，支持在 block 间休息时由训练线程
**热替换**（:meth:`Decoder.swap_model`）。``model`` 只需实现 ``infer(features) -> logits``，
``features`` 由 :func:`extract_features` 从原始 ``x`` 提取。

要接入真实模型：实现一个同样具备 ``infer(features)`` 的对象（或直接在 ``LinearModel``
基础上扩展），其余流程无需改动；也可重写 :func:`extract_features` 改用你自己的特征。
"""

from __future__ import annotations

import threading

import numpy as np


def extract_features(x: np.ndarray) -> np.ndarray:
    """从原始窗口提取定长特征向量。

    默认使用每通道的 log 功率（时间维上的均方），得到长度为 ``n_channels`` 的向量。
    这是一个占位特征——替换真实模型时可同步替换为带通能量 / CSP / 深度特征等。

    Args:
        x: 形状 ``(n_channels, n_samples)``。

    Returns:
        形状 ``(n_channels,)`` 的特征向量。
    """
    if x.ndim != 2:
        raise ValueError(f"x 期望二维 (n_channels, n_samples)，实际 {x.shape}")
    power = np.mean(np.square(x, dtype=np.float64), axis=1)
    return np.log(power + 1e-8)


def softmax(logits: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax，返回概率分布。"""
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


class LinearModel:
    """占位模型：线性分类器 (logits = W @ features + b)。

    既可被随机初始化（实验开始时），也可由 :class:`~seeg_task.model_update.ModelTrainer`
    拟合后产出新的实例用于热替换。
    """

    def __init__(self, weights: np.ndarray, bias: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)  # (n_classes, n_features)
        self.bias = np.asarray(bias, dtype=np.float64)        # (n_classes,)

    @classmethod
    def random_init(cls, n_classes: int, n_features: int, rng: np.random.Generator) -> "LinearModel":
        # 小权重 -> 初始接近均匀分布，正确率约等于随机猜测。
        w = rng.standard_normal((n_classes, n_features)) * 0.01
        b = np.zeros(n_classes)
        return cls(w, b)

    def infer(self, features: np.ndarray) -> np.ndarray:
        """返回未归一化的 logits，形状 ``(n_classes,)``。"""
        return self.weights @ features + self.bias


class Decoder:
    """线程安全的实时解码器。"""

    def __init__(self, model: LinearModel, n_classes: int) -> None:
        self._model = model
        self.n_classes = n_classes
        self._lock = threading.Lock()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """对单个窗口解码，返回 ``(n_classes,)`` 概率分布。"""
        features = extract_features(x)
        with self._lock:
            logits = self._model.infer(features)
        probs = softmax(np.asarray(logits, dtype=np.float64))
        if probs.shape != (self.n_classes,):
            raise ValueError(
                f"模型输出维度 {probs.shape} 与类别数 {self.n_classes} 不一致"
            )
        return probs

    def swap_model(self, new_model: LinearModel) -> None:
        """热替换内部模型（由训练线程在休息期调用）。"""
        with self._lock:
            self._model = new_model

    @property
    def model(self) -> LinearModel:
        with self._lock:
            return self._model
