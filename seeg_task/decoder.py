"""实时解码模块。

:class:`BaseDecoder` 定义解码器接口（抽象基类，仅声明不实现），含两个接口函数：

    predict(x: np.ndarray) -> np.ndarray              # 推理
        x      : 形状 (n_channels, n_samples)
        返回值 : 形状 (n_classes,) 的概率分布 (>=0 且和为 1)，即 one-hot 概率分布

    update(samples: list[tuple[np.ndarray, int]]) -> bool   # 模型更新
        samples: [(x, label), ...]；返回是否实际更新了模型

:class:`Decoder` 是其参考实现。要接入自定义解码器，只需继承 :class:`BaseDecoder` 并实现
:meth:`predict_inner` 与 :meth:`update`（以及 :meth:`from_config`），即可直接被 ``Experiment``
使用，其余代码无需改动。对外的 :meth:`predict` 由基类实现并自动校验输出合法性。

参考实现 :class:`Decoder` 内部持有 ``model`` 与一个 :class:`~seeg_task.model_update.ModelTrainer`：
推理用 ``model.infer``；更新时由 trainer 重新拟合并在锁内**热替换** ``model``。``model`` 只需
实现 ``infer(features) -> logits``，``features`` 由 :func:`extract_features` 提取（可重写）。
"""

from __future__ import annotations

import abc
import importlib
import threading

import numpy as np
from loguru import logger


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


def check_probs(probs: np.ndarray, n_classes: int, *, atol: float = 1e-6) -> None:
    """校验 :meth:`BaseDecoder.predict` 的输出是否为合法概率分布。

    检查三项：形状为 ``(n_classes,)``、每个元素都在 ``[0, 1]``、所有元素之和为 ``1``。
    不满足则抛出 :class:`ValueError`。自定义解码器可在 ``predict`` 末尾调用本函数自检。

    Args:
        probs: predict 的返回值。
        n_classes: 类别数 N。
        atol: 概率和与 1 的允许误差。
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape != (n_classes,):
        raise ValueError(f"predict 输出形状应为 ({n_classes},)，实际 {probs.shape}")
    if not np.all((probs >= -atol) & (probs <= 1.0 + atol)):
        raise ValueError(
            f"predict 输出须在 [0, 1] 范围内，实际 min={probs.min():.4g}, max={probs.max():.4g}"
        )
    total = float(probs.sum())
    if abs(total - 1.0) > atol:
        raise ValueError(f"predict 输出的概率之和须为 1，实际 {total:.6f}")


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


class BaseDecoder(abc.ABC):
    """实时解码器接口（抽象基类，仅定义接口，不含实现）。

    一个解码器需具备两项能力，对应下面两个接口函数：

    1. **推理** :meth:`predict` —— 把单个采集窗口映射为类别概率分布；
    2. **模型更新** :meth:`update` —— 用历史样本在线更新自身模型。

    ``Experiment`` 仅依赖这两个方法（外加 :meth:`from_config` 构造）；自定义解码器继承
    本类并实现 :meth:`predict_inner` 与 :meth:`update` 即可接入范式，无需改动其它代码。
    约定实现应提供整型属性 ``n_classes``。

    注意：对外的 :meth:`predict` 由基类实现（模板方法）——它调用子类的 :meth:`predict_inner`
    并用 :func:`check_probs` **强制校验**输出为合法概率分布。子类只需实现 :meth:`predict_inner`，
    无需自行校验。
    """

    n_classes: int

    @abc.abstractmethod
    def predict_inner(self, x: np.ndarray) -> np.ndarray:
        """子类实现：对单个采集窗口解码。

        Args:
            x: 形状 ``(n_channels, n_samples)`` 的原始信号窗口。

        Returns:
            形状 ``(n_classes,)`` 的概率分布：每个元素都在 ``[0, 1]`` 且所有元素之和为 ``1``。
            （合法性由基类 :meth:`predict` 统一校验，子类无需重复检查。）
        """
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        """推理（对外接口）：调用子类 :meth:`predict_inner` 并强制校验输出为合法概率分布。"""
        probs = self.predict_inner(x)
        check_probs(probs, self.n_classes)
        return probs

    @abc.abstractmethod
    def update(self, samples: "list[tuple[np.ndarray, int]]") -> bool:
        """模型更新：用历史样本在线更新内部模型。

        在 block 间休息时由后台线程调用，因此实现需**线程安全**（可能与 :meth:`predict`
        并发）。

        Args:
            samples: ``[(x, label), ...]``，``x`` 形状 ``(n_channels, n_samples)``；
                ``label`` 为整型类别标签，取值范围 ``0 .. N-1``（N 为类别数 ``n_classes``）。

        Returns:
            是否实际更新了模型；样本不足等情况可返回 ``False`` 表示本次跳过。
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config, rng: np.random.Generator | None = None, **params) -> "BaseDecoder":
        """从配置构造解码器实例（供 :func:`create_decoder` 按 ``config.decoder_class`` 调用）。

        子类需重写本方法以定义自身的构造方式；``params`` 来自 ``config.decoder_params``。
        """
        raise NotImplementedError(
            f"{cls.__name__} 未实现 from_config；请在子类中实现以支持从配置文件构造"
        )


class Decoder(BaseDecoder):
    """线程安全的实时解码器（:class:`BaseDecoder` 的参考实现）。

    推理用内部 ``model`` 的 ``infer``；模型更新用内部 :class:`~seeg_task.model_update.ModelTrainer`
    重新拟合并热替换 ``model``（在锁内完成，保证与并发 :meth:`predict` 安全）。
    """

    def __init__(self, model: LinearModel, n_classes: int, trainer=None) -> None:
        self._model = model
        self.n_classes = n_classes
        self._trainer = trainer  # ModelTrainer | None；为 None 时 update 会报错提示
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config, rng: np.random.Generator | None = None, **params) -> "Decoder":
        from .model_update import ModelTrainer  # 延迟导入避免与 model_update 循环依赖

        rng = rng if rng is not None else np.random.default_rng(config.random_seed)
        model = LinearModel.random_init(config.n_classes, config.n_channels, rng)
        trainer = ModelTrainer(config.n_classes, config.n_channels)
        return cls(model, config.n_classes, trainer=trainer)

    def predict_inner(self, x: np.ndarray) -> np.ndarray:
        """推理：对单个窗口解码，返回 ``(n_classes,)`` 概率分布（合法性由基类 predict 校验）。"""
        features = extract_features(x)
        with self._lock:
            logits = self._model.infer(features)
        return softmax(np.asarray(logits, dtype=np.float64))

    def update(self, samples: "list[tuple[np.ndarray, int]]") -> bool:
        """模型更新：用历史样本重新拟合并热替换内部模型，线程安全。"""
        if self._trainer is None:
            raise RuntimeError("Decoder 未配置 trainer，无法执行 update")
        logger.info("update | 开始训练，样本数={}", len(samples))
        new_model = self._trainer.train(samples)
        if new_model is None:
            logger.info("update | 样本不足，跳过本次更新")
            return False
        self.swap_model(new_model)
        logger.info("update | 训练完成，模型已热替换")
        return True

    def swap_model(self, new_model: LinearModel) -> None:
        """热替换内部模型（线程安全）；供 :meth:`update` 调用，也可外部直接替换。"""
        with self._lock:
            self._model = new_model

    @property
    def model(self) -> LinearModel:
        with self._lock:
            return self._model


class DummyDecoder(BaseDecoder):
    """占位解码器（:class:`BaseDecoder` 的最简实现）。

    不依赖任何模型/训练：:meth:`predict` 返回一个随机概率分布，:meth:`update` 直接跳过。
    用于无真实模型时打通整条范式（界面、采集、入缓冲、休息流程）。注意正确率不具备真实
    含义，应在随机水平附近波动。

    可选参数（来自 ``config.decoder_params``）：
        seed: 随机种子，便于复现；为空时用 ``config.random_seed``。
    """

    def __init__(self, n_classes: int, rng: np.random.Generator | None = None) -> None:
        self.n_classes = n_classes
        self._rng = rng if rng is not None else np.random.default_rng()

    @classmethod
    def from_config(cls, config, rng: np.random.Generator | None = None, **params) -> "DummyDecoder":
        seed = params.get("seed", config.random_seed)
        rng = rng if rng is not None else np.random.default_rng(seed)
        return cls(config.n_classes, rng=rng)

    def predict_inner(self, x: np.ndarray) -> np.ndarray:
        """推理：忽略输入，返回一个随机概率分布（合法性由基类 predict 校验）。"""
        return softmax(self._rng.standard_normal(self.n_classes))

    def update(self, samples: "list[tuple[np.ndarray, int]]") -> bool:
        """模型更新：占位解码器无模型，直接跳过。"""
        logger.info("update(dummy) | 样本数={} -> 无模型，跳过", len(samples))
        return False


def create_decoder(config, rng: np.random.Generator | None = None):
    """按 ``config.decoder_class`` 动态导入并构造解码器实例。

    ``config.decoder_class`` 为完整导入路径（如 ``"seeg_task.decoder.Decoder"`` 或
    ``"models.model_1.model.LogisticBandPowerDecoder"``）。**鸭子类型**接入——只要求实例
    具备可调用的 ``predict`` 与 ``update``，无需继承 :class:`BaseDecoder`：

    - 若类定义了 ``from_config``，用 ``cls.from_config(config, rng=rng, **decoder_params)`` 构造；
    - 否则用 ``cls(config.n_classes, **decoder_params)`` 构造（外部自带模型常见此签名）。
    """
    target = config.decoder_class
    module_path, _, class_name = target.rpartition(".")
    if not module_path:
        raise ValueError(
            f"decoder_class 需为完整导入路径（如 'seeg_task.decoder.Decoder'），实际: {target!r}"
        )
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(f"无法导入解码器类 {target!r}: {exc}") from exc
    if not isinstance(cls, type):
        raise TypeError(f"{target} 不是一个类")

    params = dict(getattr(config, "decoder_params", {}) or {})
    if hasattr(cls, "from_config"):
        decoder = cls.from_config(config, rng=rng, **params)
    else:
        decoder = cls(config.n_classes, **params)

    for method in ("predict", "update"):
        if not callable(getattr(decoder, method, None)):
            raise TypeError(f"{target} 缺少必需的可调用方法 {method}()")
    return decoder
