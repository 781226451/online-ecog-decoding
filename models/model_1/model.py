import abc
import pickle
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ============================ 参数 ============================

FS = 4096                       # 原始采样率 Hz
BANDPASS = (0.5, 500.0)             # 带通范围
LINE_FREQ = 50.0                    # 工频基频
TARGET_FS = 500.0                   # 降采样目标

EXCLUDE_CHANNELS = [10]             # Ch11 无信号 (0-based)
N_ALL_CH = 16
VALID_CH = [i for i in range(N_ALL_CH) if i not in EXCLUDE_CHANNELS]
N_VALID_CH = len(VALID_CH)          # 15

BANDS = {
    "alpha":        (8, 13),
    "beta":         (13, 30),
    "low_gamma":    (30, 70),
    "high_gamma":   (70, 150),
}
BAND_NAMES = list(BANDS.keys())
N_BANDS = len(BANDS)                # 4
N_FEATURES = N_VALID_CH * N_BANDS   # 60


class BaseDecoder(abc.ABC):
    """实时解码器接口（抽象基类，仅定义接口，不含实现）。

    一个解码器需具备两项能力，对应下面两个接口函数：

    1. **推理** :meth:`predict` —— 把单个采集窗口映射为类别概率分布；
    2. **模型更新** :meth:`update` —— 用历史样本在线更新自身模型。

    ``Experiment`` 仅依赖这两个方法（外加 :meth:`from_config` 构造）；自定义解码器继承
    本类并实现它们即可接入范式，无需改动其它代码。约定实现应提供整型属性 ``n_classes``。
    """

    n_classes: int

    @abc.abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """推理：对单个采集窗口解码。

        Args:
            x: 形状 ``(n_channels, n_samples)`` 的原始信号窗口。

        Returns:
            形状 ``(n_classes,)`` 的概率分布（每个元素 >=0 且总和为 1）。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, samples: "list[tuple[np.ndarray, int]]",
               model_dir: str = "models",
               blend_weight: float | None = None) -> bool:
        """模型更新：用历史样本在线更新内部模型。

        在 block 间休息时由后台线程调用，因此实现需**线程安全**（可能与 :meth:`predict`
        并发）。

        Args:
            samples: ``[(x, label), ...]``，``x`` 形状 ``(n_channels, n_samples)``；
                ``label`` 为整型类别标签，取值范围 ``0 .. N-1``。
            model_dir: 模型保存目录。
            blend_weight: 新旧模型融合权重（0~1），None 则按样本数自动计算。

        Returns:
            是否实际更新了模型。
        """
        raise NotImplementedError


# ======================================================================
# 具体实现：基于频段功率特征的 Logistic 回归解码器
# ======================================================================


class LogisticBandPowerDecoder(BaseDecoder):
    """用频段功率特征 + Logistic 回归做实时解码。

    处理流程：
    1.  0.5-500 Hz 零相位带通 + 50 Hz 谐波陷波 + 降采样到 ~500 Hz
    2.  提取 5 个频段（alpha / beta / low_gamma / high_gamma / vhigh_gamma）
        在 15 个有效通道（排除 Ch11）上的带通滤波方差（= 平均功率）
    3.  拼接为 15 x 5 = 75 维特征向量
    4.  Logistic Regression (multinomial softmax) 输出概率分布

    参数：
        n_classes: 类别数（构造时或 from_config 传入）。
    """

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self._lock = threading.Lock()

        # 前置滤波器
        self._nyquist = FS / 2.0
        self._sos_bp = self._build_bandpass()
        self._sos_notches = self._build_notches()
        self._dec_q = int(round(FS / TARGET_FS))
        self._fs_proc = FS / self._dec_q

        # 4 个频段的带通滤波器
        self._band_filters: list[np.ndarray] = []
        for lo, hi in BANDS.values():
            hi_ = min(hi, self._fs_proc / 2 * 0.95)
            sos = signal.butter(4, [lo, hi_], btype="bandpass",
                                fs=self._fs_proc, output="sos")
            self._band_filters.append(sos)

        # 模型 & 标准化
        self._scaler = StandardScaler()
        self._clf = LogisticRegression(
            solver="lbfgs", max_iter=500, C=1.0,
        )
        self._fitted = False
        self._n_samples = 0          # 历史训练样本总数（用于新旧加权）

    # ----- 内部：构建滤波器 -----

    def _build_bandpass(self) -> np.ndarray:
        high = min(BANDPASS[1], self._nyquist * 0.95)
        return signal.butter(4, [BANDPASS[0], high],
                             btype="bandpass", fs=FS, output="sos")

    def _build_notches(self) -> list[np.ndarray]:
        notches = []
        lf = LINE_FREQ
        while lf <= BANDPASS[1] and lf < self._nyquist:
            b, a = signal.iirnotch(w0=lf, Q=30.0, fs=FS)
            notches.append(signal.tf2sos(b, a))
            lf += LINE_FREQ
        return notches

    # ----- 内部：预处理单窗口 -----

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        """x 形状 (n_channels, n_samples) -> (n_samples_proc, 15)"""
        y = x.astype(np.float64).T         # (samples, ch)
        y = signal.sosfiltfilt(self._sos_bp, y, axis=0)
        for sos_n in self._sos_notches:
            y = signal.sosfiltfilt(sos_n, y, axis=0)
        y = signal.decimate(y, self._dec_q, ftype="iir", zero_phase=True, axis=0)
        y = y[:, VALID_CH]                 # 排除 Ch11 → 15 通道
        return y

    # ----- 内部：提取 75 维特征 -----

    def _extract_features(self, x_proc: np.ndarray) -> np.ndarray:
        """x_proc (n_samples, 15) -> (75,) 特征向量"""
        feat = []
        for sos in self._band_filters:
            filtered = signal.sosfiltfilt(sos, x_proc, axis=0)
            power = np.var(filtered, axis=0)    # 每通道方差 = 平均功率
            feat.extend(power)
        return np.asarray(feat, dtype=np.float64)

    # ----- 推理 -----

    def predict(self, x: np.ndarray) -> np.ndarray:
        """单窗口解码。

        Args:
            x: 形状 ``(n_channels, n_samples)`` 原始 EEG。

        Returns:
            形状 ``(n_classes,)`` 的概率分布。
        """
        if x.ndim != 2:
            raise ValueError("x must be 2-D (n_channels, n_samples), "
                             "got shape {}".format(x.shape))

        x_proc = self._preprocess(x)
        feat = self._extract_features(x_proc)

        with self._lock:
            if not self._fitted:
                return np.full(self.n_classes, 1.0 / self.n_classes)

            feat_s = self._scaler.transform(feat.reshape(1, -1))
            proba = np.asarray(self._clf.predict_proba(feat_s)[0], dtype=np.float64)
            proba = np.clip(proba, 0.0, 1.0)
            proba /= proba.sum()
            return proba

    # ----- 模型更新 -----

    def update(self, samples: "list[tuple[np.ndarray, int]]",
               model_dir: str = "models",
               blend_weight: float | None = None) -> bool:
        """用历史样本训练新模型，与旧模型加权融合后保存。

        新模型 = (1 - w) * 旧模型 + w * 新模型，
        其中 w = n_new / (n_old + n_new)，也可手动指定 ``blend_weight``。

        保存路径: ``{model_dir}/{YYYYMMDD_HHMMSS}.pkl``

        Args:
            samples: ``[(x, label), ...]``
            model_dir: 模型保存目录，默认 ``"models"``。
            blend_weight: 新模型权重（0~1），None 表示按样本数自动计算。

        Returns:
            是否实际更新。
        """
        n_new = len(samples)
        if n_new < 5:
            return False

        # ---- 提取新样本特征，训练新模型 ----
        X_new, y_new = [], []
        for x, label in samples:
            x_proc = self._preprocess(x)
            X_new.append(self._extract_features(x_proc))
            y_new.append(label)

        X_new = np.array(X_new)
        y_new = np.array(y_new)

        clf_new = LogisticRegression(solver="lbfgs", max_iter=500, C=1.0)
        scaler_new = StandardScaler()
        X_new_scaled = scaler_new.fit_transform(X_new)
        clf_new.fit(X_new_scaled, y_new)

        # ---- 确定融合权重 ----
        if blend_weight is None:
            total = self._n_samples + n_new
            alpha = n_new / total if total > 0 else 1.0
        else:
            alpha = float(blend_weight)
            alpha = max(0.0, min(1.0, alpha))

        # ---- 加权融合参数 ----
        with self._lock:
            if self._fitted and 0.0 < alpha < 1.0:
                # 只有当新旧模型的类别集合一致时才融合参数
                if (self._clf.coef_.shape == clf_new.coef_.shape
                        and self._clf.classes_.shape == clf_new.classes_.shape
                        and np.array_equal(self._clf.classes_, clf_new.classes_)):
                    self._clf.coef_ = ((1 - alpha) * self._clf.coef_
                                       + alpha * clf_new.coef_)
                    self._clf.intercept_ = ((1 - alpha) * self._clf.intercept_
                                            + alpha * clf_new.intercept_)
                else:
                    # 类别不一致，直接用新模型（但保留旧样本计数）
                    self._clf = clf_new
                    self._clf.classes_ = clf_new.classes_
                    self._clf.coef_ = clf_new.coef_
                    self._clf.intercept_ = clf_new.intercept_

                # scaler 始终融合（与类别无关，只依赖特征分布）
                self._scaler.mean_ = ((1 - alpha) * self._scaler.mean_
                                      + alpha * scaler_new.mean_)
                self._scaler.scale_ = ((1 - alpha) * self._scaler.scale_
                                       + alpha * scaler_new.scale_)
            else:
                # 首次训练或 alpha==1：直接替换
                self._clf = clf_new
                self._scaler = scaler_new
                self._fitted = True

            self._n_samples += n_new

        # ---- 保存到磁盘 ----
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        predict_path = Path(model_dir) / "predict.pkl"

        # 构造轻量状态
        state = {
            "n_classes": self.n_classes,
            "coef_": self._clf.coef_,
            "intercept_": self._clf.intercept_,
            "scaler_mean_": self._scaler.mean_,
            "scaler_scale_": self._scaler.scale_,
            "n_samples": self._n_samples,
            "classes_": self._clf.classes_,
        }

        # 旧 predict.pkl → 归档为 日期时间.pkl
        if predict_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = Path(model_dir) / "{}.pkl".format(timestamp)
            predict_path.rename(archive_path)

        # 新权重写入 predict.pkl
        with predict_path.open("wb") as f:
            pickle.dump(state, f)

        return True

    def load_weights(self, pkl_path: str | None = None,
                     model_dir: str = "models") -> bool:
        """从 pkl 文件加载模型权重。

        默认读取 ``{model_dir}/predict.pkl``；也可通过 ``pkl_path`` 指定。

        Args:
            pkl_path:  指定 .pkl 文件路径（None 则用 predict.pkl）。
            model_dir: 模型目录（pkl_path 为 None 时使用）。

        Returns:
            是否成功加载。
        """
        if pkl_path is None:
            p = Path(model_dir) / "predict.pkl"
        else:
            p = Path(pkl_path)
        if not p.exists():
            return False

        with p.open("rb") as f:
            state = pickle.load(f)

        with self._lock:
            self._clf.coef_ = state["coef_"]
            self._clf.intercept_ = state["intercept_"]
            self._clf.classes_ = state["classes_"]
            self._scaler.mean_ = state["scaler_mean_"]
            self._scaler.scale_ = state["scaler_scale_"]
            self._n_samples = int(state["n_samples"])
            self._fitted = True

        return True