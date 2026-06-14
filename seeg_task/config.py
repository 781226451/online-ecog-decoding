"""实验配置。

所有可调参数集中在 :class:`ExperimentConfig`。既可在代码中直接构造，也可用
:func:`load_config` 从外部 TOML 文件加载（见仓库根目录 ``paradigm_config.toml``），便于在
不改动代码的前提下调整范式参数。优先级：默认值 < 配置文件 < 命令行参数。
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ActionDef:
    """一个动作类别：仅含屏幕展示的（中文）名称。"""

    label: str


@dataclass
class ExperimentConfig:
    """实验全局配置。"""

    # ---- 动作类别（解码类别数 = len(actions)）-------------------------------
    actions: list[ActionDef] = field(
        default_factory=lambda: [
            ActionDef("左手握拳"),
            ActionDef("右手握拳"),
            ActionDef("双脚背屈"),
            ActionDef("伸舌"),
        ]
    )

    # ---- block / trial 结构 -------------------------------------------------
    n_blocks: int = 4
    trials_per_block: int = 8

    # ---- 各阶段时长（秒）----------------------------------------------------
    cue_duration: float = 2.0        # CUE：全屏「当前动作为：{}」提示时长
    fixation_duration: float = 1.0   # FIXATION：盯点（白色十字）时长
    rest_duration: float = 15.0      # block 间休息（期间后台训练）
    predict_interval: float = 0.5    # EXECUTE 采集期内每隔多少秒做一次 predict

    # ---- 采集 / 训练 --------------------------------------------------------
    acquire_samples: int = 0         # EXECUTE 采集多少样本后结束；<=0 表示取 window_samples
    train_scope: str = "block"       # "block"（每 block 清空缓冲）| "cumulative"（跨 block 累积）

    # ---- 信号参数 -----------------------------------------------------------
    n_channels: int = 64             # SEEG 通道数
    sampling_rate: float = 1000.0    # 采样率 (Hz)
    window_samples: int = 2000       # 单个解码窗口的采样点数 -> x.shape=(n_channels, window_samples)

    # ---- 信号源选择 ---------------------------------------------------------
    source_type: str = "synthetic"   # "lsl" | "dummy" | "synthetic"
    # LSL（仅 source_type == "lsl" 时使用）
    lsl_stream_name: str | None = None   # 按流名解析；为 None 时改用 lsl_stream_type
    lsl_stream_type: str = "EEG"         # 按流类型解析（如 'EEG' / 'sEEG'）
    lsl_resolve_timeout: float = 5.0     # 解析流的超时（秒）

    # ---- 解码器 -------------------------------------------------------------
    # 解码器子类的完整导入路径（须为 decoder.BaseDecoder 的子类，并实现 from_config）。
    # 未在配置中给定时，默认使用 DummyDecoder（占位：随机概率、不训练，用于无模型联调）。
    decoder_class: str = "seeg_task.decoder.DummyDecoder"
    # 传给该子类 from_config 的额外参数（dict），按需自定义。
    decoder_params: dict = field(default_factory=dict)

    # ---- 在线模型更新 -------------------------------------------------------
    history_size: int = 256          # HistoryBuffer 容量 (最多保留多少历史样本)
    train_n_samples: int = 64        # 每个 block 后送入训练的历史样本数 n

    # ---- 显示 ---------------------------------------------------------------
    fullscreen: bool = False
    window_size: tuple[int, int] = (1280, 720)
    background_color: tuple[float, float, float] = (-0.2, -0.2, -0.2)  # PsychoPy rgb [-1,1]
    text_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # 中文字体：按顺序尝试，取第一个系统可用的。
    font_candidates: list[str] = field(
        default_factory=lambda: ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS", "SimHei"]
    )

    # ---- 日志 ---------------------------------------------------------------
    log_level: str = "INFO"           # 日志级别：DEBUG/INFO/WARNING/ERROR
    log_file: str | None = None       # 日志文件路径（相对则相对配置文件目录）；为空仅输出到控制台

    # ---- 杂项 ---------------------------------------------------------------
    random_seed: int | None = 0       # 模拟数据与试次顺序的可复现种子；None 表示不固定

    # --- 派生属性 ------------------------------------------------------------
    @property
    def n_classes(self) -> int:
        return len(self.actions)

    @property
    def action_labels(self) -> list[str]:
        return [a.label for a in self.actions]

    @property
    def effective_acquire_samples(self) -> int:
        """EXECUTE 实际采集样本数：``acquire_samples`` 为正则用之，否则取 ``window_samples``。"""
        return self.acquire_samples if self.acquire_samples > 0 else self.window_samples

    def validate(self) -> None:
        if self.n_classes < 2:
            raise ValueError("至少需要 2 个动作类别")
        if self.window_samples <= 0 or self.n_channels <= 0:
            raise ValueError("n_channels 与 window_samples 必须为正")
        if self.train_n_samples <= 0:
            raise ValueError("train_n_samples 必须为正")
        if self.predict_interval <= 0:
            raise ValueError("predict_interval 必须为正")
        if self.train_scope not in ("block", "cumulative"):
            raise ValueError("train_scope 仅支持 'block' 或 'cumulative'")
        if self.train_n_samples > self.history_size:
            print(
                f"[config] 警告：train_n_samples({self.train_n_samples}) > "
                f"history_size({self.history_size})，每次实际可用样本受限于 history_size"
            )

    # --- 从外部配置文件加载 --------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict, base_dir: Path | None = None) -> "ExperimentConfig":
        """由（TOML/JSON 解析得到的）字典构造配置；未提供的字段沿用默认值。

        会做必要的类型转换：``actions`` 列表（字符串或 ``{label}`` 表）-> :class:`ActionDef`；
        颜色/窗口尺寸列表 -> 元组；``log_file`` 相对路径相对 ``base_dir`` 解析；``random_seed``
        为负数视作 ``None``（不固定种子）。未知字段会被忽略并告警。
        """
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs: dict = {}
        for key, value in data.items():
            if key not in known:
                print(f"[config] 忽略未知参数: {key!r}")
                continue
            kwargs[key] = value

        if "actions" in kwargs:
            def _to_action(a):
                if isinstance(a, ActionDef):
                    return a
                if isinstance(a, str):
                    return ActionDef(a)
                return ActionDef(**a)
            kwargs["actions"] = [_to_action(a) for a in kwargs["actions"]]
        for tup_field in ("window_size", "background_color", "text_color"):
            if tup_field in kwargs and isinstance(kwargs[tup_field], list):
                kwargs[tup_field] = tuple(kwargs[tup_field])
        if kwargs.get("log_file"):
            lf = Path(kwargs["log_file"])
            if not lf.is_absolute() and base_dir is not None:
                lf = (base_dir / lf).resolve()
            kwargs["log_file"] = str(lf)
        if kwargs.get("random_seed") is not None and kwargs["random_seed"] < 0:
            kwargs["random_seed"] = None

        cfg = cls(**kwargs)
        cfg.validate()
        return cfg


def load_config(path: str | Path) -> ExperimentConfig:
    """从 TOML 文件加载 :class:`ExperimentConfig`。相对路径（如 log_file）相对该文件所在目录解析。"""
    path = Path(path)
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return ExperimentConfig.from_dict(data, base_dir=path.parent)
