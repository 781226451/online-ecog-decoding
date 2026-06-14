"""实验配置。

所有可调参数集中在 :class:`ExperimentConfig`，便于在 ``run.py`` 中按需覆盖，
不必改动逻辑代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ActionDef:
    """一个动作类别的定义。

    Attributes:
        key:   程序内部使用的英文标识，同时用于在 ``media_dir`` 下按
               ``<key>.gif`` / ``<key>.mp4`` 约定查找素材。
        label: 屏幕上向患者展示的（中文）动作名称。
    """

    key: str
    label: str


@dataclass
class ExperimentConfig:
    """实验全局配置。"""

    # ---- 动作类别（解码类别数 = len(actions)）-------------------------------
    actions: list[ActionDef] = field(
        default_factory=lambda: [
            ActionDef("left_hand", "左手握拳"),
            ActionDef("right_hand", "右手握拳"),
            ActionDef("feet", "双脚背屈"),
            ActionDef("tongue", "伸舌"),
        ]
    )

    # ---- block / trial 结构 -------------------------------------------------
    n_blocks: int = 4
    trials_per_block: int = 8

    # ---- 各阶段时长（秒）----------------------------------------------------
    fixation_duration: float = 1.0   # 注视点 / 准备
    cue_duration: float = 4.0        # 动作执行/想象 + 采集 + 解码
    feedback_duration: float = 1.0   # 结果反馈停留
    rest_duration: float = 15.0      # block 间休息（期间后台训练）

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

    # ---- 媒体素材 -----------------------------------------------------------
    media_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "media")
    gif_frame_duration: float = 0.05  # gif 缺少帧时长信息时的回退帧间隔（秒）

    # ---- 杂项 ---------------------------------------------------------------
    random_seed: int | None = 0       # 模拟数据与试次顺序的可复现种子；None 表示不固定

    # --- 派生属性 ------------------------------------------------------------
    @property
    def n_classes(self) -> int:
        return len(self.actions)

    @property
    def action_labels(self) -> list[str]:
        return [a.label for a in self.actions]

    def validate(self) -> None:
        if self.n_classes < 2:
            raise ValueError("至少需要 2 个动作类别")
        if self.window_samples <= 0 or self.n_channels <= 0:
            raise ValueError("n_channels 与 window_samples 必须为正")
        if self.train_n_samples <= 0:
            raise ValueError("train_n_samples 必须为正")
