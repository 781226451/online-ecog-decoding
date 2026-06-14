"""PsychoPy 可视化界面。

布局（units = 'height'）::

    +---------------------------+---------------------------+
    |        左面板             |        右面板             |
    |   动作名称（中文）        |   实时解码正确率          |
    |   动作 gif / 视频         |   大号百分比 + 试次计数   |
    |                           |   各类别概率条            |
    +---------------------------+---------------------------+

``ExperimentUI`` 只提供「绘制单帧」级别的方法（draw_* 不调用 flip），由
:mod:`seeg_task.experiment` 控制各阶段的时间循环与 :meth:`flip`，从而把时序逻辑
集中在实验编排里。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from psychopy import core, event, visual

# 支持的媒体扩展名
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _resolve_font(candidates: list[str]) -> str:
    """返回 ``candidates`` 中系统已安装的第一个字体，找不到则返回第一个候选。"""
    try:
        from matplotlib import font_manager

        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in available:
                return name
    except Exception:  # noqa: BLE001 - 字体探测失败不应阻塞实验
        pass
    return candidates[0] if candidates else "Arial"


class MediaPlayer:
    """加载并逐帧绘制 gif / 视频 / 静态图。

    - 视频 (.mp4/.mov/...) 使用 :class:`psychopy.visual.MovieStim`，循环播放。
    - gif 使用 imageio 预解码为帧序列，按帧时长用内部时钟循环。
    - 静态图使用 :class:`psychopy.visual.ImageStim`。
    """

    def __init__(
        self,
        win: visual.Window,
        path: Path,
        size: tuple[float, float],
        pos: tuple[float, float],
        gif_fallback_dt: float,
    ) -> None:
        self.win = win
        self.path = path
        self.size = size
        self.pos = pos
        self.kind = "none"
        self._stim = None
        self._frames: list = []
        self._frame_dt = gif_fallback_dt
        self._clock = core.Clock()

        ext = path.suffix.lower()
        try:
            if ext in _VIDEO_EXTS:
                self._init_video()
            elif ext == ".gif":
                self._init_gif(gif_fallback_dt)
            elif ext in _IMAGE_EXTS:
                self._init_image()
        except Exception as exc:  # noqa: BLE001 - 素材损坏时退化为无媒体
            print(f"[MediaPlayer] 加载素材失败 {path}: {exc}")
            self.kind = "none"
            self._stim = None

    # --- 初始化各类型 -------------------------------------------------------
    def _init_video(self) -> None:
        self._stim = visual.MovieStim(
            self.win, str(self.path), size=self.size, pos=self.pos,
            units="height", loop=True, noAudio=True,
        )
        self.kind = "video"

    def _init_gif(self, fallback_dt: float) -> None:
        import imageio.v2 as imageio

        reader = imageio.get_reader(str(self.path))
        meta = reader.get_meta_data()
        duration = meta.get("duration")  # 毫秒，可能为标量或列表
        if isinstance(duration, (list, tuple)) and duration:
            self._frame_dt = float(np.mean(duration)) / 1000.0
        elif isinstance(duration, (int, float)) and duration:
            self._frame_dt = float(duration) / 1000.0
        else:
            self._frame_dt = fallback_dt

        frames = [np.asarray(f) for f in reader]
        reader.close()
        # 预创建每帧 ImageStim（gif 一般较短，可接受）。
        for frame in frames:
            tex = self._to_texture(frame)
            self._frames.append(
                visual.ImageStim(self.win, image=tex, size=self.size, pos=self.pos, units="height")
            )
        if self._frames:
            self.kind = "gif"

    def _init_image(self) -> None:
        self._stim = visual.ImageStim(
            self.win, image=str(self.path), size=self.size, pos=self.pos, units="height"
        )
        self.kind = "image"

    @staticmethod
    def _to_texture(frame: np.ndarray) -> np.ndarray:
        """uint8 RGB(A) 帧 -> PsychoPy 纹理（float, [-1,1], 取 RGB）。"""
        arr = frame.astype(np.float64)
        if arr.ndim == 2:  # 灰度 -> RGB
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        arr = arr[:, :, :3]
        return arr / 127.5 - 1.0

    # --- 播放控制 -----------------------------------------------------------
    def reset(self) -> None:
        """重置到起始帧并开始计时（每个 trial 开始时调用）。"""
        self._clock.reset()
        if self.kind == "video" and self._stim is not None:
            try:
                self._stim.seek(0)
                self._stim.play()
            except Exception:  # noqa: BLE001
                pass

    def draw(self) -> None:
        if self.kind == "video" and self._stim is not None:
            self._stim.draw()
        elif self.kind == "gif" and self._frames:
            idx = int(self._clock.getTime() / self._frame_dt) % len(self._frames)
            self._frames[idx].draw()
        elif self.kind == "image" and self._stim is not None:
            self._stim.draw()

    def stop(self) -> None:
        if self.kind == "video" and self._stim is not None:
            try:
                self._stim.pause()
            except Exception:  # noqa: BLE001
                pass

    def unload(self) -> None:
        if self.kind == "video" and self._stim is not None:
            try:
                self._stim.stop()
            except Exception:  # noqa: BLE001
                pass


class ExperimentUI:
    """实验可视化界面。"""

    def __init__(self, config) -> None:
        self.config = config
        self.font = _resolve_font(config.font_candidates)

        self.win = visual.Window(
            size=config.window_size,
            fullscr=config.fullscreen,
            color=config.background_color,
            units="height",
            allowGUI=True,
            winType="pyglet",
        )
        # macOS 上窗口常常打开后不获得键盘焦点，导致 space/esc 收不到——主动激活窗口。
        try:
            self.win.winHandle.activate()
        except Exception:  # noqa: BLE001
            pass

        aspect = self.win.size[0] / self.win.size[1]
        self._half_w = aspect / 2.0          # 屏幕半宽（height 单位）
        self._left_x = -aspect / 4.0         # 左面板中心 x
        self._right_x = aspect / 4.0         # 右面板中心 x

        # 运行统计：按动作分别累计分类正确率（整个实验过程累计）
        self.total_correct = 0
        self.total_trials = 0
        self.current_action: int | None = None
        self.action_correct = np.zeros(config.n_classes, dtype=int)
        self.action_total = np.zeros(config.n_classes, dtype=int)

        self._build_static_stims()
        self._build_media_players()

    # --- 构建 ---------------------------------------------------------------
    def _text(self, **kwargs) -> visual.TextStim:
        kwargs.setdefault("font", self.font)
        kwargs.setdefault("color", self.config.text_color)
        return visual.TextStim(self.win, **kwargs)

    def _build_static_stims(self) -> None:
        cfg = self.config
        # 中央分隔线
        self.divider = visual.Line(
            self.win, start=(0, -0.5), end=(0, 0.5), units="height",
            lineColor=(0.3, 0.3, 0.3), lineWidth=2,
        )
        # 注视点
        self.fixation = self._text(text="+", height=0.12, pos=(self._left_x, 0))

        # 左面板：动作名称
        self.action_title = self._text(
            text="", height=0.07, pos=(self._left_x, 0.40), bold=True,
        )
        # 左面板：无素材时的占位框 + 文字
        self.media_placeholder = visual.Rect(
            self.win, width=min(0.55, self._half_w * 0.8), height=0.4,
            pos=(self._left_x, -0.02), units="height",
            lineColor=(0.35, 0.35, 0.35), fillColor=None,
        )
        self.placeholder_text = self._text(
            text="(无动作素材)", height=0.04, pos=(self._left_x, -0.02),
            color=(0.4, 0.4, 0.4),
        )
        # 左面板：反馈文字
        self.feedback_text = self._text(text="", height=0.06, pos=(self._left_x, -0.42))

        # 右面板：当前动作的分类正确率
        self.acc_title = self._text(
            text="当前动作正确率", height=0.055, pos=(self._right_x, 0.30), bold=True,
        )
        self.acc_action = self._text(text="", height=0.06, pos=(self._right_x, 0.14))
        self.acc_value = self._text(text="--%", height=0.22, pos=(self._right_x, -0.06), bold=True)
        self.acc_detail = self._text(text="", height=0.04, pos=(self._right_x, -0.26))

        # 全屏提示用文字
        self.center_text = self._text(text="", height=0.05, pos=(0, 0), wrapWidth=self._half_w * 1.6)
        self.rest_title = self._text(text="", height=0.07, pos=(0, 0.18), bold=True)
        self.rest_status = self._text(text="", height=0.045, pos=(0, -0.05))
        self.rest_count = self._text(text="", height=0.06, pos=(0, -0.22))

    def _build_media_players(self) -> None:
        cfg = self.config
        media_size = (min(0.6, self._half_w * 0.85), 0.42)
        media_pos = (self._left_x, -0.02)
        self.players: dict[int, MediaPlayer | None] = {}
        for i, action in enumerate(cfg.actions):
            path = self._find_media(action.key)
            if path is None:
                self.players[i] = None
            else:
                self.players[i] = MediaPlayer(self.win, path, media_size, media_pos, cfg.gif_frame_duration)

    def _find_media(self, key: str) -> Path | None:
        media_dir: Path = self.config.media_dir
        if not media_dir.exists():
            return None
        for ext in [".gif", *_VIDEO_EXTS, *_IMAGE_EXTS]:
            candidate = media_dir / f"{key}{ext}"
            if candidate.exists():
                return candidate
        return None

    # --- 统计更新 -----------------------------------------------------------
    def record_result(self, predicted: int, true_label: int, probs: np.ndarray | None = None) -> bool:
        correct = int(predicted) == int(true_label)
        self.total_trials += 1
        self.action_total[true_label] += 1
        if correct:
            self.total_correct += 1
            self.action_correct[true_label] += 1
        return correct

    # --- 绘制（不 flip）-----------------------------------------------------
    def draw_right_panel(self) -> None:
        """右面板：仅显示「当前动作」的累计分类正确率。"""
        i = self.current_action
        if i is not None:
            self.acc_action.text = self.config.actions[i].label
            total = int(self.action_total[i])
            if total:
                pct = 100.0 * self.action_correct[i] / total
                self.acc_value.text = f"{pct:.0f}%"
            else:
                self.acc_value.text = "--%"
            self.acc_detail.text = f"正确 {int(self.action_correct[i])} / {total} 次"
        else:
            self.acc_action.text = ""
            self.acc_value.text = "--%"
            self.acc_detail.text = ""
        self.acc_title.draw()
        self.acc_action.draw()
        self.acc_value.draw()
        self.acc_detail.draw()

    def draw_divider(self) -> None:
        self.divider.draw()

    def draw_left_media(self, action_index: int) -> None:
        player = self.players.get(action_index)
        if player is not None and player.kind != "none":
            player.draw()
        else:
            self.media_placeholder.draw()
            self.placeholder_text.draw()

    def draw_cue(self, action_index: int) -> None:
        """绘制左侧动作提示（名称 + 媒体）+ 右侧面板 + 分隔线（单帧）。"""
        self.current_action = action_index
        self.action_title.text = self.config.actions[action_index].label
        self.draw_divider()
        self.action_title.draw()
        self.draw_left_media(action_index)
        self.draw_right_panel()

    def draw_fixation(self, action_index: int | None = None) -> None:
        self.current_action = action_index
        self.draw_divider()
        if action_index is not None:
            self.action_title.text = self.config.actions[action_index].label
            self.action_title.draw()
        self.fixation.draw()
        self.draw_right_panel()

    def draw_feedback(self, action_index: int, correct: bool, predicted_index: int) -> None:
        self.current_action = action_index
        self.action_title.text = self.config.actions[action_index].label
        if correct:
            self.feedback_text.text = "✓ 解码正确"
            self.feedback_text.color = (0.1, 0.8, 0.2)
        else:
            pred_label = self.config.actions[predicted_index].label
            self.feedback_text.text = f"✗ 解码为：{pred_label}"
            self.feedback_text.color = (0.9, 0.3, 0.2)
        self.draw_divider()
        self.action_title.draw()
        self.draw_left_media(action_index)
        self.feedback_text.draw()
        self.draw_right_panel()

    def draw_rest(self, remaining: float, status: str) -> None:
        self.rest_title.text = "请休息"
        self.rest_status.text = status
        self.rest_count.text = f"{remaining:0.0f} s"
        self.rest_title.draw()
        self.rest_status.draw()
        self.rest_count.draw()

    def draw_message(self, text: str) -> None:
        self.center_text.text = text
        self.center_text.draw()

    # --- 时序/输入 ----------------------------------------------------------
    def flip(self) -> float:
        return self.win.flip()

    def start_trial_media(self, action_index: int) -> None:
        player = self.players.get(action_index)
        if player is not None:
            player.reset()

    def stop_trial_media(self, action_index: int) -> None:
        player = self.players.get(action_index)
        if player is not None:
            player.stop()

    def _pump(self) -> None:
        """主动派发窗口事件队列（pyglet），确保按键被捕获。"""
        try:
            self.win.winHandle.dispatch_events()
        except Exception:  # noqa: BLE001
            pass

    def quit_requested(self) -> bool:
        return "escape" in event.getKeys(keyList=["escape"])

    def wait_keys(self, keys: list[str] | None = None) -> list[str]:
        """等待按键。显式 pump + 轮询，不依赖 waitKeys 内部的事件泵时机。"""
        event.clearEvents()
        while True:
            self._pump()
            pressed = event.getKeys(keyList=keys)
            if pressed:
                return pressed
            core.wait(0.005)

    def close(self) -> None:
        for player in getattr(self, "players", {}).values():
            if player is not None:
                player.unload()
        try:
            self.win.close()
        except Exception:  # noqa: BLE001
            pass
