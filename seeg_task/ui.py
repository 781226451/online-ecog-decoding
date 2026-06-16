"""PsychoPy 可视化界面。

布局（units = 'height'）::

    +---------------------------+---------------------------+
    |        左面板             |        右面板             |
    |   动作名称（中文）        |   实时解码正确率          |
    +---------------------------+---------------------------+

``ExperimentUI`` 只提供「绘制单帧」级别的方法（draw_* 不调用 flip），由
:mod:`seeg_task.fsm` 控制各阶段的时间循环与 :meth:`flip`，从而把时序逻辑集中在编排里。
"""

from __future__ import annotations

import numpy as np
from psychopy import core, event, visual


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

        self.current_action: int | None = None
        self._cue_prob: float | None = None

        self._build_static_stims()

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

        # 左面板：动作名称（在左半区居中）
        self.action_title = self._text(
            text="", height=0.09, pos=(self._left_x, 0.0), bold=True,
        )

        # 右面板：实时解码结果
        self.acc_title = self._text(
            text="准确率", height=0.06, pos=(self._right_x, 0.12), bold=True,
        )
        self.acc_value = self._text(text="--", height=0.18, pos=(self._right_x, -0.02), bold=True)
        self.pred_prob = self._text(text="", height=0.05, pos=(self._right_x, -0.22))

        # 全屏 CUE 文字 与 全屏盯点白色十字（trial 级 FSM 使用）
        self.cue_text = self._text(text="", height=0.09, pos=(0, 0), bold=True,
                                   wrapWidth=self._half_w * 1.8)
        self.fixation_cross = self._text(text="+", height=0.22, pos=(0, 0), color=(1.0, 1.0, 1.0))

        # 全屏提示用文字
        self.center_text = self._text(text="", height=0.05, pos=(0, 0), wrapWidth=self._half_w * 1.6)
        self.rest_title = self._text(text="", height=0.07, pos=(0, 0.10), bold=True)
        self.rest_prompt = self._text(text="", height=0.05, pos=(0, -0.12))

    # --- 统计更新 -----------------------------------------------------------
    def reset_trial_stats(self) -> None:
        """每个 trial 的 EXECUTE 开始时调用，清空上一个 trial 残留的概率显示。"""
        self._cue_prob = None

    def record_result(self, true_label: int, probs: np.ndarray | None = None) -> None:
        """记录本次预测：右面板显示真实类别（true_label）的预测概率。"""
        self._cue_prob = float(probs[true_label]) if probs is not None else None

    # --- 绘制（不 flip）-----------------------------------------------------
    def draw_right_panel(self) -> None:
        """右面板：显示给定动作的实时概率值。"""
        self.acc_value.text = f"{self._cue_prob * 100:.1f}%" if self._cue_prob is not None else "--"
        self.acc_title.draw()
        self.acc_value.draw()
        self.pred_prob.draw()

    def draw_divider(self) -> None:
        self.divider.draw()

    def draw_cue(self, action_index: int) -> None:
        """EXECUTE 阶段单帧：左侧动作名 + 右侧实时正确率 + 分隔线。"""
        self.current_action = action_index
        self.action_title.text = self.config.actions[action_index].label
        self.draw_divider()
        self.action_title.draw()
        self.draw_right_panel()

    def draw_fixation(self, action_index: int | None = None) -> None:
        self.current_action = action_index
        self.draw_divider()
        if action_index is not None:
            self.action_title.text = self.config.actions[action_index].label
            self.action_title.draw()
        self.fixation.draw()
        self.draw_right_panel()

    def draw_rest(self, prompt: str = "") -> None:
        # 休息界面：标题“请休息” + 可选第二行提示（不显示任何模型训练文字）
        self.rest_title.text = "请休息"
        self.rest_title.draw()
        if prompt:
            self.rest_prompt.text = prompt
            self.rest_prompt.draw()

    def draw_cue_text(self, label: str) -> None:
        """CUE 阶段：全屏显示「当前动作为：{label}」。"""
        self.cue_text.text = f"当前动作为：{label}"
        self.cue_text.draw()

    def draw_fixation_cross(self) -> None:
        """FIXATION 阶段：全屏居中白色十字。"""
        self.fixation_cross.draw()

    def draw_message(self, text: str) -> None:
        self.center_text.text = text
        self.center_text.draw()

    # --- 时序/输入 ----------------------------------------------------------
    def flip(self) -> float:
        return self.win.flip()

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
        try:
            self.win.close()
        except Exception:  # noqa: BLE001
            pass
