"""实验编排：构造组件并用两层 FSM（:mod:`seeg_task.fsm`）驱动范式运行。

`Experiment` 负责装配（信号源 / 解码器 / 缓冲 / UI）与引导页、总结页；block-trial 的
状态流转交给 :class:`~seeg_task.fsm.BlockFSM`（其内部运行 :class:`~seeg_task.fsm.TrialFSM`）。
"""

from __future__ import annotations

import numpy as np
from psychopy import core

from .buffer import BlockBuffer
from .config import ExperimentConfig
from .decoder import BaseDecoder, create_decoder
from .fsm import BlockFSM, QuitExperiment
from .signal_source import SignalSource, create_source
from .ui import ExperimentUI


class Experiment:
    """把信号源、解码器、缓冲、FSM 与 UI 串联成完整范式。"""

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        source: SignalSource | None = None,
        decoder: BaseDecoder | None = None,
    ) -> None:
        self.config = config or ExperimentConfig()
        self.config.validate()
        cfg = self.config

        self.rng = np.random.default_rng(cfg.random_seed)
        self.source = source or create_source(cfg, rng=self.rng)
        self.decoder = decoder or create_decoder(cfg, rng=self.rng)
        # current_item 窗口大小 = 解码窗口长度；EXECUTE 流式推入 acquire_samples 个样本
        self.buffer = BlockBuffer(cfg.n_channels, cfg.window_samples)

        self.ui: ExperimentUI | None = None

    def run(self) -> None:
        cfg = self.config
        self.ui = ExperimentUI(cfg)
        ui = self.ui
        try:
            ui.draw_message(
                "SEEG 脑机接口任务\n\n"
                "每个 trial：先全屏提示动作，再盯点，然后执行/想象该动作。\n"
                "右侧实时显示解码正确率。\n\n"
                "按 空格 开始，按 Esc 随时退出。"
            )
            ui.flip()
            keys = ui.wait_keys(keys=["space", "escape"])
            if "escape" in keys:
                raise QuitExperiment

            BlockFSM(cfg, self.source, self.decoder, self.buffer, ui, self.rng).run()

            self._show_summary()
        except QuitExperiment:
            ui.draw_message("已退出实验。")
            ui.flip()
            core.wait(0.8)
        finally:
            ui.close()
            self.source.close()

    def _show_summary(self) -> None:
        ui = self.ui
        pct = 100.0 * ui.total_correct / ui.total_trials if ui.total_trials else 0.0
        ui.draw_message(
            "实验结束\n\n"
            f"总体解码正确率：{pct:.1f}%  ({ui.total_correct}/{ui.total_trials})\n\n"
            "按任意键退出。"
        )
        ui.flip()
        ui.wait_keys()
