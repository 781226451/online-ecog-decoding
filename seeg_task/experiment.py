"""实验编排：block-trial 主循环 + 休息期后台训练。"""

from __future__ import annotations

import threading

import numpy as np
from psychopy import core

from .config import ExperimentConfig
from .decoder import Decoder, LinearModel
from .model_update import HistoryBuffer, ModelTrainer
from .signal_source import SignalSource, SyntheticSource
from .ui import ExperimentUI


class QuitExperiment(Exception):
    """用户按下 Esc 请求退出。"""


class Experiment:
    """把信号源、解码器、在线训练与 UI 串联成完整范式。"""

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        source: SignalSource | None = None,
        decoder: Decoder | None = None,
        trainer: ModelTrainer | None = None,
    ) -> None:
        self.config = config or ExperimentConfig()
        self.config.validate()
        cfg = self.config

        self.rng = np.random.default_rng(cfg.random_seed)
        n_features = cfg.n_channels  # extract_features 输出长度 = 通道数

        self.source = source or SyntheticSource(
            cfg.n_channels, cfg.window_samples, cfg.n_classes, rng=self.rng
        )
        self.decoder = decoder or Decoder(
            LinearModel.random_init(cfg.n_classes, n_features, self.rng), cfg.n_classes
        )
        self.trainer = trainer or ModelTrainer(cfg.n_classes, n_features)
        self.history = HistoryBuffer(cfg.history_size)

        self.ui: ExperimentUI | None = None

    # --- 通用时间循环 -------------------------------------------------------
    def _run_for(self, duration: float, draw_fn) -> None:
        """以一个固定时长循环绘制；每帧调用 ``draw_fn()`` 后 flip 并检测退出。"""
        clock = core.Clock()
        while clock.getTime() < duration:
            draw_fn()
            self.ui.flip()
            if self.ui.quit_requested():
                raise QuitExperiment

    # --- 单次 trial ---------------------------------------------------------
    def _run_trial(self, action_index: int) -> None:
        ui = self.ui
        cfg = self.config

        # 1) 注视/准备
        self._run_for(cfg.fixation_duration, lambda: ui.draw_fixation(action_index))

        # 2) 动作执行/想象（采集期）
        ui.start_trial_media(action_index)
        self._run_for(cfg.cue_duration, lambda: ui.draw_cue(action_index))
        ui.stop_trial_media(action_index)

        # 3) 采集窗口 -> 解码
        x = self.source.read_window(true_label=action_index)  # (n_channels, window_samples)
        probs = self.decoder.predict(x)                        # (n_classes,)
        predicted = int(np.argmax(probs))
        correct = ui.record_result(predicted, action_index, probs)

        # 4) 入历史缓冲（供 block 后训练）
        self.history.add(x, action_index)

        # 5) 反馈
        self._run_for(
            cfg.feedback_duration, lambda: ui.draw_feedback(action_index, correct, predicted)
        )

    # --- block 顺序 ---------------------------------------------------------
    def _block_order(self) -> list[int]:
        """生成一个 block 内平衡且打乱的动作标签序列。"""
        cfg = self.config
        reps = int(np.ceil(cfg.trials_per_block / cfg.n_classes))
        seq = np.tile(np.arange(cfg.n_classes), reps)[: cfg.trials_per_block]
        self.rng.shuffle(seq)
        return seq.tolist()

    # --- 休息 + 后台训练 ----------------------------------------------------
    def _rest_and_train(self) -> None:
        ui = self.ui
        cfg = self.config

        samples = self.history.recent(cfg.train_n_samples)
        result: dict[str, LinearModel | None] = {}
        error: dict[str, BaseException] = {}

        def worker() -> None:
            try:
                result["model"] = self.trainer.train(samples)
            except BaseException as exc:  # noqa: BLE001 - 把异常带回主线程展示
                error["err"] = exc

        thread = threading.Thread(target=worker, name="model-trainer", daemon=True)
        thread.start()

        applied = False
        status = "正在更新模型…"
        clock = core.Clock()
        while clock.getTime() < cfg.rest_duration:
            remaining = cfg.rest_duration - clock.getTime()
            if not applied and not thread.is_alive():
                status = self._apply_trained_model(result, error)
                applied = True
            ui.draw_rest(remaining, status)
            ui.flip()
            if ui.quit_requested():
                raise QuitExperiment

        # 休息结束仍未训完（极少见）：阻塞等待再热替换，保证模型确有更新。
        if not applied:
            thread.join()
            self._apply_trained_model(result, error)

    def _apply_trained_model(self, result: dict, error: dict) -> str:
        if "err" in error:
            print(f"[训练] 失败: {error['err']}")
            return "模型更新失败，沿用当前模型"
        new_model = result.get("model")
        if new_model is None:
            return "样本不足，本次跳过更新"
        self.decoder.swap_model(new_model)
        return "模型已更新 ✓"

    # --- 入口 ---------------------------------------------------------------
    def run(self) -> None:
        cfg = self.config
        self.ui = ExperimentUI(cfg)
        ui = self.ui
        try:
            ui.draw_message(
                "SEEG 脑机接口任务\n\n"
                "左侧将提示动作并播放示范，请按提示执行/想象对应动作。\n"
                "右侧实时显示解码正确率。\n\n"
                "按 空格 开始，按 Esc 随时退出。"
            )
            ui.flip()
            keys = ui.wait_keys(keys=["space", "escape"])
            if "escape" in keys:
                raise QuitExperiment

            for block in range(cfg.n_blocks):
                for action_index in self._block_order():
                    self._run_trial(action_index)

                if block < cfg.n_blocks - 1:
                    self._rest_and_train()

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
