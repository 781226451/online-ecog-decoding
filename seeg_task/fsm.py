"""范式运行的两层有限状态机（FSM）。

- :class:`TrialFSM`：单个 trial 的 ``CUE → FIXATION → EXECUTE → FINISH`` 流程。
  仅 EXECUTE 为采集态——计数器与缓存（:class:`~seeg_task.buffer.BlockBuffer`）只在此运行；
  进入 EXECUTE 清单次缓存，FINISH 入口把窗口+label 存档。
- :class:`BlockFSM`：``TRIAL → TRAIN_REST → FINISH``，驱动 trial 循环与 block 末后台训练。

设计见 ``docs/fsm_design.md``。
"""

from __future__ import annotations

import threading
from enum import Enum, auto

import numpy as np
from loguru import logger
from psychopy import core


class QuitExperiment(Exception):
    """用户按下 Esc 请求退出（视为强制转移到终态）。"""


class TrialState(Enum):
    CUE = auto()
    FIXATION = auto()
    EXECUTE = auto()
    FINISH = auto()


class BlockState(Enum):
    TRIAL = auto()
    TRAIN_REST = auto()
    FINISH = auto()


def _run_for(ui, duration: float, draw_fn) -> None:
    """以固定时长逐帧绘制；每帧 ``draw_fn()`` 后 flip 并检测退出。"""
    clock = core.Clock()
    while clock.getTime() < duration:
        draw_fn()
        ui.flip()
        if ui.quit_requested():
            raise QuitExperiment


class TrialFSM:
    """单个 trial 的状态机：CUE → FIXATION → EXECUTE → FINISH。"""

    def __init__(self, config, source, decoder, buffer, ui) -> None:
        self.config = config
        self.source = source
        self.decoder = decoder
        self.buffer = buffer
        self.ui = ui
        self.acquire_samples = config.effective_acquire_samples

    def run(self, action_index: int) -> None:
        self._cue(action_index)
        self._fixation()
        self._execute(action_index)
        self._finish(action_index)

    # --- CUE：全屏「当前动作为：{}」---------------------------------------
    def _cue(self, action_index: int) -> None:
        label = self.config.actions[action_index].label
        _run_for(self.ui, self.config.cue_duration, lambda: self.ui.draw_cue_text(label))

    # --- FIXATION：全屏白色十字 -------------------------------------------
    def _fixation(self) -> None:
        _run_for(self.ui, self.config.fixation_duration, self.ui.draw_fixation_cross)

    # --- EXECUTE：流式采集 + 实时解码 -------------------------------------
    def _execute(self, action_index: int) -> None:
        cfg, ui = self.config, self.ui
        self.buffer.reset_current_item()  # 进入 EXECUTE：清单次缓存
        counter = 0
        fs = cfg.sampling_rate
        run_clock = core.Clock()
        tick = core.Clock()
        first = True
        while counter < self.acquire_samples:
            # 按采样率节流：本帧应已到达的样本数
            target = self.acquire_samples if fs <= 0 else min(
                self.acquire_samples, int(run_clock.getTime() * fs)
            )
            while counter < target:
                sample = self.source.read_sample(true_label=action_index)
                if sample is None:  # LSL 暂无新样本
                    break
                self.buffer.update_current_item(sample)
                counter += 1
            # 定时实时解码 + 刷新正确率（至少一次）
            if first or tick.getTime() >= cfg.predict_interval:
                first = False
                tick.reset()
                probs = self.decoder.predict(self.buffer.current_item)
                ui.record_result(int(np.argmax(probs)), action_index, probs)
            ui.draw_cue(action_index)  # 动作名 + 右侧实时正确率
            ui.flip()
            if ui.quit_requested():
                raise QuitExperiment

    # --- FINISH：存档（trial 收尾）---------------------------------------
    def _finish(self, action_index: int) -> None:
        self.buffer.update_buffer(action_index)  # 窗口+label 入 block 存档


class BlockFSM:
    """block 级状态机：TRIAL → TRAIN_REST → FINISH，驱动整场实验（不含引导/总结页）。"""

    def __init__(self, config, source, decoder, buffer, ui, rng) -> None:
        self.config = config
        self.source = source
        self.decoder = decoder
        self.buffer = buffer
        self.ui = ui
        self.rng = rng
        self.trial = TrialFSM(config, source, decoder, buffer, ui)

    def run(self) -> None:
        cfg = self.config
        block = 0
        self._begin_block()
        order = self._block_order()
        trial_idx = 0
        state = BlockState.TRIAL
        while True:
            if state == BlockState.TRIAL:
                if trial_idx < cfg.trials_per_block:
                    action_index = order[trial_idx]
                    label = cfg.actions[action_index].label
                    # 绑定上下文：后续日志（含 EXECUTE 期 predict）都带 blk/trial/act
                    with logger.contextualize(block=block + 1, trial=trial_idx + 1, action=label):
                        logger.info("trial 开始")
                        self.trial.run(action_index)
                    trial_idx += 1
                else:
                    state = BlockState.TRAIN_REST
            elif state == BlockState.TRAIN_REST:
                if block < cfg.n_blocks - 1:
                    with logger.contextualize(block=block + 1, trial="-", action="train"):
                        self._rest_and_train()
                    block += 1
                    self._begin_block()
                    order = self._block_order()
                    trial_idx = 0
                    state = BlockState.TRIAL
                else:
                    state = BlockState.FINISH
            else:  # FINISH
                break

    # --- block 起始动作 ---------------------------------------------------
    def _begin_block(self) -> None:
        if self.config.train_scope == "block":
            self.buffer.clean()            # 每 block 清空（仅用本 block 样本训练）
        else:
            self.buffer.reset_current_item()  # cumulative：保留 items，仅清单次窗口

    def _block_order(self) -> list[int]:
        cfg = self.config
        reps = int(np.ceil(cfg.trials_per_block / cfg.n_classes))
        seq = np.tile(np.arange(cfg.n_classes), reps)[: cfg.trials_per_block]
        self.rng.shuffle(seq)
        return seq.tolist()

    # --- 休息 + 后台训练 --------------------------------------------------
    def _rest_and_train(self) -> None:
        ui, cfg = self.ui, self.config
        samples = list(self.buffer.items)  # 本 block（或累积）样本快照
        result: dict[str, bool] = {}
        error: dict[str, BaseException] = {}

        def worker() -> None:
            try:
                result["updated"] = self.decoder.update(samples)
            except BaseException as exc:  # noqa: BLE001
                error["err"] = exc

        thread = threading.Thread(target=worker, name="model-trainer", daemon=True)
        thread.start()

        applied = False
        clock = core.Clock()
        while clock.getTime() < cfg.rest_duration:
            remaining = cfg.rest_duration - clock.getTime()
            if not applied and not thread.is_alive():
                self._log_update_result(result, error)
                applied = True
            ui.draw_rest(remaining)
            ui.flip()
            if ui.quit_requested():
                raise QuitExperiment

        if not applied:  # 极少见：休息结束仍未训完，阻塞等待
            thread.join()
            self._log_update_result(result, error)

    def _log_update_result(self, result: dict, error: dict) -> None:
        if "err" in error:
            logger.error("模型更新失败: {}", error["err"])
        elif result.get("updated"):
            logger.info("模型已更新（本 block 样本 {} 条）", len(self.buffer.items))
        else:
            # update 返回 False：可能是解码器不训练（如 DummyDecoder）或样本不足
            logger.info("本次未更新模型（解码器跳过；样本 {} 条）", len(self.buffer.items))
