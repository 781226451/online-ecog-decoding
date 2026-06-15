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
from pathlib import Path

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

    # --- EXECUTE：按时长计时，定时推理；到点进 FINISH --------------------
    def _execute(self, action_index: int) -> None:
        cfg, ui = self.config, self.ui
        self.source.flush()               # 丢弃 CUE/FIXATION 期积压，只采执行期数据
        self.buffer.reset_window()  # 进入 EXECUTE：清单次缓存
        exec_clock = core.Clock()         # 执行期计时：到 execute_duration 即结束
        tick = core.Clock()               # 推理计时：每 predict_interval 推理一次
        while exec_clock.getTime() < cfg.execute_duration:
            chunk = self.source.read()  # 取走全部可用数据，喂入滑动窗口
            if chunk is not None:
                self.buffer.update_current_items(chunk)
            # 定时推理：对整个 current_item 解码并刷新正确率
            if tick.getTime() >= cfg.predict_interval:
                tick.reset()
                item = self.buffer.record_predict()
                probs = self.decoder.predict(item)
                pred = int(np.argmax(probs))
                logger.info("predict | x={} -> {}({}) p={:.3f} probs={}",
                            tuple(item.shape), pred,
                            cfg.actions[pred].label, float(probs[pred]),
                            [round(float(p), 4) for p in probs])
                ui.record_result(pred, action_index, probs)
            ui.draw_cue(action_index)  # 动作名 + 右侧实时正确率
            ui.flip()
            if ui.quit_requested():
                raise QuitExperiment
        # 退出循环即进入 FINISH（执行期计时器随之失效）

    # --- FINISH：存档（trial 收尾）---------------------------------------
    def _finish(self, action_index: int) -> None:
        self.buffer.save_sample(action_index)  # 窗口+label 入 block 存档


class BlockFSM:
    """block 级状态机：TRIAL → TRAIN_REST → FINISH，驱动整场实验（不含引导/总结页）。"""

    def __init__(self, config, source, decoder, buffer, ui, rng,
                 session_dir: Path | None = None) -> None:
        self.config = config
        self.source = source
        self.decoder = decoder
        self.buffer = buffer
        self.ui = ui
        self.rng = rng
        self.session_dir = session_dir
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
                    # trial 间隔（ITI）：仅在 trial 之间，最后一个 trial 后不做（直接进 block 休息）
                    if trial_idx < cfg.trials_per_block and cfg.iti_duration > 0:
                        _run_for(self.ui, cfg.iti_duration, lambda: None)
                else:
                    state = BlockState.TRAIN_REST
            elif state == BlockState.TRAIN_REST:
                if block < cfg.n_blocks - 1:
                    self._save_block_data(block)
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
                self._save_block_data(block)
                break

    # --- block 数据保存 ---------------------------------------------------
    def _save_block_data(self, block: int) -> None:
        base = self.session_dir if self.session_dir is not None else Path("data")
        path = base / "block" / f"block_{block + 1}.pkl"
        self.buffer.save_block(path)
        logger.info("block {} 数据已保存至 {}", block + 1, path)

    # --- block 起始动作 ---------------------------------------------------
    def _begin_block(self) -> None:
        self.buffer.clean()  # 每个 block 起始清空所有数据（仅用本 block 样本训练）

    def _block_order(self) -> list[int]:
        cfg = self.config
        reps = int(np.ceil(cfg.trials_per_block / cfg.n_classes))
        seq = np.tile(np.arange(cfg.n_classes), reps)[: cfg.trials_per_block]
        self.rng.shuffle(seq)
        return seq.tolist()

    # --- 休息：先更新模型，再按空格进入下一 block --------------------------
    def _rest_and_train(self) -> None:
        ui = self.ui
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

        # 1) 模型更新进行中：显示“请休息”，等待训练完成（保持可 Esc 退出）
        while thread.is_alive():
            ui.draw_rest()
            ui.flip()
            if ui.quit_requested():
                raise QuitExperiment
        self._log_update_result(result, error)

        # 2) 更新完成：提示按空格进入下一 block（无倒计时，由患者自行决定）
        ui.draw_rest("按 空格键 开始下一个 block")
        ui.flip()
        if "escape" in ui.wait_keys(keys=["space", "escape"]):
            raise QuitExperiment

    def _log_update_result(self, result: dict, error: dict) -> None:
        if "err" in error:
            logger.error("模型更新失败: {}", error["err"])
        elif result.get("updated"):
            logger.info("模型已更新（本 block 样本 {} 条）", len(self.buffer.items))
        else:
            # update 返回 False：可能是解码器不训练（如 DummyDecoder）或样本不足
            logger.info("本次未更新模型（解码器跳过；样本 {} 条）", len(self.buffer.items))
