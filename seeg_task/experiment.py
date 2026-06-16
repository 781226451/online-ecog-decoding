"""实验编排：构造组件并用两层 FSM（:mod:`seeg_task.fsm`）驱动范式运行。

`Experiment` 负责装配（信号源 / 解码器 / 缓冲 / UI）与引导页、总结页；block-trial 的
状态流转交给 :class:`~seeg_task.fsm.BlockFSM`（其内部运行 :class:`~seeg_task.fsm.TrialFSM`）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger
from psychopy import core
from pylsl import StreamInfo, StreamOutlet, cf_string

from .buffer import BlockBuffer
from .config import ExperimentConfig
from .decoder import BaseDecoder, create_decoder
from .fsm import BlockFSM, QuitExperiment
from .signal_source import SignalSource, create_source
from .ui import ExperimentUI


def _create_marker_outlet():
    # pylsl 为必装依赖；try/except 仅兜底 outlet 运行期创建失败（如 LSL 网络异常），
    # 失败时返回 None，事件改为不推送（见 _make_event_pusher 的 no-op 降级）。
    try:
        info = StreamInfo(
            "ParadigmEvents", "Markers", 1, 0, cf_string,
            source_id="seeg-interaction-task",
        )
        outlet = StreamOutlet(info)
        logger.info("[events] LSL marker 流已创建 (ParadigmEvents / Markers)")
        return outlet
    except Exception as exc:
        logger.error("[events] 创建 LSL marker 流出错，事件将不推送: {}", exc)
        return None


class Experiment:
    """把信号源、解码器、缓冲、FSM 与 UI 串联成完整范式。"""

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        source: SignalSource | None = None,
        decoder: BaseDecoder | None = None,
        session_dir: Path | None = None,
        marker_outlet=None,
    ) -> None:
        self.config = config or ExperimentConfig()
        self.config.validate()
        cfg = self.config

        self.rng = np.random.default_rng(cfg.random_seed)
        self.source = source or create_source(cfg, rng=self.rng)
        self.decoder = decoder or create_decoder(cfg, rng=self.rng)
        # current_item 窗口大小 = 解码窗口长度；EXECUTE 期间按时长流式推入样本
        self.buffer = BlockBuffer(cfg.n_channels, cfg.window_samples)
        self.session_dir = session_dir
        # marker outlet 可由调用方（run.py）尽早创建并注入，使 LabRecorder 在被试信息
        # 输入框阶段就能发现并连上 ParadigmEvents 流，避免范式开始后才建流导致 marker 被丢。
        self.marker_outlet = marker_outlet

        self.ui: ExperimentUI | None = None

    def run(self) -> None:
        cfg = self.config
        self.ui = ExperimentUI(cfg)
        ui = self.ui
        # 已注入则复用；否则退化为运行时创建（保持单独使用 Experiment 时仍能推送 marker）
        marker_outlet = self.marker_outlet or _create_marker_outlet()
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

            BlockFSM(cfg, self.source, self.decoder, self.buffer, ui, self.rng,
                     session_dir=self.session_dir,
                     push_event=_make_event_pusher(marker_outlet)
                 ).run()

            ui.draw_message("实验结束，按任意键退出。")
            ui.flip()
            ui.wait_keys()
        except QuitExperiment:
            ui.draw_message("已退出实验。")
            ui.flip()
            core.wait(0.8)
        finally:
            ui.close()
            self.source.close()


def _make_event_pusher(outlet):
    """如果 outlet 存在，返回 ``push_event(**fields)`` 闭包；否则返回 None（FSM 端降级为 no-op）。

    每次推送都记录 ``have_consumers()``：LSL 不规则字符串流不会为「尚未连接的消费者」
    缓冲样本，连接建立前推送的 marker 会被静默丢弃。日志中若持续出现「无消费者」告警，
    即说明录制器（如 LabRecorder）尚未连上，marker 正在被丢——这是定位「xdf 里 0 事件」
    问题的关键信号。
    """
    if outlet is None:
        return None
    stats = {"sent": 0, "no_consumer": 0}

    def push_event(timestamp=None, **fields):
        # timestamp 不为 None 时作为 LSL 样本时间戳显式传入（如 PREDICT 用推理前的时刻），
        # 它不进 JSON payload——读取端从 LSL 样本时间还原（见 xdf_viewer）。
        phase = fields.get("phase", "?")
        has_consumer = outlet.have_consumers()
        try:
            payload = [json.dumps(fields, ensure_ascii=False, separators=(",", ":"))]
            if timestamp is None:
                outlet.push_sample(payload)
            else:
                outlet.push_sample(payload, timestamp)
        except Exception as exc:
            logger.error("[events] 推送 {} 失败: {}", phase, exc)
            return
        stats["sent"] += 1
        if has_consumer:
            logger.debug("[events] 推送 {} 成功 (累计 {}, consumers=yes)", phase, stats["sent"])
        else:
            stats["no_consumer"] += 1
            logger.warning(
                "[events] 推送 {} 时无消费者连接，样本很可能被丢弃 "
                "(累计无消费者 {} 次)；请确认录制器已开始录制并连上 ParadigmEvents",
                phase, stats["no_consumer"],
            )
    return push_event
