"""命令行入口。

用法::

    python -m seeg_task.run              # 启动完整实验（需要显示器）
    python -m seeg_task.run --selftest   # 无界面自检：校验解码/缓冲/训练契约
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from .config import ExperimentConfig


_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | "
    "blk={extra[block]} trial={extra[trial]} act={extra[action]} | "
    "{name}:{function}:{line} - {message}"
)


def _configure_logging(level: str = "INFO", log_file: str | None = None, *, write_file: bool = True) -> None:
    """配置 Loguru：控制台输出，``write_file=True`` 时同时写入文件。

    日志格式带上下文字段 ``blk/trial/act``（block id、trial id、当前动作名）；
    无上下文时显示默认占位 ``-``（见 :meth:`~seeg_task.fsm.BlockFSM.run` 的 contextualize）。
    ``write_file=True`` 且 ``log_file`` 为 None 时自动生成路径 ``logs/YYYY-MM-DD_HHMMSS.log``。
    """
    level = str(level).upper()
    logger.remove()  # 移除 Loguru 默认 handler，避免重复输出
    logger.configure(extra={"block": "-", "trial": "-", "action": "-"})  # 上下文默认值
    logger.add(sys.stderr, level=level, format=_LOG_FORMAT)
    if write_file:
        if log_file is None:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = str(log_dir / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log")
        logger.add(log_file, level=level, format=_LOG_FORMAT, encoding="utf-8")


def _selftest() -> int:
    """不打开窗口，验证核心数据/模型契约。返回 0 表示通过。"""
    _configure_logging("WARNING", write_file=False)  # 抑制自检中数百次 predict 的 INFO 日志
    from .buffer import BlockBuffer
    from .decoder import Decoder, LinearModel
    from .model_update import ModelTrainer

    cfg = ExperimentConfig()
    cfg.validate()
    rng = np.random.default_rng(0)
    n_features = cfg.n_channels
    wlen = 256  # 自检用窗口长度（任意：特征对时间维做归约，与长度无关）

    decoder = Decoder(LinearModel.random_init(cfg.n_classes, n_features, rng), cfg.n_classes)
    trainer = ModelTrainer(cfg.n_classes, n_features)

    # 直接合成类别可分窗口（不经信号源）：每通道功率随类别模式而变 -> 可被解码区分
    patterns = rng.standard_normal((cfg.n_classes, cfg.n_channels))

    def window(label: int) -> np.ndarray:
        idx = np.arange(wlen)
        osc = np.sin(2.0 * np.pi * (0.01 * (1 + label)) * idx)            # (wlen,)
        sig = patterns[label][:, None] * osc[None, :]                     # (n_channels, wlen)
        return sig + rng.standard_normal((cfg.n_channels, wlen))

    # 1) predict 契约：形状 (n_classes,) 且概率和≈1
    x = window(0)
    assert x.shape == (cfg.n_channels, wlen), x.shape
    probs = decoder.predict(x)
    assert probs.shape == (cfg.n_classes,), probs.shape
    assert abs(float(probs.sum()) - 1.0) < 1e-6, probs.sum()
    assert np.all(probs >= 0)
    print(f"[ok] predict 输出形状 {probs.shape}，概率和 = {probs.sum():.6f}")

    # 2) BlockBuffer：批量推入 -> 存档 -> 清空
    bb = BlockBuffer(cfg.n_channels, wlen)
    bb.update_current_items(window(0)); bb.save_sample(0)
    bb.update_current_items(window(1)); bb.save_sample(1)
    assert len(bb) == 2 and bb.items[0][0].shape == (cfg.n_channels, wlen)
    bb.clean(); assert len(bb) == 0
    print("[ok] BlockBuffer 存档/清空正常")

    # 3) 训练产物可被 swap_model 接收且 predict 不抛错
    samples = []
    for _ in range(64):
        lab = int(rng.integers(cfg.n_classes))
        samples.append((window(lab), lab))
    new_model = trainer.train(samples)
    assert isinstance(new_model, LinearModel), type(new_model)
    decoder.swap_model(new_model)
    probs2 = decoder.predict(window(0))
    assert probs2.shape == (cfg.n_classes,)
    print("[ok] ModelTrainer 产物可热替换，predict 正常")

    # 4) 训练应在模拟数据上提升正确率（链路有效性）
    def accuracy(dec: Decoder, n: int = 200) -> float:
        hit = 0
        for _ in range(n):
            lab = int(rng.integers(cfg.n_classes))
            if int(np.argmax(dec.predict(window(lab)))) == lab:
                hit += 1
        return hit / n

    fresh = Decoder(LinearModel.random_init(cfg.n_classes, n_features, rng), cfg.n_classes)
    before = accuracy(fresh)
    fresh.swap_model(new_model)
    after = accuracy(fresh)
    print(f"[ok] 训练前正确率 ≈ {before:.2f}，训练后 ≈ {after:.2f}")
    assert after >= before, "训练后正确率未提升"

    print("\n全部自检通过 ✅")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SEEG 脑机接口交互范式")
    parser.add_argument(
        "--selftest", action="store_true", help="无界面自检（不打开 PsychoPy 窗口）"
    )
    parser.add_argument(
        "--config", default=None,
        help="范式配置文件(TOML)路径；缺省时若仓库根目录存在 paradigm_config.toml 则自动加载",
    )
    parser.add_argument("--fullscreen", action="store_true", help="全屏运行")
    parser.add_argument("--blocks", type=int, default=None, help="覆盖 block 数")
    parser.add_argument("--trials", type=int, default=None, help="覆盖每 block 试次数")
    parser.add_argument(
        "--source", choices=["synthetic", "dummy", "lsl"], default=None,
        help="信号源：synthetic(默认,可分模拟) / dummy(纯随机) / lsl(外部实时流)",
    )
    parser.add_argument("--lsl-name", default=None, help="LSL 流名（source=lsl 时）")
    parser.add_argument("--lsl-type", default=None, help="LSL 流类型，默认 EEG（source=lsl 时）")
    parser.add_argument("--log-level", default=None, help="日志级别 DEBUG/INFO/WARNING/...")
    parser.add_argument("--log-file", default=None, help="日志文件路径；缺省仅输出到控制台")
    args = parser.parse_args()

    if args.selftest:
        return _selftest()

    # 默认值 < 配置文件 < 命令行参数
    from pathlib import Path

    from .config import load_config

    if args.config is not None:
        cfg = load_config(args.config)
        print(f"[run] 已加载范式配置: {args.config}")
    else:
        default_cfg = Path(__file__).resolve().parent.parent / "paradigm_config.toml"
        if default_cfg.exists():
            cfg = load_config(default_cfg)
            print(f"[run] 已加载默认范式配置: {default_cfg}")
        else:
            cfg = ExperimentConfig()

    if args.fullscreen:
        cfg.fullscreen = True
    if args.blocks is not None:
        cfg.n_blocks = args.blocks
    if args.trials is not None:
        cfg.trials_per_block = args.trials
    if args.source is not None:
        cfg.source_type = args.source
    if args.lsl_name is not None:
        cfg.lsl_stream_name = args.lsl_name
    if args.lsl_type is not None:
        cfg.lsl_stream_type = args.lsl_type
    if args.log_level is not None:
        cfg.log_level = args.log_level
    if args.log_file is not None:
        cfg.log_file = args.log_file

    from psychopy import gui
    dlg = gui.Dlg(title="被试信息")
    dlg.addField("被试编号:", "")
    data = dlg.show()
    if not dlg.OK or not data[0].strip():
        print("未输入被试编号，已取消。")
        return 1
    subject_name = data[0].strip()

    session_dir = Path("data") / subject_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    log_file = cfg.log_file if cfg.log_file is not None else str(session_dir / "run.log")
    _configure_logging(cfg.log_level, log_file)

    from .experiment import Experiment

    Experiment(config=cfg, session_dir=session_dir).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
