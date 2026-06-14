"""命令行入口。

用法::

    python -m seeg_task.run              # 启动完整实验（需要显示器）
    python -m seeg_task.run --selftest   # 无界面自检：校验解码/缓冲/训练契约
"""

from __future__ import annotations

import argparse

import numpy as np

from .config import ExperimentConfig


def _selftest() -> int:
    """不打开窗口，验证核心数据/模型契约。返回 0 表示通过。"""
    from .decoder import Decoder, LinearModel, extract_features
    from .model_update import HistoryBuffer, ModelTrainer
    from .signal_source import SyntheticSource

    cfg = ExperimentConfig()
    cfg.validate()
    rng = np.random.default_rng(0)
    n_features = cfg.n_channels

    source = SyntheticSource(cfg.n_channels, cfg.window_samples, cfg.n_classes, rng=rng)
    decoder = Decoder(LinearModel.random_init(cfg.n_classes, n_features, rng), cfg.n_classes)
    trainer = ModelTrainer(cfg.n_classes, n_features)
    buffer = HistoryBuffer(cfg.history_size)

    # 1) predict 契约：形状 (n_classes,) 且概率和≈1
    x = source.read_window(true_label=0)
    assert x.shape == (cfg.n_channels, cfg.window_samples), x.shape
    probs = decoder.predict(x)
    assert probs.shape == (cfg.n_classes,), probs.shape
    assert abs(float(probs.sum()) - 1.0) < 1e-6, probs.sum()
    assert np.all(probs >= 0)
    print(f"[ok] predict 输出形状 {probs.shape}，概率和 = {probs.sum():.6f}")

    # 2) HistoryBuffer 容量截断
    small = HistoryBuffer(capacity=3)
    for i in range(5):
        small.add(np.zeros((cfg.n_channels, cfg.window_samples)), i % cfg.n_classes)
    assert len(small) == 3, len(small)
    assert len(small.recent(10)) == 3
    print(f"[ok] HistoryBuffer 截断到容量 = {len(small)}")

    # 3) 训练产物可被 swap_model 接收且 predict 不抛错
    labels = []
    for _ in range(cfg.train_n_samples):
        lab = int(rng.integers(cfg.n_classes))
        buffer.add(source.read_window(true_label=lab), lab)
        labels.append(lab)
    new_model = trainer.train(buffer.recent(cfg.train_n_samples))
    assert isinstance(new_model, LinearModel), type(new_model)
    decoder.swap_model(new_model)
    probs2 = decoder.predict(source.read_window(true_label=0))
    assert probs2.shape == (cfg.n_classes,)
    print("[ok] ModelTrainer 产物可热替换，predict 正常")

    # 4) 训练应在模拟数据上提升正确率（链路有效性）
    def accuracy(dec: Decoder, n: int = 200) -> float:
        hit = 0
        for _ in range(n):
            lab = int(rng.integers(cfg.n_classes))
            if int(np.argmax(dec.predict(source.read_window(true_label=lab)))) == lab:
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

    from .experiment import Experiment

    Experiment(config=cfg).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
