"""LSL 模拟数据源（发送端 / outlet）。

无采集硬件时，用本脚本在网络上推一路模拟 SEEG 流，供 :class:`~seeg_task.signal_source.LSLSource`
（接收端 / inlet）解析、联调整条范式。

用法::

    # 终端 A：先启动模拟流（保持运行）
    python -m seeg_task.lsl_mock_stream                 # 默认 name=MockSEEG, type=EEG, 64ch, 1000Hz
    python -m seeg_task.lsl_mock_stream --channels 64 --srate 1000 --name MockSEEG --type EEG

    # 终端 B：再启动实验，按流类型或流名连接
    python -m seeg_task.run --source lsl --lsl-type EEG
    python -m seeg_task.run --source lsl --lsl-name MockSEEG

按 Ctrl+C 停止推流。注意：模拟数据与试次标签无关，解码正确率不会有真实含义，
仅用于验证「采集→解码→在线训练→界面」整条链路是否打通。
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="LSL 模拟 SEEG 数据源（发送端）")
    parser.add_argument("--name", default="MockSEEG", help="LSL 流名")
    parser.add_argument("--type", dest="stype", default="EEG", help="LSL 流类型（如 EEG/sEEG）")
    parser.add_argument("--channels", type=int, default=64, help="通道数")
    parser.add_argument("--srate", type=float, default=1000.0, help="采样率 (Hz)")
    parser.add_argument("--chunk", type=int, default=20, help="每次推送的样本数（影响时延/CPU）")
    args = parser.parse_args()

    try:
        from pylsl import StreamInfo, StreamOutlet, local_clock
    except ImportError:
        print("需要安装 pylsl：`uv sync` 或 `pip install pylsl`")
        return 1

    info = StreamInfo(
        name=args.name, type=args.stype, channel_count=args.channels,
        nominal_srate=args.srate, channel_format="float32", source_id=f"mock_{args.name}",
    )
    # 写入通道标签元数据（部分接收端会用到）
    chns = info.desc().append_child("channels")
    for i in range(args.channels):
        ch = chns.append_child("channel")
        ch.append_child_value("label", f"Ch{i + 1}")
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", args.stype)

    outlet = StreamOutlet(info, chunk_size=args.chunk)
    print(f"[mock] 正在推流：name={args.name} type={args.stype} "
          f"channels={args.channels} srate={args.srate}Hz —— 按 Ctrl+C 停止")

    rng = np.random.default_rng(0)
    # 每通道一个基础频率，叠加随机游走 + 噪声，模拟多通道神经信号。
    freqs = rng.uniform(8.0, 30.0, size=args.channels)
    phase = np.zeros(args.channels)
    drift = np.zeros(args.channels)

    dt = 1.0 / args.srate
    period = args.chunk * dt
    n = 0
    try:
        t0 = local_clock()
        while True:
            chunk = np.empty((args.chunk, args.channels), dtype=np.float32)
            for k in range(args.chunk):
                phase += 2.0 * np.pi * freqs * dt
                drift += rng.standard_normal(args.channels) * 0.05
                sample = 20.0 * np.sin(phase) + drift + rng.standard_normal(args.channels) * 5.0
                chunk[k] = sample.astype(np.float32)
            outlet.push_chunk(chunk.tolist())
            n += args.chunk
            # 节流到目标采样率
            n_target = t0 + (n / args.srate)
            sleep = n_target - local_clock()
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print(f"\n[mock] 已停止，共推送约 {n} 个样本。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
