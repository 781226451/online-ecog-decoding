"""SEEG 脑机接口在线交互范式 (PsychoPy)。

模块概览
--------
- ``config``        : 实验参数 :class:`~seeg_task.config.ExperimentConfig`
- ``signal_source`` : 信号采集接口与模拟桩
- ``decoder``       : 实时解码器（线程安全，可热替换模型）
- ``model_update``  : 历史样本缓冲与离线训练桩
- ``ui``            : PsychoPy 可视化（左侧动作提示 / 右侧正确率 / 休息界面）
- ``experiment``    : block-trial 主循环与休息期训练编排
- ``run``           : 命令行入口
"""

__all__ = [
    "config",
    "signal_source",
    "decoder",
    "model_update",
    "ui",
    "experiment",
]
