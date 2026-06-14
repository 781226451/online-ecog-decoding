"""SEEG 脑机接口在线交互范式 (PsychoPy)。

模块概览
--------
- ``config``        : 实验参数 :class:`~seeg_task.config.ExperimentConfig`
- ``signal_source`` : 信号采集接口与模拟桩
- ``decoder``       : 实时解码器（线程安全，可热替换模型）
- ``model_update``  : 离线训练桩（ModelTrainer）
- ``buffer``        : BlockBuffer 环形滑窗缓存（trial 窗口 + block 存档）
- ``fsm``           : 两层有限状态机（TrialFSM / BlockFSM）
- ``ui``            : PsychoPy 可视化（动作提示 / 盯点 / 正确率 / 休息界面）
- ``experiment``    : 组件装配与 FSM 驱动
- ``run``           : 命令行入口
"""

__all__ = [
    "config",
    "signal_source",
    "decoder",
    "model_update",
    "buffer",
    "fsm",
    "ui",
    "experiment",
]
