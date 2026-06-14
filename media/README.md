# 动作素材目录

左面板的动作示范素材放在这里。程序按 **动作 key** 自动查找文件，命名约定为：

```
<动作key>.<扩展名>
```

查找顺序：`.gif` → 视频(`.mp4/.mov/.avi/.mkv/.webm`) → 静态图(`.png/.jpg/.jpeg/.bmp`)。
找到第一个即使用；找不到则左面板退化为占位框 + 文字「(无动作素材)」。

## 默认动作 key（见 `seeg_task/config.py` 的 `ExperimentConfig.actions`）

| 动作 key      | 屏幕显示 | 期望文件示例           |
|---------------|----------|------------------------|
| `left_hand`   | 左手握拳 | `left_hand.gif` / `.mp4`  |
| `right_hand`  | 右手握拳 | `right_hand.gif` / `.mp4` |
| `feet`        | 双脚背屈 | `feet.gif` / `.mp4`       |
| `tongue`      | 伸舌     | `tongue.gif` / `.mp4`     |

## 说明

- **GIF**：用 imageio 预解码为帧序列循环播放，帧时长读取自 gif 元数据，缺失时回退为
  `ExperimentConfig.gif_frame_duration`（默认 0.05s）。
- **视频**：用 `psychopy.visual.MovieStim` 循环、静音播放。
- 想新增/修改动作或素材命名，编辑 `seeg_task/config.py` 中的 `actions` 列表即可，逻辑代码无需改动。
