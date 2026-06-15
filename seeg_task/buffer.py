"""滑窗缓存 + 标注样本存档。

本模块提供一个独立的缓存结构 :class:`BlockBuffer`，用于在「逐采样点流式输入」场景下
维护一个固定大小的滑动窗口，并在需要时把窗口连同标签打包存档：

- ``current_item`` : 固定大小 ``(C, N)`` 的滑动窗口（C=通道数，N=采样点数），最旧在前。
- :meth:`update_current_items` : 传入 ``(C, k)`` 多列新数据，按 FIFO 推入——丢弃最旧、
  在末尾追加（窗口整体左移 k 格）。
- :meth:`save_sample` : 传入一个整型 ``label``，把 ``current_item`` 的副本与 ``label`` 打包成
  定长 tuple ``(ndarray, int)`` 存入存档列表 :attr:`items`。
- :meth:`clean` : 清除所有数据——把 ``current_item`` 重置为全 0，并清空已存档的 :attr:`items`。

实现采用**环形缓冲**：:meth:`update_current_items` 整段写入 + 移动写指针（O(k)），把
O(C·N) 的“有序化”推迟到读取 ``current_item`` 时才做，从而在高频推入时高效。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class BlockBuffer:
    """逐采样点滑动窗口缓存（环形缓冲），支持把窗口连同标签存档。

    Attributes:
        current_item: 只读属性，返回形状 ``(n_channels, window_samples)`` 的**时间有序**窗口
            （最旧在前、最新在后）的副本。
        items: 存档列表，每个元素为定长 tuple ``(ndarray[C, N], label:int)``。
    """

    def __init__(self, n_channels: int, window_samples: int, dtype=np.float64) -> None:
        if n_channels <= 0 or window_samples <= 0:
            raise ValueError("n_channels 与 window_samples 必须为正")
        self.n_channels = n_channels
        self.window_samples = window_samples
        self.dtype = dtype
        # 环形缓冲底层存储 + 下一个写入列索引
        self._buf: np.ndarray = np.zeros((n_channels, window_samples), dtype=dtype)
        self._pos: int = 0
        self.items: list[tuple[np.ndarray, int]] = []
        self._predict_cache: list[np.ndarray] = []

    def update_current_items(self, chunk: np.ndarray) -> None:
        """批量 FIFO 推入多列新数据（按时间顺序，第 0 列最旧；最新的覆盖最旧的）。

        用整段写入实现，O(k)（k=列数）且仅一两次拷贝；``k`` 可为 0（空操作）。

        Args:
            chunk: 形状 ``(n_channels, k)``。
        """
        chunk = np.asarray(chunk, dtype=self.dtype)
        if chunk.ndim != 2 or chunk.shape[0] != self.n_channels:
            raise ValueError(
                f"chunk 形状应为 ({self.n_channels}, k)，实际 {chunk.shape}"
            )
        k = chunk.shape[1]
        if k == 0:
            return
        n = self.window_samples
        if k >= n:
            # 新数据已铺满整窗：只保留最后 n 列，写指针归零（最旧在第 0 列）
            self._buf[:] = chunk[:, -n:]
            self._pos = 0
            return
        end = self._pos + k
        if end <= n:                       # 不跨越环边界
            self._buf[:, self._pos:end] = chunk
        else:                              # 跨越边界：拆成尾段 + 头段
            first = n - self._pos
            self._buf[:, self._pos:] = chunk[:, :first]
            self._buf[:, : end - n] = chunk[:, first:]
        self._pos = end % n

    def _snapshot(self) -> np.ndarray:
        """返回时间有序（最旧在前）的滑动窗口副本，不写入 ``_predict_cache``。"""
        return np.concatenate([self._buf[:, self._pos:], self._buf[:, : self._pos]], axis=1)

    @property
    def current_item(self) -> np.ndarray:
        """返回时间有序（最旧在前）的滑动窗口副本，形状 ``(n_channels, window_samples)``。

        纯读取，不产生任何副作用。如需同时记录到 predict_cache，请使用 :meth:`record_predict`。
        """
        return self._snapshot()

    def record_predict(self) -> np.ndarray:
        """取当前窗口快照，追加到 ``_predict_cache``，并返回该快照供推理使用。

        每次推理应调用此方法而非直接访问 ``current_item``，以确保预测记录准确。
        """
        item = self._snapshot()
        self._predict_cache.append(item)
        return item

    def save_sample(self, label: int) -> None:
        """把当前窗口（有序副本）与 ``label`` 打包成 tuple 存入 :attr:`items`，随后清空当前窗口。

        Args:
            label: 整型类别标签（取值范围 ``0 .. N-1``）。存档为 ``(current_item, label)``。
        """
        self.items.append((self._snapshot(), int(label)))
        self.reset_window()  # 仅清空 current_item（保留已存档 items），避免跨段残留

    def reset_window(self) -> None:
        """把滑动窗口 ``current_item`` 重置为全 0（不影响 :attr:`items`）。"""
        self._buf[:] = 0
        self._pos = 0

    def save_block(self, path: str | Path) -> None:
        """把本 block 的预测缓存与训练样本存档保存为 pkl 文件。

        文件内容为 dict，键为：
        - ``predict_cache_list`` : ``_predict_cache`` 的快照（每次 ``current_item`` 的副本列表）。
        - ``update_cache_list``  : ``items`` 的快照（``(ndarray, label)`` 元组列表）。

        Args:
            path: 目标文件路径（不存在的父目录会自动创建）。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "predict_cache_list": list(self._predict_cache),
            "update_cache_list": list(self.items),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def clean(self) -> None:
        """清除所有数据：``current_item`` 重置为全 0，并清空已存档的 :attr:`items` 与 ``_predict_cache``。"""
        self.reset_window()
        self.items.clear()
        self._predict_cache.clear()

    def __len__(self) -> int:
        """已存档样本数。"""
        return len(self.items)
