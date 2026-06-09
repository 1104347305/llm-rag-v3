"""检索器抽象基类。

定义所有检索通道的统一接口，方便 Pipeline 通过依赖注入替换实现。
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """所有检索器的抽象基类。

    每个检索器实现 search() 方法，返回 [(chunk_id, score), ...]。
    is_available 属性用于运行时判断检索通道是否可用（如 ES 未配置时自动跳过）。
    """

    @abstractmethod
    def search(self, query: str, project_id: str, top_k: int) -> list[tuple[str, float]]:
        """执行检索，返回按分数降序排列的 (chunk_id, score) 列表。

        Args:
            query: 检索查询文本。
            project_id: 项目标识。
            top_k: 返回的最大结果数。

        Returns:
            [(chunk_id, score), ...]，按 score 降序。
        """
        ...

    @property
    def is_available(self) -> bool:
        """检索器是否可用。子类可覆写以检查外部服务连通性。"""
        return True
