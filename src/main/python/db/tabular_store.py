"""表格数据存储与查询。

支持多维费率表的存储（SQLite）和精确查找。
wiki 实体页只存表头描述，具体数值通过 lookup 工具查询。
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from src.main.python.config import settings
from src.main.python.utils.logging import get_logger

logger = get_logger(__name__)


class TabularStore:
    """多维表格数据存储。

    每张表由维度列 + 值列组成，存入 SQLite 后通过精确匹配查询。
    """

    def __init__(self) -> None:
        self._path = settings.storage_dir / "tabular.db"

    @property
    def path(self) -> Path:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        return self._path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        conn.execute("pragma journal_mode=wal")
        conn.execute("pragma synchronous=normal")
        conn.execute(f"pragma busy_timeout={settings.sqlite_busy_timeout}")
        return conn

    def init_schema(self) -> None:
        """初始化 Parquet 表结构。"""

        with closing(self._connect()) as conn:
            conn.executescript("""
                create table if not exists tabular_meta (
                    table_id   text primary key,
                    title      text not null,
                    unit       text not null,
                    dimensions text not null,   -- JSON array of dim names
                    wiki_path  text,             -- 对应的 wiki entity 路径
                    remarks    text
                );
                create table if not exists tabular_data (
                    table_id text not null,
                    dims     text not null,       -- JSON object of dim values
                    value    real not null,
                    primary key (table_id, dims)
                );
            """)
            conn.commit()

    def upsert_table(self, table_id: str, title: str, unit: str,
                     dimensions: list[str], rows: list[dict[str, Any]],
                     wiki_path: str = "", remarks: str = "") -> None:
        """写入一张表的数据。

        Args:
            table_id: 表唯一标识，如 "rate.zhiying26.male"
            title: 表标题
            unit: 数值单位
            dimensions: 维度名列表，如 ["gender", "age", "period"]
            rows: 每行 {dim1: val1, ..., "value": rate}
        """
        import json as _json
        with closing(self._connect()) as conn:
            conn.execute(
                "insert or replace into tabular_meta values(?,?,?,?,?,?)",
                (table_id, title, unit, _json.dumps(dimensions, ensure_ascii=False),
                 wiki_path, remarks),
            )
            conn.executemany(
                "insert or replace into tabular_data values(?,?,?)",
                [
                    (table_id,
                     _json.dumps({k: v for k, v in row.items() if k != "value"},
                                 ensure_ascii=False, sort_keys=True),
                     row["value"])
                    for row in rows
                ],
            )
            conn.commit()

    def lookup(self, table_id: str, **dims: object) -> dict[str, Any] | None:
        """精确查找一个值。

        Returns:
            {"value": 335, "unit": "每万元基本保险金额"} 或 None
        """
        import json as _json
        dims_json = _json.dumps(
            {k: str(v) for k, v in sorted(dims.items())},
            ensure_ascii=False, sort_keys=True,
        )
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "select d.value, m.unit, m.title "
                "from tabular_data d join tabular_meta m on d.table_id = m.table_id "
                "where d.table_id = ? and d.dims = ?",
                (table_id, dims_json),
            ).fetchone()
        if row is None:
            return None
        return {"value": row["value"], "unit": row["unit"], "title": row["title"]}

    def search_rows(self, table_id: str, **filters: object) -> list[dict[str, Any]]:
        """按条件搜索多行。filters 支持 {age: 18} 即只筛选年龄=18的所有行。"""
        import json as _json
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "select dims, value from tabular_data where table_id = ?", (table_id,)
            ).fetchall()
        results = []
        for row in rows:
            dims_dict = _json.loads(row["dims"])
            if filters and not all(str(dims_dict.get(k)) == str(v) for k, v in filters.items()):
                continue
            results.append({**dims_dict, "value": row["value"]})
        return results

    def get_meta(self, table_id: str) -> dict[str, Any] | None:
        """获取表的元信息。"""
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "select * from tabular_meta where table_id = ?", (table_id,)
            ).fetchone()
        if row is None:
            return None
        import json as _json
        return {
            "table_id": row["table_id"],
            "title": row["title"],
            "unit": row["unit"],
            "dimensions": _json.loads(row["dimensions"]),
            "wiki_path": row["wiki_path"],
            "remarks": row["remarks"],
        }

    def list_tables(self) -> list[dict[str, Any]]:
        """列出所有表。"""
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("select * from tabular_meta").fetchall()
        import json as _json
        return [
            {
                "table_id": r["table_id"],
                "title": r["title"],
                "unit": r["unit"],
                "dimensions": _json.loads(r["dimensions"]),
                "wiki_path": r["wiki_path"],
            }
            for r in rows
        ]


# ── 工具函数（LLM 可调用）──────────────────────────────────

def lookup_rate(product: str, gender: str, age: int, period: str) -> dict[str, Any]:
    """查询保险费率。

    Args:
        product: 产品简称，如 "智盈倍护26", "盛世优享26"
        gender: "男" 或 "女"
        age: 投保年龄
        period: 交费期间，如 "10年交", "趸交", "20年交"

    Returns:
        {"value": 335, "unit": "元/每万元保额"} 或 {"error": "..."}
    """
    store = TabularStore()
    table_id = f"rate.{product}.{gender}"
    result = store.lookup(table_id, age=str(age), period=period)
    if result is None:
        return {"error": f"未找到 {product} {gender}性 {age}岁 {period} 的费率"}
    return result


def describe_tables() -> list[dict[str, Any]]:
    """列出所有可用的表格，供 LLM 了解有什么表可以查。"""
    return TabularStore().list_tables()
