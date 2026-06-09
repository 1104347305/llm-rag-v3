"""费率表 PDF → 结构化存储 + Wiki 页面生成。

用法:
  python scripts/ingest_rate_table.py \
    --pdf "dataset/平保寿发〔2025〕559号附件2：费率表.pdf" \
    --product "智盈倍护26" \
    --table-id "rate.zhiying26"
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WIKI_ENTITY_TEMPLATE = """---
schema_version: 2.1
industry: insurance
knowledge_domain: product
taxonomy_path: ["product", "rate_table"]
type: rate_table
entity_type: rate_table
business_phase: underwriting
dedup_key: "rate_table.{table_id}"
title: "{title}"
summary: "{summary}"
created: {date}
updated: {date}
created_at: {date}
updated_at: {date}
---

# {title}

{description}

## 查询方法

{lookup_guide}

## 表结构

{table_structure}

## 各维度的取值

{dimension_values}

---
*本文档由 PDF 表格自动生成。具体数值通过 lookup_rate 工具查询。*
"""


def main() -> None:
    args = parse_args()

    # 1. 提取 PDF 文本
    text = extract_pdf_text(args.pdf)

    # 2. 识别表格区块
    fallback_title = Path(args.pdf).stem
    tables = split_tables(text, fallback_title)

    if not tables:
        print("未检测到表格数据，请检查 PDF。")
        sys.exit(1)

    # 3. 逐表解析维度+数据
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for i, tbl in enumerate(tables):
        dimensions, rows = parse_rate_table(tbl["lines"])

        if not rows:
            print(f"  表 {i+1}: 解析失败，跳过")
            continue

        table_id = f"{args.table_id}.{_safe_id(tbl['subtitle'] or tbl['title'])}"
        title = tbl["subtitle"] or tbl["title"]

        # 4. 计算行数、数值范围等统计
        values = [r["value"] for r in rows]
        dim_values = _extract_dim_values(rows, dimensions)
        summary = f"{title}，{len(rows)}行，费率范围{min(values):.0f}~{max(values):.0f}元"

        # 5. 写入存储
        _write_to_store(args, table_id, title, dimensions, rows, now_str)

        # 6. 生成 Wiki 页面
        wiki_path = f"wiki/entities/{_safe_id(table_id)}.md"
        wiki_content = WIKI_ENTITY_TEMPLATE.format(
            table_id=table_id,
            title=title,
            summary=summary,
            date=now_str,
            description=_build_description(title, dimensions, rows),
            lookup_guide=_build_lookup_guide(args.product, title, dimensions, dim_values),
            table_structure=_build_structure(dimensions),
            dimension_values=_build_dim_values(dimensions, dim_values),
        )
        Path(wiki_path).parent.mkdir(parents=True, exist_ok=True)
        Path(wiki_path).write_text(wiki_content, encoding="utf-8")
        print(f"  Wiki: {wiki_path} ({len(rows)} rows)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="费率表 PDF → 存储 + Wiki 页面")
    p.add_argument("--pdf", required=True, help="费率表 PDF 路径")
    p.add_argument("--product", required=True, help="产品名称，如 '智盈倍护26'")
    p.add_argument("--table-id", required=True, help="表 ID 前缀，如 'rate.zhiying26'")
    p.add_argument("--unit", default="每万元基本保险金额", help="费率单位")
    return p.parse_args()


def extract_pdf_text(pdf_path: str) -> str:
    """调用 pdftotext 提取文本。"""
    result = subprocess.run(
        ["pdftotext", "-layout", pdf_path, "-"],
        capture_output=True, text=True,
    )
    return result.stdout


def split_tables(text: str, fallback_title: str = "") -> list[dict[str, Any]]:
    """按性别区块（男性/女性）分割表格。"""
    lines = text.splitlines()

    # 找到 "男性" 和 "女性" 的分界
    male_start = female_start = -1
    for i, line in enumerate(lines):
        cleaned = line.strip()
        if cleaned == "男性" and male_start < 0:
            male_start = i
        elif cleaned == "女性" and female_start < 0 and i > male_start:
            female_start = i

    tables = []
    if male_start >= 0:
        end = female_start if female_start > 0 else len(lines)
        tables.append({
            "title": fallback_title,
            "subtitle": f"{fallback_title}（男性）",
            "lines": lines[male_start:end],
        })
    if female_start >= 0:
        tables.append({
            "title": fallback_title,
            "subtitle": f"{fallback_title}（女性）",
            "lines": lines[female_start:],
        })

    if not tables:
        return [{"title": fallback_title, "subtitle": "", "lines": lines}]
    return tables


HEADERS = ["6年交", "10年交", "15年交", "20年交", "交至55周岁", "交至60周岁", "交至65周岁"]


def parse_rate_table(lines: list[str]) -> tuple[list[str], list[dict[str, Any]]]:
    """从 pdftotext -layout 输出中解析固定列宽费率表。

    输入格式:
        交费期间
                    6 年交   10 年交    15 年交   ...
        投保年龄
          0           364     234      160       ...
          18          520     335      229       ...

    返回: (dimensions, rows)
    """
    dimensions: list[str] = []
    headers: list[str] = HEADERS

    # 检测列标题
    for line in lines:
        cleaned = line.strip()
        if "交费期间" in cleaned or "交费方式" in cleaned:
            dimensions.append("period")
        if "投保年龄" in cleaned or "年龄" in cleaned:
            dimensions.append("age")

    if not dimensions:
        dimensions = ["age", "period"]

    # 提取数据行：行首是整数年龄，后续是空格分隔的数值
    rows: list[dict[str, Any]] = []
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        # 跳过非数据行
        if any(kw in cleaned for kw in ["费率表", "单位：", "人民币", "每万元",
                                          "交费期间", "投保年龄", "男性", "女性",
                                          "平安智盈", "《平安"]):
            continue
        # 纯数字单独一行是页码
        if re.match(r"^\s*\d+\s*$", cleaned):
            continue
        if re.match(r"^-\d+-$", cleaned):
            continue

        parts = _split_row(cleaned)

        # 解析：第一个应为年龄（0-120），后续为费率值
        age_val = None
        rate_vals: list[float] = []
        for p in parts:
            try:
                num = float(p.replace(",", ""))
                if age_val is None and 0 <= num <= 120 and num == int(num):
                    age_val = int(num)
                else:
                    rate_vals.append(num)
            except ValueError:
                continue

        if age_val is None or not rate_vals:
            continue

        # 跳过年龄异常的行（如0岁以下或120岁以上）
        if age_val > 120:
            continue

        for col_idx, rate in enumerate(rate_vals):
            if col_idx < len(headers):
                period = headers[col_idx]
            else:
                break  # 超出列数忽略
            rows.append({"age": age_val, "period": period, "value": rate})

    return dimensions, rows


def _split_row(line: str) -> list[str]:
    """按多空格或制表符分割表格行。"""
    return re.split(r"\s{2,}|\t", line.strip())


def _is_dim_label(text: str) -> bool:
    return any(kw in text for kw in ["年龄", "投保", "交费", "费率", "单位", "基本保险"])


def _safe_id(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fff\-]", "_", name).strip("_")[:80]


def _extract_dim_values(rows: list[dict[str, Any]], dimensions: list[str]) -> dict[str, list[str]]:
    result: dict[str, set[str]] = {d: set() for d in dimensions}
    for row in rows:
        for d in dimensions:
            if d in row:
                result[d].add(str(row[d]))
    return {k: sorted(v, key=_sort_key) for k, v in result.items()}


def _sort_key(x: str) -> str:
    try:
        return f"{int(re.sub(r'[^0-9]', '', x) or 0):05d}"
    except ValueError:
        return x


def _build_description(title: str, dimensions: list[str], rows: list[dict[str, Any]]) -> str:
    return (
        f"本表记录 {title} 的费率数据，"
        f"共 {len(rows)} 条记录，"
        f"维度包括 {' × '.join(dimensions)}。"
    )


def _build_lookup_guide(product: str, title: str, dimensions: list[str],
                         dim_values: dict[str, list[str]]) -> str:
    guide = f"使用 `lookup_rate(product=\"{product}\", gender=\"男或女\", age=年龄, period=\"交费期间\")` 查询具体费率。\n\n"
    guide += "当前表的维度取值：\n"
    for dim in dimensions:
        vals = dim_values.get(dim, [])
        sample = vals[:10]
        guide += f"- {dim}: {', '.join(sample)}"
        if len(vals) > 10:
            guide += f" ...（共 {len(vals)} 个取值）"
        guide += "\n"
    return guide


def _build_structure(dimensions: list[str]) -> str:
    return " | ".join(dimensions + ["值"]) + "\n" + "|".join(["---"] * (len(dimensions) + 1))


def _build_dim_values(dimensions: list[str],
                      dim_values: dict[str, list[str]]) -> str:
    lines = []
    for dim in dimensions:
        vals = dim_values.get(dim, [])
        lines.append(f"### {dim}")
        lines.append(f"取值数: {len(vals)}")
        lines.append(f"示例: {', '.join(vals[:20])}")
        lines.append("")
    return "\n".join(lines)


def _write_to_store(args: argparse.Namespace, table_id: str, title: str,
                    dimensions: list[str], rows: list[dict[str, Any]],
                    date: str) -> None:
    """写入 TabularStore。"""
    from src.main.python.storage.tabular_store import TabularStore

    # 只有 gender 作为额外维时，从 table_id 中提取
    if "男性" in title or "男性" in table_id:
        for r in rows:
            r["gender"] = "男"
        dimensions = ["gender"] + dimensions
    elif "女性" in title or "女性" in table_id:
        for r in rows:
            r["gender"] = "女"
        dimensions = ["gender"] + dimensions

    TabularStore().init_schema()
    TabularStore().upsert_table(
        table_id=table_id,
        title=title,
        unit=args.unit,
        dimensions=dimensions,
        rows=rows,
        wiki_path=f"wiki/entities/{_safe_id(table_id)}.md",
    )


if __name__ == "__main__":
    main()
