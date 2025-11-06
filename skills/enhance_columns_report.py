from pathlib import Path
from datetime import datetime
from typing import Dict, List

import math

try:
    import pandas as pd
    import numpy as np
    from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
except Exception:
    raise RuntimeError("请先安装依赖：pip install pandas pyarrow numpy")


BUSINESS_PARQUET = Path("/Users/zihao_/Documents/coding/dataset/formatted/business_daily_metrics.parquet")
INTENTION_PARQUET = Path("/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet")
OUTPUT_MD = Path("/Users/zihao_/Documents/github/W35_workflow/metrics_columns_report.md")


def _file_size_mb(p: Path) -> float:
    try:
        return round(p.stat().st_size / (1024 * 1024), 2)
    except Exception:
        return float("nan")


def _identify_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    numeric_cols: List[str] = []
    date_cols: List[str] = []
    categorical_cols: List[str] = []

    for c in df.columns:
        s = df[c]
        if is_datetime64_any_dtype(s):
            date_cols.append(c)
        elif is_numeric_dtype(s):
            numeric_cols.append(c)
        else:
            # 通过列名辅助识别日期型
            lc = str(c).lower()
            if any(k in lc for k in ["time", "date", "时间", "日期"]):
                # 尝试解析为日期但不改变原数据类型
                date_cols.append(c)
            else:
                categorical_cols.append(c)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "date": date_cols,
    }


def _completeness(df: pd.DataFrame) -> float:
    total = df.shape[0] * df.shape[1]
    non_null = int(df.notna().sum().sum())
    return round(100.0 * non_null / total, 2) if total > 0 else float("nan")


def analyze_dataset(parquet_path: Path) -> Dict:
    if not parquet_path.exists():
        raise FileNotFoundError(f"文件不存在: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    types = _identify_types(df)
    comp = _completeness(df)
    dup_count = int(df.duplicated().sum())
    missing_counts = df.isna().sum()
    missing = (
        missing_counts[missing_counts > 0]
        .sort_values(ascending=False)
        .to_dict()
    )
    missing_percent = {
        k: round(100.0 * v / df.shape[0], 2) if df.shape[0] > 0 else float("nan")
        for k, v in missing.items()
    }
    return {
        "path": str(parquet_path),
        "size_mb": _file_size_mb(parquet_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "shape": (df.shape[0], df.shape[1]),
        "columns": list(df.columns),
        "types": types,
        "completeness": comp,
        "duplicates": dup_count,
        "missing": missing,
        "missing_percent": missing_percent,
    }


def _format_missing_table(stats: Dict) -> List[str]:
    lines: List[str] = []
    if not stats["missing"]:
        lines.append("✅ 未发现缺失值")
        return lines
    lines.append("")
    lines.append("### 缺失值异常")
    lines.append("")
    lines.append("| 列名 | 缺失数量 | 缺失比例 |")
    lines.append("|------|----------|----------|")
    for col, cnt in stats["missing"].items():
        pct = stats["missing_percent"].get(col, float("nan"))
        pct_str = f"{pct:.2f}%" if not math.isnan(pct) else "NA"
        lines.append(f"| {col} | {cnt} | {pct_str} |")
    return lines


def format_md_for_dataset(title: str, stats: Dict) -> str:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## 数据概览")
    lines.append(f"- **数据文件**: {stats['path']}")
    lines.append(f"- **生成时间**: {stats['generated_at']}")
    if not math.isnan(stats["size_mb"]):
        lines.append(f"- **文件大小**: {stats['size_mb']:.2f} MB")
    lines.append("")

    lines.append("## 数据基本信息")
    lines.append(f"- **数据形状**: {stats['shape'][0]} 行 × {stats['shape'][1]} 列")
    lines.append(f"- **数据完整性**: {stats['completeness']:.2f}%")
    lines.append(f"- **重复行数**: {stats['duplicates']}")
    lines.append("")

    lines.append("## 数据类型分布")
    lines.append(f"- **数值列**: {len(stats['types']['numeric'])} 个")
    lines.append(f"- **分类列**: {len(stats['types']['categorical'])} 个  ")
    lines.append(f"- **日期列**: {len(stats['types']['date'])} 个")
    lines.append("")

    lines.append("## 列信息详情")
    if stats['types']['numeric']:
        lines.append("### 数值列")
        lines.append(", ".join(stats['types']['numeric']))
        lines.append("")
    if stats['types']['categorical']:
        lines.append("### 分类列")
        lines.append(", ".join(stats['types']['categorical']))
        lines.append("")
    if stats['types']['date']:
        lines.append("### 日期列")
        lines.append(", ".join(stats['types']['date']))
        lines.append("")

    lines.extend(_format_missing_table(stats))
    lines.append("")

    lines.append("## 字段列表")
    for c in stats['columns']:
        lines.append(f"- `{c}`")

    return "\n".join(lines)


def main():
    biz_stats = analyze_dataset(BUSINESS_PARQUET)
    intention_stats = analyze_dataset(INTENTION_PARQUET)

    biz_md = format_md_for_dataset("Business Daily Metrics 字段清单", biz_stats)
    intention_md = format_md_for_dataset("附：意向订单分析字段清单", intention_stats)

    # 合并为一个文件，业务日指标在前，意向订单附录在后
    content = biz_md + "\n\n" + intention_md + "\n"
    OUTPUT_MD.write_text(content, encoding="utf-8")
    print(f"已生成完善版字段说明: {OUTPUT_MD}")


if __name__ == "__main__":
    main()