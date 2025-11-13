#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
周期小订与上市后锁单相关性分析

目标：
- 参考 intention_order_start_payment_refund_summary 的指标定义
- 聚合每个车型（CM0, CM1, CM2, DM0, DM1, LS9）：
  - 小订数（支付时间 ∈ [start, end]）
  - 上市后锁单数（锁单时间 ∈ [end, end+30)）
- 计算整体相关性：
  - Pearson 相关（numpy corrcoef）
  - Spearman 秩相关（对两个计数序列进行秩转换后再计算 Pearson）

输出：
- CSV：models/cycle_conversion_correlation.csv（每车型的两个计数）
- HTML：models/cycle_conversion_correlation.html（表格与相关性结论）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "formatted" / "intention_order_analysis.parquet"
FALLBACK_DATA_PATH = Path("/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet")
BUSINESS_DEF_PATH = ROOT_DIR / "business_definition.json"


def _normalize(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def resolve_column(df: pd.DataFrame, synonyms: List[str]) -> str:
    target_norm = {_normalize(n) for n in synonyms}
    mapping = {_normalize(c): c for c in df.columns}
    for key_norm, original in mapping.items():
        if key_norm in target_norm:
            return original
    for key_norm, original in mapping.items():
        if any(key_norm.find(n) != -1 for n in target_norm):
            return original
    raise KeyError(f"未找到匹配列，期望列之一：{synonyms}，实际列：{list(df.columns)}")


def coerce_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def load_data(path: Path) -> pd.DataFrame:
    if path and path.exists():
        use_path = path
    elif DEFAULT_DATA_PATH.exists():
        use_path = DEFAULT_DATA_PATH
    elif FALLBACK_DATA_PATH.exists():
        use_path = FALLBACK_DATA_PATH
        print(f"相对路径不存在，回退到绝对路径: {use_path}")
    else:
        raise FileNotFoundError(f"数据文件不存在: {path} 或 {DEFAULT_DATA_PATH} 或 {FALLBACK_DATA_PATH}")
    print(f"读取数据: {use_path}")
    df = pd.read_parquet(use_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_business_periods(path: Path) -> Dict[str, Dict[str, pd.Timestamp]]:
    if not path.exists():
        raise FileNotFoundError(f"业务定义文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    periods_raw = j.get("time_periods", {})
    periods: Dict[str, Dict[str, pd.Timestamp]] = {}
    for group, se in periods_raw.items():
        start = pd.to_datetime(se.get("start"), errors="coerce") if se.get("start") else pd.NaT
        end = pd.to_datetime(se.get("end"), errors="coerce") if se.get("end") else pd.NaT
        periods[group] = {"start": start, "end": end}
    return periods


def aggregate_counts(
    df: pd.DataFrame,
    group_col: str,
    pay_col: str,
    lock_col: str,
    periods: Dict[str, Dict[str, pd.Timestamp]],
    target_groups: List[str],
    lock_window_days: int = 30,
) -> pd.DataFrame:
    df = df.copy()
    df[pay_col] = coerce_datetime(df[pay_col])
    df[lock_col] = coerce_datetime(df[lock_col])

    # 排除指定车型（LS7、L7）
    exclude_set_norm = {_normalize(x) for x in ["LS7", "L7"]}
    df = df[~df[group_col].astype(str).map(_normalize).isin(exclude_set_norm)].copy()

    rows = []
    for key in target_groups:
        rng = periods.get(key)
        if not rng:
            print(f"跳过 {key}：business_definition.json 未提供该周期")
            continue
        start, end = rng.get("start"), rng.get("end")
        if pd.isna(start) or pd.isna(end):
            print(f"跳过 {key}：start/end 解析失败：{rng}")
            continue

        # 仅保留与周期同名车型分组的数据
        norm_key = _normalize(key)
        df_k = df[df[group_col].astype(str).map(_normalize) == norm_key].copy()

        pay_mask = df_k[pay_col].notna() & (df_k[pay_col] >= start) & (df_k[pay_col] <= end)
        lock_start = end
        lock_end = end + pd.Timedelta(days=lock_window_days)
        lock_mask = df_k[lock_col].notna() & (df_k[lock_col] >= lock_start) & (df_k[lock_col] < lock_end)

        rows.append({
            "车型分组": key,
            "小订数_全周期": int(pay_mask.sum()),
            "锁单数_end后30天": int(lock_mask.sum()),
        })

    return pd.DataFrame(rows)


def compute_correlations(df_counts: pd.DataFrame) -> Dict[str, float]:
    a = df_counts["小订数_全周期"].astype(float).to_numpy()
    b = df_counts["锁单数_end后30天"].astype(float).to_numpy()
    if len(a) < 2:
        return {"pearson": np.nan, "spearman": np.nan}
    pearson = float(np.corrcoef(a, b)[0, 1])
    # Spearman：对两个序列进行秩转换后计算 Pearson
    rank_a = pd.Series(a).rank(method="average").to_numpy()
    rank_b = pd.Series(b).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(rank_a, rank_b)[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def main():
    parser = argparse.ArgumentParser(description="周期小订与上市后锁单相关性分析")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="数据文件路径（parquet）")
    parser.add_argument("--business_def", default=str(BUSINESS_DEF_PATH), help="业务定义JSON（包含各车型time_periods）")
    parser.add_argument("--save_csv", default=str(ROOT_DIR / "models" / "cycle_conversion_correlation.csv"), help="CSV导出路径")
    parser.add_argument("--save_html", default=str(ROOT_DIR / "models" / "cycle_conversion_correlation.html"), help="HTML导出路径")
    args = parser.parse_args()

    df = load_data(Path(args.data))
    periods = load_business_periods(Path(args.business_def))

    # 列解析
    group_col = resolve_column(df, [
        "车型分组", "Product_Types", "产品类型", "车型类别", "车系", "product_types", "product_type"
    ])
    pay_col = resolve_column(df, [
        "Intention_Payment_Time", "intention_payment_time", "支付时间"
    ])
    lock_col = resolve_column(df, [
        "Lock_Time", "lock_time", "锁单时间", "锁单日期"
    ])

    target_groups = ["CM0", "CM1", "CM2", "DM0", "DM1", "LS9"]
    counts = aggregate_counts(df, group_col, pay_col, lock_col, periods, target_groups, lock_window_days=30)

    save_csv = Path(args.save_csv)
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(save_csv, index=False, encoding="utf-8-sig")
    print(f"已导出计数CSV: {save_csv}")

    corr = compute_correlations(counts)
    print(f"相关性：Pearson={corr['pearson']:.4f}, Spearman={corr['spearman']:.4f}")

    # 生成HTML结果页
    html = f"""
    <!doctype html>
    <html lang=\"zh-CN\">
    <head>
      <meta charset=\"utf-8\" />
      <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
      <title>周期小订与上市后锁单相关性</title>
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif; margin: 24px; }}
        h1 {{ font-size: 20px; margin-bottom: 12px; }}
        .desc {{ color: #555; margin-bottom: 12px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }}
        th {{ background: #f9fafb; }}
        .links {{ margin: 12px 0; }}
        .links a {{ margin-right: 16px; }}
        .metrics {{ margin-top: 16px; }}
      </style>
    </head>
    <body>
      <h1>周期小订与上市后锁单相关性</h1>
      <div class=\"desc\">小订数统计范围为 [start, end]；锁单数统计范围为 [end, end+30)。</div>
      <div class=\"links\"><a href=\"{save_csv.name}\" download>下载 CSV</a></div>
      {counts.to_html(index=False)}
      <div class=\"metrics\">
        <p><strong>Pearson 相关：</strong> {corr['pearson']:.4f}</p>
        <p><strong>Spearman 秩相关：</strong> {corr['spearman']:.4f}</p>
      </div>
    </body>
    </html>
    """

    save_html = Path(args.save_html)
    save_html.parent.mkdir(parents=True, exist_ok=True)
    save_html.write_text(html, encoding="utf-8")
    print(f"已保存HTML页面: {save_html}")


if __name__ == "__main__":
    main()