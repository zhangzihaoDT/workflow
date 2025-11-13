#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
锁单行为深度分析

目标：
- 统计每个“车型分组”的至锁单时间差（Lock_Time 相对以下时间点的差值）：
  - Order_Create_Time
  - Intention_Payment_Time
  - first_assign_time
  - first_touch_time
- 按车型分组做描述统计：平均数、中位数、标准差（单位：天）
- 使用 Plotly 生成散点+四分位箱线图（box + points='all'），每个指标一张图

输出：
- CSV：models/lock_depth_summary.csv
- HTML：models/lock_depth_box_order_create.html 等四张图

用法示例：
python skills/lock_depth_analysis.py \
  --data formatted/intention_order_analysis.parquet \
  --save_csv models/lock_depth_summary.csv \
  --save_html_dir models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import plotly.express as px


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "formatted" / "intention_order_analysis.parquet"
# 兼容：若相对路径不存在则回退到绝对路径
FALLBACK_DATA_PATH = Path("/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet")
BUSINESS_DEF_PATH = ROOT_DIR / "business_definition.json"


def _normalize(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def resolve_column(df: pd.DataFrame, synonyms: List[str]) -> str:
    """在多个同义列名中解析实际列名，支持大小写/空格/下划线差异。"""
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
    """加载业务周期定义，并返回每个车型的 start/end 时间戳。"""
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


def filter_by_business_cycle_lock_window(
    df: pd.DataFrame,
    group_col: str,
    lock_col: str,
    periods: Dict[str, Dict[str, pd.Timestamp]],
    days_after_end: int = 30,
) -> pd.DataFrame:
    """保留各车型在“end 日起的 30 天窗口”内锁单的订单，窗口为 [end, end+30)。

    若某车型在业务定义中没有 end 或无法解析，则该车型的订单不纳入分析。
    """
    df = df.copy()
    df[lock_col] = coerce_datetime(df[lock_col])

    per_df = pd.DataFrame([
        {"group_norm": _normalize(k), "end": v.get("end")}
        for k, v in periods.items()
        if v.get("end") is not pd.NaT and pd.notna(v.get("end"))
    ])

    # 若没有任何有效的 end，直接返回空集，避免误分析
    if per_df.empty:
        return df.iloc[0:0].copy()

    df["__group_norm__"] = df[group_col].astype(str).map(_normalize)
    df = df.merge(per_df, left_on="__group_norm__", right_on="group_norm", how="left")

    window_end = df["end"] + pd.Timedelta(days=days_after_end)
    mask = (
        df["end"].notna()
        & df[lock_col].notna()
        & (df[lock_col] >= df["end"]) 
        & (df[lock_col] < window_end)
    )
    out = df[mask].copy()
    # 清理辅助列
    for col in ["group_norm", "__group_norm__", "end"]:
        if col in out.columns:
            out.drop(columns=[col], inplace=True)
    return out


def compute_lock_deltas(df: pd.DataFrame, group_col: str, cols: Dict[str, str]) -> pd.DataFrame:
    """计算 Lock_Time 相对多个时间点的差值（单位：天）。"""
    # 转为时间类型
    for c in [cols["order_create"], cols["payment"], cols["assign"], cols["touch"], cols["lock"]]:
        if c in df.columns:
            df[c] = coerce_datetime(df[c])

    # 差值：锁单 - 各时间点（天）
    out = df[[group_col, cols["order_create"], cols["payment"], cols["assign"], cols["touch"], cols["lock"]]].copy()
    out["days_lock_from_order_create"] = (out[cols["lock"]] - out[cols["order_create"]]) / pd.Timedelta(days=1)
    out["days_lock_from_payment"] = (out[cols["lock"]] - out[cols["payment"]]) / pd.Timedelta(days=1)
    out["days_lock_from_first_assign"] = (out[cols["lock"]] - out[cols["assign"]]) / pd.Timedelta(days=1)
    out["days_lock_from_first_touch"] = (out[cols["lock"]] - out[cols["touch"]]) / pd.Timedelta(days=1)
    return out


def describe_by_group(df_deltas: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """按车型分组输出平均数/中位数/标准差（忽略缺失值）。"""
    metrics = [
        "days_lock_from_order_create",
        "days_lock_from_payment",
        "days_lock_from_first_assign",
        "days_lock_from_first_touch",
    ]
    agg_map = {m: ["mean", "median", "std"] for m in metrics}
    # 仅保留实际观测到的分组，避免未使用类别出现在汇总中
    desc = df_deltas.groupby(group_col, observed=True).agg(agg_map)
    # 展平多级列索引
    desc.columns = [f"{m}_{stat}" for m, stat in desc.columns]
    desc = desc.reset_index()
    return desc


def save_box_plots(df_deltas: pd.DataFrame, group_col: str, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        ("days_lock_from_order_create", "锁单-订单创建（天）", "lock_depth_box_order_create.html"),
        ("days_lock_from_payment", "锁单-小订支付（天）", "lock_depth_box_payment.html"),
        ("days_lock_from_first_assign", "锁单-首次分配（天）", "lock_depth_box_assign.html"),
        ("days_lock_from_first_touch", "锁单-首次触达（天）", "lock_depth_box_touch.html"),
    ]
    for col, title, fname in plots:
        # 仅保留该指标非空的记录，避免箱线图渲染异常
        data = df_deltas[[group_col, col]].dropna()
        fig = px.box(
            data,
            x=group_col,
            y=col,
            points="all",
            title=f"{title}（散点+四分位）",
        )
        fig.update_layout(
            xaxis_title="车型分组",
            yaxis_title="天数",
            hovermode="x unified",
            template="plotly_white",
        )
        fig.write_html(str(save_dir / fname))


def main():
    parser = argparse.ArgumentParser(description="锁单行为深度分析：按车型分组的至锁单时间差与分布")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="数据文件路径（parquet）")
    parser.add_argument("--save_csv", default=str(ROOT_DIR / "models" / "lock_depth_summary.csv"), help="汇总CSV导出路径")
    parser.add_argument("--save_html_dir", default=str(ROOT_DIR / "models"), help="箱线图保存目录")
    parser.add_argument("--business_def", default=str(BUSINESS_DEF_PATH), help="业务定义JSON（包含各车型time_periods）")
    args = parser.parse_args()

    df = load_data(Path(args.data))

    # 解析所需列
    group_col = resolve_column(df, [
        "车型分组", "Product_Types", "产品类型", "车型类别", "车系", "product_types", "product_type"
    ])
    order_create_col = resolve_column(df, [
        "Order_Create_Time", "order_create_time", "订单创建时间", "订单创建日期"
    ])
    payment_col = resolve_column(df, [
        "Intention_Payment_Time", "intention_payment_time", "支付时间"
    ])
    assign_col = resolve_column(df, [
        "first_assign_time", "首次分配时间", "首次分配日期"
    ])
    touch_col = resolve_column(df, [
        "first_touch_time", "首次触达时间", "首次触达日期"
    ])
    lock_col = resolve_column(df, [
        "Lock_Time", "lock_time", "锁单时间", "锁单日期"
    ])

    cols = {
        "order_create": order_create_col,
        "payment": payment_col,
        "assign": assign_col,
        "touch": touch_col,
        "lock": lock_col,
    }

    # 过滤掉指定车型（LS7、L7）
    exclude_set_norm = {_normalize(x) for x in ["LS7", "L7"]}
    df = df[~df[group_col].astype(str).map(_normalize).isin(exclude_set_norm)].copy()

    # 按业务周期 end 后 30 天锁单时间窗口过滤
    periods = load_business_periods(Path(args.business_def))
    before_cnt = len(df)
    df = filter_by_business_cycle_lock_window(df, group_col, lock_col, periods, days_after_end=30)
    after_cnt = len(df)
    print(f"按业务周期end后30天过滤：由 {before_cnt} 条 → 保留 {after_cnt} 条")

    # 输出车型分组所有取值（过滤后）
    unique_groups = sorted(df[group_col].astype(str).unique())
    print("车型分组所有取值：")
    for v in unique_groups:
        print(f"- {v}")

    # 计算时间差
    df_deltas = compute_lock_deltas(df, group_col, cols)

    # 描述统计
    summary = describe_by_group(df_deltas, group_col)
    # 每车型样本数：按 Lock_Time 非空计数
    samples = (
        df_deltas.groupby(group_col)[lock_col]
        .count()
        .rename("样本数")
        .reset_index()
    )
    summary = summary.merge(samples, on=group_col, how="left")
    # 保险过滤：若仍存在未观测组或全空统计，进一步剔除
    metric_cols = [
        "days_lock_from_order_create_mean",
        "days_lock_from_order_create_median",
        "days_lock_from_order_create_std",
        "days_lock_from_payment_mean",
        "days_lock_from_payment_median",
        "days_lock_from_payment_std",
        "days_lock_from_first_assign_mean",
        "days_lock_from_first_assign_median",
        "days_lock_from_first_assign_std",
        "days_lock_from_first_touch_mean",
        "days_lock_from_first_touch_median",
        "days_lock_from_first_touch_std",
    ]
    summary = summary.dropna(how="all", subset=metric_cols)
    summary = summary[~summary[group_col].astype(str).map(_normalize).isin(exclude_set_norm)].copy()

    # 保存CSV
    save_csv = Path(args.save_csv)
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(save_csv, index=False, encoding="utf-8-sig")
    print(f"已导出汇总CSV: {save_csv}")

    # 生成箱线图（散点+四分位）
    save_html_dir = Path(args.save_html_dir)
    save_box_plots(df_deltas, group_col, save_html_dir)
    print(f"已保存箱线图到目录: {save_html_dir}")


if __name__ == "__main__":
    main()