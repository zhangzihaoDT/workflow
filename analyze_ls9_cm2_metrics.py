from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 数据集路径（引用 skills/intention_order_start_payment_refund_summary.py#L16 的同一路径）
DATASET_PATH = Path("/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet")
# 业务周期定义文件（仓库根目录）
BUSINESS_DEF_PATH = Path("business_definition.json")


@dataclass
class Period:
    start: pd.Timestamp
    end: pd.Timestamp


def coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def normalize_text(x: Any) -> str:
    return str(x).strip().lower()


def load_business_periods(path: Path, keys: List[str]) -> Dict[str, Period]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # 兼容嵌套结构：如果存在 time_periods，则使用其子项
    if isinstance(data, dict) and "time_periods" in data and isinstance(data["time_periods"], dict):
        data = data["time_periods"]
    periods: Dict[str, Period] = {}
    for k in keys:
        if k in data:
            start = pd.to_datetime(data[k]["start"], errors="coerce")
            end = pd.to_datetime(data[k]["end"], errors="coerce")
            if pd.isna(start) or pd.isna(end):
                continue
            periods[k] = Period(start=start, end=end)
    return periods


def pick_group_col(df: pd.DataFrame) -> str:
    # 优先使用中文列名，其次英文备用列
    if "车型分组" in df.columns:
        return "车型分组"
    elif "pre_vehicle_model_type" in df.columns:
        return "pre_vehicle_model_type"
    # 兜底：如果都没有，抛错更明确
    raise KeyError("未找到车型分组列：期望 '车型分组' 或 'pre_vehicle_model_type'")


def pick_city_level_col(df: pd.DataFrame) -> Tuple[str, str | None]:
    # 返回 (层级列名, 城市列名备用)
    level_col = None
    city_col = None
    if "license_city_level" in df.columns:
        level_col = "license_city_level"
    elif "License City Level" in df.columns:
        level_col = "License City Level"
    # 城市列备用
    if "License City" in df.columns:
        city_col = "License City"
    elif "license_city" in df.columns:
        city_col = "license_city"
    return level_col, city_col


def count_unique_orders(df: pd.DataFrame) -> int:
    col = None
    for c in ["Order Number", "order_number", "订单号"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        return len(df)
    return df[col].nunique(dropna=True)


def compute_locked_window(df: pd.DataFrame, group_name: str, period: Period, lock_days: int, group_col: str) -> pd.DataFrame:
    df = df.copy()
    # 时间列
    if "Lock_Time" not in df.columns:
        raise KeyError("数据集中未找到 'Lock_Time' 列")
    df["Lock_Time"] = coerce_datetime(df["Lock_Time"])  # 锁单时间

    # 仅保留目标车型分组
    df = df[df[group_col].astype(str).map(normalize_text) == normalize_text(group_name)]

    # end 到 end+N 天（左闭右开）
    lock_start = period.end
    lock_end = period.end + pd.Timedelta(days=lock_days)
    mask_lock = df["Lock_Time"].notna() & (df["Lock_Time"] >= lock_start) & (df["Lock_Time"] < lock_end)
    return df.loc[mask_lock].copy()


def compute_locked_window_custom(
    df: pd.DataFrame,
    group_name: str,
    period: Period,
    start_offset_days: int,
    end_offset_days: int,
    group_col: str,
) -> pd.DataFrame:
    """自定义锁单窗口，区间为 [end + start_offset_days, end + end_offset_days)。"""
    df = df.copy()
    if "Lock_Time" not in df.columns:
        raise KeyError("数据集中未找到 'Lock_Time' 列")
    df["Lock_Time"] = coerce_datetime(df["Lock_Time"])  # 锁单时间

    df = df[df[group_col].astype(str).map(normalize_text) == normalize_text(group_name)]
    lock_start = period.end + pd.Timedelta(days=start_offset_days)
    lock_end = period.end + pd.Timedelta(days=end_offset_days)
    mask_lock = df["Lock_Time"].notna() & (df["Lock_Time"] >= lock_start) & (df["Lock_Time"] < lock_end)
    return df.loc[mask_lock].copy()


def compute_daily_lock_std(locked_df: pd.DataFrame, lock_start: pd.Timestamp, lock_days: int) -> float:
    # 生成完整日期范围并填充0，衡量真实波动
    if len(locked_df) == 0:
        return 0.0
    counts = (
        locked_df.assign(day=locked_df["Lock_Time"].dt.floor("D"))
        .groupby("day")
        .apply(count_unique_orders)
    )
    full_index = pd.date_range(lock_start, periods=lock_days, freq="D")
    counts = counts.reindex(full_index, fill_value=0)
    return float(pd.Series(counts).std(ddof=1))


def compute_daily_counts(locked_df: pd.DataFrame, lock_start: pd.Timestamp, lock_days: int) -> pd.Series:
    """返回完整日期范围的每日锁单数（按订单号去重），缺失日填0。"""
    full_index = pd.date_range(lock_start, periods=lock_days, freq="D")
    if len(locked_df) == 0:
        return pd.Series([0] * len(full_index), index=full_index)
    counts = (
        locked_df.assign(day=locked_df["Lock_Time"].dt.floor("D"))
        .groupby("day")
        .apply(count_unique_orders)
    )
    counts = counts.reindex(full_index, fill_value=0)
    return counts


def is_extended_range_product(name: Any) -> bool:
    s = str(name)
    return ("52" in s) or ("66" in s)


def is_battery_66(name: Any) -> bool:
    return "66" in str(name)


def compute_conversion_rate(df: pd.DataFrame, group_name: str, period: Period, lock_days: int, group_col: str) -> Tuple[int, int, float]:
    # 小订：支付在 [start, end]
    pay_col = "Intention_Payment_Time"
    lock_col = "Lock_Time"
    if pay_col not in df.columns:
        raise KeyError("数据集中未找到 'Intention_Payment_Time' 列")
    if lock_col not in df.columns:
        raise KeyError("数据集中未找到 'Lock_Time' 列")

    df = df.copy()
    df[pay_col] = coerce_datetime(df[pay_col])
    df[lock_col] = coerce_datetime(df[lock_col])
    df = df[df[group_col].astype(str).map(normalize_text) == normalize_text(group_name)]

    pay_mask = df[pay_col].notna() & (df[pay_col] >= period.start) & (df[pay_col] <= period.end)
    lock_start = period.end
    lock_end = period.end + pd.Timedelta(days=lock_days)
    lock_mask = df[lock_col].notna() & (df[lock_col] >= lock_start) & (df[lock_col] < lock_end)

    # 留存锁单：同时满足支付在周期内 + 锁单在上市后窗口内
    retained = df[pay_mask & lock_mask]
    small_orders = df.loc[pay_mask]
    retained_count = count_unique_orders(retained)
    small_count = count_unique_orders(small_orders)
    conv = (retained_count / small_count) if small_count > 0 else 0.0
    return small_count, retained_count, float(conv)


def compute_tier1_ratio(locked_df: pd.DataFrame, city_level_col: str | None, city_col: str | None) -> float:
    if len(locked_df) == 0:
        return 0.0
    if city_level_col and city_level_col in locked_df.columns:
        series = locked_df[city_level_col].astype(str).map(normalize_text)
        tier1 = (series == normalize_text("一线")) | (series == normalize_text("一线城市"))
        denom = len(locked_df)
        num = int(tier1.sum())
        return (num / denom) if denom > 0 else 0.0
    # 备用：用城市名称推断一线
    if city_col and city_col in locked_df.columns:
        tier1_cities = {"北京", "上海", "广州", "深圳"}
        series = locked_df[city_col].astype(str)
        num = int(series.isin(tier1_cities).sum())
        denom = len(locked_df)
        return (num / denom) if denom > 0 else 0.0
    return 0.0

def compute_city_concentration(locked_df: pd.DataFrame, top_k: int = 5) -> float:
    """城市集中度：按城市分组的订单数占比之和（TopK）。
    优先使用 `License City`，其次 `license_city`，再次 `Store City`。
    订单数采用订单号去重统计。
    返回比例（0~1）。
    """
    if len(locked_df) == 0:
        return 0.0
    city_col = None
    # 优先使用 Store City，其次 License City，再次 license_city
    for c in ["Store City", "License City", "license_city"]:
        if c in locked_df.columns:
            city_col = c
            break
    if not city_col:
        return 0.0
    counts = (
        locked_df.groupby(city_col)
        .apply(count_unique_orders)
        .sort_values(ascending=False)
    )
    total = counts.sum()
    if total == 0:
        return 0.0
    top_share = float((counts.head(top_k) / total).sum())
    return top_share


def fmt_float(x: float) -> str:
    return f"{x:.2f}"


def fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def analyze_groups(df: pd.DataFrame, periods: Dict[str, Period], groups: List[str], lock_days: int) -> pd.DataFrame:
    group_col = pick_group_col(df)
    city_level_col, city_col = pick_city_level_col(df)

    rows = []
    for g in groups:
        if g not in periods:
            # 跳过未定义周期的组
            continue
        p = periods[g]

        # 锁单窗口内数据
        locked_df = compute_locked_window(df, g, p, lock_days, group_col)
        lock_start = p.end

        # 1. 上市后N日锁单数（按订单号去重）
        lock_count = count_unique_orders(locked_df)

        # 2. 增程车型发布会+1日后锁单数标准差（区间 [end+1, end+N) 且 Product Name 含 52/66）
        ext_shift_df = compute_locked_window_custom(df, g, p, start_offset_days=1, end_offset_days=lock_days, group_col=group_col)
        if "Product Name" in ext_shift_df.columns:
            ext_shift_df = ext_shift_df[ext_shift_df["Product Name"].apply(is_extended_range_product)].copy()
        daily_std = compute_daily_lock_std(ext_shift_df, p.end + pd.Timedelta(days=1), lock_days - 1 if lock_days > 1 else 0)

        # 3. 增程车型锁单数（Product Name含52或66）
        ext_lock_count = 0
        if "Product Name" in locked_df.columns:
            ext_mask = locked_df["Product Name"].apply(is_extended_range_product)
            ext_lock_count = count_unique_orders(locked_df.loc[ext_mask])

        # 4. 小订转化率（支付在[start,end] 且 锁单在[end,end+N)）
        small_count, retained_count, conv_rate = compute_conversion_rate(df, g, p, lock_days, group_col)

        # 5. 66度电占比（66 / (66+52)）
        ratio_66 = 0.0
        if "Product Name" in locked_df.columns:
            mask_66 = locked_df["Product Name"].apply(is_battery_66)
            mask_52 = locked_df["Product Name"].astype(str).str.contains("52", na=False)
            denom_df = locked_df.loc[mask_66 | mask_52]
            num_df = locked_df.loc[mask_66]
            denom = count_unique_orders(denom_df)
            num = count_unique_orders(num_df)
            ratio_66 = (num / denom) if denom > 0 else 0.0

        # 6. 城市集中度（Top5城市占比，基于 Store City 优先）
        city_concentration = compute_city_concentration(locked_df, top_k=5)

        # 7. 增程车型锁单数集中度：先筛选增程产品，再计算Top5城市占比
        ext_city_concentration = 0.0
        if "Product Name" in locked_df.columns:
            ext_locked_df = locked_df[locked_df["Product Name"].apply(is_extended_range_product)].copy()
            ext_city_concentration = compute_city_concentration(ext_locked_df, top_k=5)

        rows.append({
            "车型分组": g,
            "上市后N日锁单数": float(lock_count),
            "增程车型发布会+1日后锁单数标准差": float(daily_std),
            "增程车型锁单数": float(ext_lock_count),
            "小订转化率": float(conv_rate),
            "66度电占比": float(ratio_66),
            "城市集中度": float(city_concentration),
            "增程车型锁单数集中度": float(ext_city_concentration),
            # 附带基数，便于理解
            "小订数": float(small_count),
            "留存锁单数": float(retained_count),
        })

    result = pd.DataFrame(rows)
    return result


def build_comparison_table(df_metrics: pd.DataFrame, groups: List[str]) -> pd.DataFrame:
    # 生成比较表，包含差异（LS9 - CM2）
    df_metrics = df_metrics.set_index("车型分组")
    # 指标列顺序
    metric_cols = [
        "上市后N日锁单数",
        "增程车型锁单数",
        "增程车型发布会+1日后锁单数标准差",
        "小订转化率",
        "66度电占比",
        "城市集中度",
        "增程车型锁单数集中度",
    ]
    rows = []
    for m in metric_cols:
        v_ls9 = float(df_metrics.loc[groups[0], m]) if groups[0] in df_metrics.index else np.nan
        v_cm2 = float(df_metrics.loc[groups[1], m]) if groups[1] in df_metrics.index else np.nan
        diff = v_ls9 - v_cm2 if (not np.isnan(v_ls9) and not np.isnan(v_cm2)) else np.nan
        is_pct = m in {"小订转化率", "66度电占比", "城市集中度", "增程车型锁单数集中度"}
        rows.append({
            "指标": m,
            groups[0]: fmt_pct(v_ls9) if is_pct else fmt_float(v_ls9),
            groups[1]: fmt_pct(v_cm2) if is_pct else fmt_float(v_cm2),
            "差异(LS9-CM2)": (fmt_pct(diff) if is_pct else fmt_float(diff)) if not np.isnan(diff) else "-",
        })
    return pd.DataFrame(rows)


def save_plotly_bars_and_table(
    table_df: pd.DataFrame,
    ls9_daily: pd.Series,
    cm2_daily: pd.Series,
    title: str,
    save_path: Path,
    cm2_label: str = "CM2 每日锁单",
):
    """生成上半部分柱状图 + 下半部分指标表的综合页面。"""
    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        vertical_spacing=0.08,
        subplot_titles=("上市后N日每日锁单对比", "指标汇总表"),
    )

    # 顶部柱状图
    fig.add_trace(
        go.Bar(name="LS9 每日锁单", x=ls9_daily.index, y=ls9_daily.values),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(name=cm2_label, x=cm2_daily.index, y=cm2_daily.values),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="日期", row=1, col=1)
    fig.update_yaxes(title_text="锁单数(订单数)", row=1, col=1)
    fig.update_layout(barmode="group")

    # 底部表格
    fig.add_trace(
        go.Table(
            header=dict(values=list(table_df.columns), fill_color="#3f51b5", font=dict(color="white"), align="center"),
            cells=dict(values=[table_df[c] for c in table_df.columns], align="center"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(title=title, template="plotly_white", height=900)
    fig.write_html(str(save_path))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="对比计算 LS9 与 CM2 指标差异（Plotly DataFrame 输出）"
    )
    parser.add_argument(
        "--lock-window-days",
        type=int,
        default=30,
        help="上市后锁单统计窗口天数，默认30",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DATASET_PATH),
        help="数据集parquet路径，默认使用仓外固定数据路径",
    )
    parser.add_argument(
        "--business-def",
        type=str,
        default=str(BUSINESS_DEF_PATH),
        help="业务周期定义JSON路径（默认仓库根目录）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analyze_ls9_vs_cm2_increase.html",
        help="输出HTML文件名（综合页面：柱状图+汇总表）",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    business_path = Path(args.business_def)
    output_path = Path(args.output)

    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集未找到：{dataset_path}")
    if not business_path.exists():
        raise FileNotFoundError(f"业务定义未找到：{business_path}")

    df = pd.read_parquet(dataset_path)
    periods = load_business_periods(business_path, keys=["LS9", "CM2"])  # 仅提取所需周期

    groups = ["LS9", "CM2"]
    metrics_df = analyze_groups(df, periods, groups=groups, lock_days=args.lock_window_days)

    # 比较表（含差异列），并统一两位小数/百分比格式
    table_df = build_comparison_table(metrics_df, groups=groups)

    # 另存未格式化的原始指标CSV，便于后续复核
    metrics_df.to_csv("analyze_ls9_cm2_metrics_raw.csv", index=False, encoding="utf-8-sig")

    # 生成每日锁单柱状图（LS9 全量；CM2 全量；CM2 增程）
    group_col = pick_group_col(df)
    # LS9 锁单窗口
    ls9_locked = compute_locked_window(df, "LS9", periods["LS9"], args.lock_window_days, group_col)
    ls9_daily = compute_daily_counts(ls9_locked, periods["LS9"].end, args.lock_window_days)
    # CM2 全量锁单窗口
    cm2_full_locked = compute_locked_window(df, "CM2", periods["CM2"], args.lock_window_days, group_col)
    cm2_full_daily = compute_daily_counts(cm2_full_locked, periods["CM2"].end, args.lock_window_days)
    # CM2 增程：Product Name 含 52 或 66
    cm2_ext_locked = cm2_full_locked
    cm2_ext_label = "CM2增程 每日锁单"
    if "Product Name" in cm2_ext_locked.columns:
        cm2_ext_locked = cm2_ext_locked[cm2_ext_locked["Product Name"].apply(is_extended_range_product)].copy()
    cm2_ext_daily = compute_daily_counts(cm2_ext_locked, periods["CM2"].end, args.lock_window_days)

    # 复用原始输出文件名生成表格；另生成包含柱状图的页面
    # 综合页面输出（单文件）：顶部三条柱状图 + 底部汇总表
    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        vertical_spacing=0.08,
        subplot_titles=("上市后N日每日锁单对比", "指标汇总表"),
    )
    fig.add_trace(go.Bar(name="LS9 每日锁单", x=ls9_daily.index, y=ls9_daily.values, marker=dict(color="#27AD00")), row=1, col=1)
    fig.add_trace(go.Bar(name="CM2 每日锁单", x=cm2_full_daily.index, y=cm2_full_daily.values, marker=dict(color="#005783")), row=1, col=1)
    fig.add_trace(go.Bar(name=cm2_ext_label, x=cm2_ext_daily.index, y=cm2_ext_daily.values, marker=dict(color="#A3ACB9")), row=1, col=1)
    fig.update_xaxes(title_text="日期", row=1, col=1)
    fig.update_yaxes(title_text="锁单数(订单数)", row=1, col=1)
    fig.update_layout(barmode="group")

    fig.add_trace(
        go.Table(
            header=dict(values=list(table_df.columns), fill_color="#3f51b5", font=dict(color="white"), align="center"),
            cells=dict(values=[table_df[c] for c in table_df.columns], align="center"),
        ),
        row=2,
        col=1,
    )
    fig.update_layout(title=f"LS9 vs CM2 指标对比（N={args.lock_window_days}）", template="plotly_white", height=900)
    fig.write_html(str(output_path))
    print(f"✅ 已生成对比页面：{output_path}")


if __name__ == "__main__":
    main()
