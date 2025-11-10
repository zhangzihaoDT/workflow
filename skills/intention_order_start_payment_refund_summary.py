from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "formatted" / "intention_order_analysis.parquet"
BUSINESS_DEF_PATH = ROOT_DIR / "business_definition.json"

# 兼容：若相对路径不存在则回退到 order_trend_monitor 的绝对路径
FALLBACK_DATA_PATH = Path("/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet")


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


def load_periods(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    periods = data.get("time_periods", {})
    if not periods:
        raise ValueError("business_definition.json 中未找到 time_periods")
    return periods


def load_data(path: Path) -> pd.DataFrame:
    if path.exists():
        use_path = path
    elif FALLBACK_DATA_PATH.exists():
        use_path = FALLBACK_DATA_PATH
        print(f"相对路径不存在，回退到绝对路径: {use_path}")
    else:
        raise FileNotFoundError(f"数据文件不存在: {path} 或 {FALLBACK_DATA_PATH}")
    print(f"读取数据: {use_path}")
    df = pd.read_parquet(use_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def coerce_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def summarize_for_period(
    df: pd.DataFrame,
    group_col: str,
    pay_col: str,
    refund_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    window_days: int,
) -> pd.DataFrame:
    # day1～N：包含 start 当天算 day1（区间：start ～ start+N 天的开区间上界，覆盖第N天整天）
    if window_days < 1:
        raise ValueError("window_days 必须为 >=1 的整数")
    d1 = start
    dN_exclusive = start + pd.Timedelta(days=window_days)

    df_pay = df[~df[pay_col].isna()].copy()
    df_refund = df[~df[refund_col].isna()].copy()

    in_d1_dN_pay = df_pay[(df_pay[pay_col] >= d1) & (df_pay[pay_col] < dN_exclusive)]
    # 退订数定义更新：仅统计有支付时间且支付时间早于退订时间的订单在起始1～3天内发生的退订
    df_refund_paid = df_refund[~df_refund[pay_col].isna()].copy()
    # 退订时间窗口：起始日（含）到起始+N天（不含），等价于退订时间≤起始+(N-1)的整天
    mask_window = (df_refund_paid[refund_col] >= d1) & (df_refund_paid[refund_col] < dN_exclusive)
    mask_pay_before_refund = df_refund_paid[pay_col] < df_refund_paid[refund_col]
    in_d1_d3_refund = df_refund_paid[mask_window & mask_pay_before_refund]
    in_full_pay = df_pay[(df_pay[pay_col] >= start) & (df_pay[pay_col] <= end)]

    pay_col_name = f"payment_d1_d{window_days}"
    refund_col_name = f"refund_d1_d{window_days}"
    g1 = in_d1_dN_pay.groupby(group_col).size().rename(pay_col_name)
    g2 = in_d1_d3_refund.groupby(group_col).size().rename(refund_col_name)
    g3 = in_full_pay.groupby(group_col).size().rename("payment_full_period")

    out = pd.concat([g1, g2, g3], axis=1).fillna(0).astype(int).reset_index()
    # 新增比值列（百分比形式，保留一位小数）：refund_d1_dN / payment_d1_dN
    ratio = (
        out[refund_col_name].astype("Float64")
        / out[pay_col_name].replace({0: pd.NA}).astype("Float64")
    )
    ratio_col_name = f"refund_to_payment_ratio_d1_d{window_days}"
    out[ratio_col_name] = ratio.astype("Float64").mul(100).round(1).map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else pd.NA
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="按车型分组输出各业务周期起始第1～N天及全周期的支付/退订统计")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="数据文件路径（parquet）")
    parser.add_argument("--business", default=str(BUSINESS_DEF_PATH), help="business_definition.json 路径")
    parser.add_argument("--save", default=str(ROOT_DIR / "models" / "period_start_payment_refund_summary.csv"), help="导出CSV路径")
    parser.add_argument("--window_days", type=int, default=3, help="起始 N 天窗口（含起始日，统计范围为 [start, start+N) ）")
    args = parser.parse_args()

    df = load_data(Path(args.data))
    periods = load_periods(Path(args.business))

    # 解析所需列
    group_col = resolve_column(df, [
        "车型分组", "Product_Types", "产品类型", "车型类别", "车系", "product_types", "product_type"
    ])
    pay_col = resolve_column(df, [
        "intention_payment_time", "Intention_Payment_Time", "payment_time", "支付时间"
    ])
    refund_col = resolve_column(df, [
        "intention_refund_time", "Intention_Refund_Time", "refund_time", "退订时间"
    ])

    # 转为时间类型
    df[pay_col] = coerce_datetime(df[pay_col])
    df[refund_col] = coerce_datetime(df[refund_col])

    # 输出车型分组所有取值
    unique_groups = sorted(df[group_col].astype(str).unique())
    print("车型分组所有取值：")
    for v in unique_groups:
        print(f"- {v}")

    # 逐周期统计
    all_rows: List[pd.DataFrame] = []
    for key, rng in periods.items():
        start = pd.to_datetime(rng["start"], errors="coerce")
        end = pd.to_datetime(rng["end"], errors="coerce")
        if pd.isna(start) or pd.isna(end):
            print(f"跳过周期 {key}，start/end 无法解析：{rng}")
            continue
        res = summarize_for_period(df, group_col, pay_col, refund_col, start, end, args.window_days)
        # 仅保留与周期同名的车型分组（大小写/符号归一化比较）
        norm_key = _normalize(key)
        res = res[res[group_col].astype(str).map(_normalize) == norm_key]
        # 若该周期没有对应车型数据，补一行零值，保持输出一致性
        if res.empty:
            pay_col_name = f"payment_d1_d{args.window_days}"
            refund_col_name = f"refund_d1_d{args.window_days}"
            ratio_col_name = f"refund_to_payment_ratio_d1_d{args.window_days}"
            res = pd.DataFrame({
                group_col: [key],
                pay_col_name: [0],
                refund_col_name: [0],
                "payment_full_period": [0],
                ratio_col_name: [pd.NA],
            })
        res.insert(0, "period", key)
        res.insert(1, "start", start.date())
        res.insert(2, "end", end.date())
        all_rows.append(res)

    if not all_rows:
        raise RuntimeError("无可用周期统计结果")

    final = pd.concat(all_rows, axis=0, ignore_index=True)
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(save_path, index=False)

    print(f"\n已保存统计结果: {save_path}")
    print("\n示例输出（前20行）：")
    print(final.head(20).to_string(index=False))


if __name__ == "__main__":
    main()