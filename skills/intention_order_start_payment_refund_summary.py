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


def summarize_lock_view_for_period(
    df: pd.DataFrame,
    group_col: str,
    pay_col: str,
    refund_col: str,
    lock_col: str,
    deposit_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    window_days: int,
) -> pd.DataFrame:
    """
    锁单视角（以业务周期 end 为观察起点，统计 end 后 1～N 天）：
    - 锁单数（[end, end+N) 半开区间，Lock_Time 非空）
    - 小订数（支付时间在 [start, end]）
    - 小订留存锁单数（同时满足支付在 [start, end] 与锁单在 [end, end+N)）
    - 小订转化率 = 留存锁单数 / 小订数（百分比，保留一位小数）
    - 小订退订数（支付在 [start, end]，且退款时间 < end+N，且支付时间早于退款时间）
    - 小订退订率 = 退订数 / 小订数（百分比，保留一位小数）
    """
    if window_days < 1:
        raise ValueError("window_days 必须为 >=1 的整数")

    d_end = end
    d_end_plus_n = end + pd.Timedelta(days=window_days)

    # 基础切片
    df_lock = df[~df[lock_col].isna()].copy()
    df_pay = df[~df[pay_col].isna()].copy()
    df_refund = df[~df[refund_col].isna()].copy()
    df_deposit = df[~df[deposit_col].isna()].copy()

    # 锁单窗口：[end, end+N)（半开区间，上界不含）
    in_end_dN_lock = df_lock[(df_lock[lock_col] >= d_end) & (df_lock[lock_col] < d_end_plus_n)]
    g_lock = in_end_dN_lock.groupby(group_col).size().rename(f"locks_dend_d{window_days}")

    # 大定：Deposit_Payment_Time 在 [end, end+N)
    in_end_dN_deposit = df_deposit[(df_deposit[deposit_col] >= d_end) & (df_deposit[deposit_col] < d_end_plus_n)]
    g_deposit = in_end_dN_deposit.groupby(group_col).size().rename(f"deposit_dend_d{window_days}")

    # 全周期大定/小订数：满足 1) 大定在 [end, end+N) 或 2) 小订在 [start, end+N)
    mask_deposit_window = (~df[deposit_col].isna()) & (df[deposit_col] >= d_end) & (df[deposit_col] < d_end_plus_n)
    mask_pay_ext_window = (~df[pay_col].isna()) & (df[pay_col] >= start) & (df[pay_col] < d_end_plus_n)
    union_df = df[mask_deposit_window | mask_pay_ext_window]
    g_union = union_df.groupby(group_col).size().rename(f"full_deposit_or_intention_dend_d{window_days}")

    # 小订（支付在全周期 [start, end]）
    in_full_pay = df_pay[(df_pay[pay_col] >= start) & (df_pay[pay_col] <= end)]
    g_pay = in_full_pay.groupby(group_col).size().rename("payment_full_period")

    # 留存锁单：支付在 [start, end] 且锁单在 [end, end+N)（半开区间）
    df_pay_lock = df[(~df[pay_col].isna()) & (~df[lock_col].isna())].copy()
    mask_pay_period = (df_pay_lock[pay_col] >= start) & (df_pay_lock[pay_col] <= end)
    mask_lock_window = (df_pay_lock[lock_col] >= d_end) & (df_pay_lock[lock_col] < d_end_plus_n)
    retained = df_pay_lock[mask_pay_period & mask_lock_window]
    g_retained = retained.groupby(group_col).size().rename(f"retained_locks_dend_d{window_days}")

    # 退订：支付在 [start, end] 且退款 < end+N，且支付时间早于退款时间
    df_refund_paid = df_refund[~df_refund[pay_col].isna()].copy()
    mask_pay_period_r = (df_refund_paid[pay_col] >= start) & (df_refund_paid[pay_col] <= end)
    mask_refund_window = df_refund_paid[refund_col] < d_end_plus_n
    mask_pay_before_refund = df_refund_paid[pay_col] < df_refund_paid[refund_col]
    refund_in_window = df_refund_paid[mask_pay_period_r & mask_refund_window & mask_pay_before_refund]
    g_refund = refund_in_window.groupby(group_col).size().rename(f"refund_by_end_plus_{window_days}d")

    out = pd.concat([g_lock, g_deposit, g_union, g_pay, g_retained, g_refund], axis=1).fillna(0).astype(int).reset_index()

    # 比率列（保留一位小数，百分比格式）
    conv_ratio = (
        out[g_retained.name].astype("Float64")
        / out[g_pay.name].replace({0: pd.NA}).astype("Float64")
    )
    refund_ratio = (
        out[g_refund.name].astype("Float64")
        / out[g_pay.name].replace({0: pd.NA}).astype("Float64")
    )
    out[f"conversion_rate_dend_d{window_days}"] = conv_ratio.astype("Float64").mul(100).round(1).map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else pd.NA
    )
    out[f"refund_rate_by_end_plus_{window_days}d"] = refund_ratio.astype("Float64").mul(100).round(1).map(
        lambda x: f"{x:.1f}%" if pd.notna(x) else pd.NA
    )

    return out


def main():
    parser = argparse.ArgumentParser(description="按车型分组输出各业务周期起始第1～N天及全周期的支付/退订统计")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="数据文件路径（parquet）")
    parser.add_argument("--business", default=str(BUSINESS_DEF_PATH), help="business_definition.json 路径")
    parser.add_argument("--save", default=str(ROOT_DIR / "models" / "period_start_payment_refund_summary.csv"), help="导出CSV路径")
    parser.add_argument("--window_days", type=int, default=3, help="起始 N 天窗口（含起始日，统计范围为 [start, start+N) ）")
    parser.add_argument("--save_lock", default=str(ROOT_DIR / "models" / "period_end_lock_conversion_summary.csv"), help="锁单视角导出CSV路径")
    parser.add_argument("--save_html", default=str(ROOT_DIR / "models" / "period_summary.html"), help="导出HTML页面路径")
    parser.add_argument("--gradio", action="store_true", help="启动 Gradio Dataframe 交互界面")
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
    lock_col = resolve_column(df, [
        "lock_time", "Lock_Time", "锁单时间", "锁单日期"
    ])
    deposit_col = resolve_column(df, [
        "Deposit_Payment_Time", "deposit_payment_time", "大定支付时间", "大定时间"
    ])

    # 转为时间类型
    df[pay_col] = coerce_datetime(df[pay_col])
    df[refund_col] = coerce_datetime(df[refund_col])
    df[lock_col] = coerce_datetime(df[lock_col])
    df[deposit_col] = coerce_datetime(df[deposit_col])

    # 输出车型分组所有取值
    unique_groups = sorted(df[group_col].astype(str).unique())
    print("车型分组所有取值：")
    for v in unique_groups:
        print(f"- {v}")

    # 逐周期统计
    all_rows: List[pd.DataFrame] = []
    all_lock_rows: List[pd.DataFrame] = []
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

        # 锁单视角统计
        lock_res = summarize_lock_view_for_period(df, group_col, pay_col, refund_col, lock_col, deposit_col, start, end, args.window_days)
        lock_res = lock_res[lock_res[group_col].astype(str).map(_normalize) == norm_key]
        if lock_res.empty:
            lock_col_name = f"locks_dend_d{args.window_days}"
            deposit_col_name = f"deposit_dend_d{args.window_days}"
            union_col_name = f"full_deposit_or_intention_dend_d{args.window_days}"
            retained_col_name = f"retained_locks_dend_d{args.window_days}"
            refund_col_name2 = f"refund_by_end_plus_{args.window_days}d"
            conv_ratio_name = f"conversion_rate_dend_d{args.window_days}"
            refund_ratio_name = f"refund_rate_by_end_plus_{args.window_days}d"
            lock_res = pd.DataFrame({
                group_col: [key],
                lock_col_name: [0],
                deposit_col_name: [0],
                union_col_name: [0],
                "payment_full_period": [0],
                retained_col_name: [0],
                refund_col_name2: [0],
                conv_ratio_name: [pd.NA],
                refund_ratio_name: [pd.NA],
            })
        lock_res.insert(0, "period", key)
        lock_res.insert(1, "start", start.date())
        lock_res.insert(2, "end", end.date())
        all_lock_rows.append(lock_res)

    if not all_rows:
        raise RuntimeError("无可用周期统计结果")

    final = pd.concat(all_rows, axis=0, ignore_index=True)
    final_lock = pd.concat(all_lock_rows, axis=0, ignore_index=True)

    # 列名中文化（含动态 N 天）
    start_pay_col = f"payment_d1_d{args.window_days}"
    start_refund_col = f"refund_d1_d{args.window_days}"
    start_ratio_col = f"refund_to_payment_ratio_d1_d{args.window_days}"
    lock_col_name = f"locks_dend_d{args.window_days}"
    retained_col_name = f"retained_locks_dend_d{args.window_days}"
    deposit_col_name = f"deposit_dend_d{args.window_days}"
    union_col_name = f"full_deposit_or_intention_dend_d{args.window_days}"
    refund_by_end_col = f"refund_by_end_plus_{args.window_days}d"
    conv_ratio_name = f"conversion_rate_dend_d{args.window_days}"
    refund_ratio_name = f"refund_rate_by_end_plus_{args.window_days}d"

    start_renames = {
        "period": "周期",
        "start": "开始日期",
        "end": "结束日期",
        group_col: "车型分组",
        start_pay_col: f"起始{args.window_days}日小订数",
        start_refund_col: f"起始{args.window_days}日退订数",
        "payment_full_period": "全周期小订数",
        start_ratio_col: f"起始{args.window_days}日退订率",
    }

    lock_renames = {
        "period": "周期",
        "start": "开始日期",
        "end": "结束日期",
        group_col: "车型分组",
        lock_col_name: f"上市后{args.window_days}日锁单数",
        deposit_col_name: f"上市后{args.window_days}日大定数",
        union_col_name: "全周期大定/小订数",
        "payment_full_period": "全周期小订数",
        retained_col_name: "小订留存锁单数",
        refund_by_end_col: f"上市后{args.window_days}日小订退订数",
        conv_ratio_name: "小订转化率",
        refund_ratio_name: "小订退订率",
    }

    final_cn = final.rename(columns=start_renames)
    final_lock_cn = final_lock.rename(columns=lock_renames)

    save_path = Path(args.save)
    save_lock_path = Path(args.save_lock)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_lock_path.parent.mkdir(parents=True, exist_ok=True)

    final_cn.to_csv(save_path, index=False)
    final_lock_cn.to_csv(save_lock_path, index=False)

    print(f"\n已保存起始窗口统计结果: {save_path}")
    print("示例输出（前20行）：")
    print(final_cn.head(20).to_string(index=False))

    print(f"\n已保存锁单视角统计结果: {save_lock_path}")
    print("示例输出（前20行）：")
    print(final_lock_cn.head(20).to_string(index=False))

    # 生成HTML页面，包含两个表格和CSV下载链接
    html_path = Path(args.save_html)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    style = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif; margin: 24px; }
      h1 { font-size: 20px; margin-bottom: 12px; }
      h2 { font-size: 18px; margin-top: 20px; }
      .desc { color: #555; margin-bottom: 12px; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }
      th { background: #f9fafb; }
      .links { margin: 12px 0; }
      .links a { margin-right: 16px; }
    </style>
    """
    html_content = f"""
    <!doctype html>
    <html lang=\"zh-CN\">
    <head>
      <meta charset=\"utf-8\" />
      <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
      <title>周期统计与锁单视角概览</title>
      {style}
    </head>
    <body>
      <h1>周期统计与锁单视角概览</h1>
      <div class=\"desc\">本页包含两类统计：起始窗口（[start, start+N)）的支付/退订对比与上市后窗口（[end, end+N)）的锁单/留存/退订对比。</div>

      <h2>起始窗口统计（支付/退订）</h2>
      <div class=\"links\">
        <a href=\"{save_path.name}\" download>下载 CSV：起始窗口统计</a>
      </div>
      {final_cn.to_html(index=False)}

      <h2>锁单视角统计（锁单/留存/退订）</h2>
      <div class=\"links\">
        <a href=\"{save_lock_path.name}\" download>下载 CSV：锁单视角统计</a>
      </div>
      {final_lock_cn.to_html(index=False)}
    </body>
    </html>
    """
    # HTML 文件与CSV位于同一目录，便于下载链接直接使用文件名
    with html_path.open("w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n已保存HTML页面: {html_path}")

    # 若启用 Gradio 界面，则提供交互式 Dataframe 展示与下载
    if args.gradio:
        try:
            import gradio as gr
        except Exception as e:
            print(f"Gradio 导入失败或未安装：{e}")
            return

        # 复用已加载的数据与列解析，提供可调整 N 的重计算函数
        def compute_tables_for_n(n: int):
            all_rows_local: List[pd.DataFrame] = []
            all_lock_rows_local: List[pd.DataFrame] = []
            for key, rng in periods.items():
                start_local = pd.to_datetime(rng["start"], errors="coerce")
                end_local = pd.to_datetime(rng["end"], errors="coerce")
                if pd.isna(start_local) or pd.isna(end_local):
                    continue
                res_local = summarize_for_period(df, group_col, pay_col, refund_col, start_local, end_local, n)
                norm_key_local = _normalize(key)
                res_local = res_local[res_local[group_col].astype(str).map(_normalize) == norm_key_local]
                if res_local.empty:
                    pay_col_name_local = f"payment_d1_d{n}"
                    refund_col_name_local = f"refund_d1_d{n}"
                    ratio_col_name_local = f"refund_to_payment_ratio_d1_d{n}"
                    res_local = pd.DataFrame({
                        group_col: [key],
                        pay_col_name_local: [0],
                        refund_col_name_local: [0],
                        "payment_full_period": [0],
                        ratio_col_name_local: [pd.NA],
                    })
                res_local.insert(0, "period", key)
                res_local.insert(1, "start", start_local.date())
                res_local.insert(2, "end", end_local.date())
                all_rows_local.append(res_local)
                lock_res_local = summarize_lock_view_for_period(df, group_col, pay_col, refund_col, lock_col, deposit_col, start_local, end_local, n)
                lock_res_local = lock_res_local[lock_res_local[group_col].astype(str).map(_normalize) == norm_key_local]
                if lock_res_local.empty:
                    lock_col_name_local = f"locks_dend_d{n}"
                    deposit_col_name_local = f"deposit_dend_d{n}"
                    union_col_name_local = f"full_deposit_or_intention_dend_d{n}"
                    retained_col_name_local = f"retained_locks_dend_d{n}"
                    refund_col_name2_local = f"refund_by_end_plus_{n}d"
                    conv_ratio_name_local = f"conversion_rate_dend_d{n}"
                    refund_ratio_name_local = f"refund_rate_by_end_plus_{n}d"
                    lock_res_local = pd.DataFrame({
                        group_col: [key],
                        lock_col_name_local: [0],
                        deposit_col_name_local: [0],
                        union_col_name_local: [0],
                        "payment_full_period": [0],
                        retained_col_name_local: [0],
                        refund_col_name2_local: [0],
                        conv_ratio_name_local: [pd.NA],
                        refund_ratio_name_local: [pd.NA],
                    })
                lock_res_local.insert(0, "period", key)
                lock_res_local.insert(1, "start", start_local.date())
                lock_res_local.insert(2, "end", end_local.date())
                all_lock_rows_local.append(lock_res_local)

            final_local = pd.concat(all_rows_local, axis=0, ignore_index=True)
            final_lock_local = pd.concat(all_lock_rows_local, axis=0, ignore_index=True)

            start_pay_col_local = f"payment_d1_d{n}"
            start_refund_col_local = f"refund_d1_d{n}"
            start_ratio_col_local = f"refund_to_payment_ratio_d1_d{n}"
            lock_col_name_local = f"locks_dend_d{n}"
            retained_col_name_local = f"retained_locks_dend_d{n}"
            deposit_col_name_local = f"deposit_dend_d{n}"
            union_col_name_local = f"full_deposit_or_intention_dend_d{n}"
            refund_by_end_col_local = f"refund_by_end_plus_{n}d"
            conv_ratio_name_local = f"conversion_rate_dend_d{n}"
            refund_ratio_name_local = f"refund_rate_by_end_plus_{n}d"

            start_renames_local = {
                "period": "周期",
                "start": "开始日期",
                "end": "结束日期",
                group_col: "车型分组",
                start_pay_col_local: f"起始{n}日小订数",
                start_refund_col_local: f"起始{n}日退订数",
                "payment_full_period": "全周期小订数",
                start_ratio_col_local: f"起始{n}日退订率",
            }

            lock_renames_local = {
                "period": "周期",
                "start": "开始日期",
                "end": "结束日期",
                group_col: "车型分组",
                lock_col_name_local: f"上市后{n}日锁单数",
                deposit_col_name_local: f"上市后{n}日大定数",
                union_col_name_local: "全周期大定/小订数",
                "payment_full_period": "全周期小订数",
                retained_col_name_local: "小订留存锁单数",
                refund_by_end_col_local: f"上市后{n}日小订退订数",
                conv_ratio_name_local: "小订转化率",
                refund_ratio_name_local: "小订退订率",
            }

            final_cn_local = final_local.rename(columns=start_renames_local)
            final_lock_cn_local = final_lock_local.rename(columns=lock_renames_local)

            # 覆写同一路径，保持下载按钮始终可用
            final_cn_local.to_csv(save_path, index=False)
            final_lock_cn_local.to_csv(save_lock_path, index=False)
            return final_cn_local, final_lock_cn_local, str(save_path), str(save_lock_path)

        with gr.Blocks(title="周期统计与锁单视角概览") as demo:
            gr.Markdown("### 起始窗口与上市后视角统计\n- 支持调整 N 并实时更新表格\n- 表格支持复制与滚动查看")
            n_slider = gr.Slider(minimum=1, maximum=30, step=1, value=args.window_days, label="窗口天数 N")
            df_start = gr.Dataframe(value=final_cn, label="起始窗口统计（支付/退订）", interactive=True)
            df_lock = gr.Dataframe(value=final_lock_cn, label="上市后统计（锁单/留存/退订）", interactive=True)
            dl1 = gr.DownloadButton("下载起始窗口CSV", value=str(save_path))
            dl2 = gr.DownloadButton("下载上市后窗口CSV", value=str(save_lock_path))

            def on_change(n):
                fc, fl, p1, p2 = compute_tables_for_n(int(n))
                return fc, fl, p1, p2

            n_slider.change(on_change, inputs=n_slider, outputs=[df_start, df_lock, dl1, dl2])

        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
        return


if __name__ == "__main__":
    main()