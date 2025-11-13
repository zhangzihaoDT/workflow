#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按天聚合：小订数 vs 上市后锁单数，并做 Granger 因果检验（小订→锁单）。

过滤/聚合逻辑（按车型周期 business_definition.json#time_periods）：
- 小订数（订阅/支付）：Intention_Payment_Time ∈ [start, end]（含端点），按天计数
- 锁单数：Lock_Time ∈ [end, end+30)（半开区间），按天计数
- 汇总区间：连续日期范围 [start, end+30)，两组序列按天对齐，不在窗口的天数记为 0
- 排除车型：LS7、L7
- 车型范围：CM0、CM1、CM2、DM0、DM1、LS9

输出：
- CSV：models/granger_subscription_lock_daily.csv（包含车型、日期、两个计数）
- HTML：models/granger_subscription_lock_daily.html（包含可视化与各车型 Granger 结果）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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


def aggregate_daily_series(
    df: pd.DataFrame,
    group_col: str,
    pay_col: str,
    lock_col: str,
    key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lock_window_days: int = 30,
) -> pd.DataFrame:
    """返回指定车型在 [start, end+30) 的每日小订与锁单计数（未落在各自窗口的天数填 0）。"""
    norm_key = _normalize(key)
    df_k = df[df[group_col].astype(str).map(_normalize) == norm_key].copy()
    df_k[pay_col] = coerce_datetime(df_k[pay_col])
    df_k[lock_col] = coerce_datetime(df_k[lock_col])

    # 定义日期范围
    range_start = start.normalize()
    range_end_inclusive = (end + pd.Timedelta(days=lock_window_days) - pd.Timedelta(days=1)).normalize()
    days = pd.date_range(range_start, range_end_inclusive, freq="D")

    # 小订窗口：[start, end]（含端点）
    pay_mask = df_k[pay_col].notna() & (df_k[pay_col] >= start) & (df_k[pay_col] <= end)
    sub_daily = (
        df_k.loc[pay_mask, pay_col].dt.normalize().value_counts().rename("subscription_count")
    )
    # 锁单窗口：[end, end+30)（半开区间）
    lock_start = end
    lock_end = end + pd.Timedelta(days=lock_window_days)
    lock_mask = df_k[lock_col].notna() & (df_k[lock_col] >= lock_start) & (df_k[lock_col] < lock_end)
    lock_daily = (
        df_k.loc[lock_mask, lock_col].dt.normalize().value_counts().rename("lock_count")
    )

    # 合并并补全日期，缺失填 0
    out = pd.DataFrame({"date": days})
    out = out.merge(sub_daily.rename_axis("date").reset_index(), on="date", how="left")
    out = out.merge(lock_daily.rename_axis("date").reset_index(), on="date", how="left")
    out["subscription_count"] = out["subscription_count"].fillna(0).astype(int)
    out["lock_count"] = out["lock_count"].fillna(0).astype(int)
    out.insert(0, "车型分组", key)
    return out


def granger_test(series_sub: np.ndarray, series_lock: np.ndarray, max_lag: int = 7) -> Dict[int, Dict[str, float]]:
    """对给定的两个序列执行 Granger 检验，返回各滞后阶的主检验 p 值。
    采用 statsmodels 的 ssr_chi2test 与 f_test 的 p 值作为参考。
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except Exception as e:
        return {0: {"error": f"未安装或无法导入 statsmodels: {e}"}}

    # 目标：锁单是否被小订 Granger 导致，y=lock，x=subscription
    x = pd.Series(series_sub).astype(float)
    y = pd.Series(series_lock).astype(float)
    # 移除两者同时为0的日期以降低完美拟合风险
    mask_keep = ~((x.fillna(0) == 0) & (y.fillna(0) == 0))
    x_trim = x[mask_keep]
    y_trim = y[mask_keep]
    # 若样本过短，降低滞后阶，至少保证 1 阶
    nobs = int(min(len(x_trim.dropna()), len(y_trim.dropna())))
    if nobs < 5:
        return {0: {"error": f"有效样本不足（{nobs}），无法进行因果检验"}}
    use_max_lag = int(min(max_lag, max(1, nobs - 2)))
    data = np.column_stack([y_trim.to_numpy(), x_trim.to_numpy()])
    try:
        results = grangercausalitytests(data, maxlag=use_max_lag, verbose=False)
    except Exception as e:
        # 无法计算时返回各滞后阶的空结果
        return {lag: {"f_stat": np.nan, "p_f": np.nan, "p_ssr_chi2": np.nan, "error": str(e)} for lag in range(1, use_max_lag + 1)}

    out: Dict[int, Dict[str, float]] = {}
    for lag, res in results.items():
        tests = res[0]  # dict of test results
        ssr_chi2 = tests.get("ssr_chi2test", (np.nan, np.nan, np.nan))
        ssr_ftest = tests.get("ssr_ftest", (np.nan, np.nan, np.nan, np.nan))
        f_stat = float(ssr_ftest[0]) if ssr_ftest is not None else np.nan
        p_f = float(ssr_ftest[1]) if ssr_ftest is not None else np.nan
        p_ssr_chi2 = float(ssr_chi2[1]) if ssr_chi2 is not None else np.nan
        out[lag] = {"f_stat": f_stat, "p_f": p_f, "p_ssr_chi2": p_ssr_chi2}
    return out


def adf_test(series: np.ndarray) -> Dict[str, float]:
    """ADF 平稳性检验，返回统计量与 p 值。"""
    try:
        from statsmodels.tsa.stattools import adfuller
    except Exception as e:
        return {"error": f"未安装或无法导入 statsmodels: {e}"}
    # 将序列转换为 pandas Series 以避免全常数触发异常
    s = pd.Series(series).astype(float)
    # 若全常数或长度不足，返回 NaN
    if s.nunique() <= 1 or len(s.dropna()) < 3:
        return {"adf_stat": np.nan, "pvalue": np.nan}
    try:
        stat, pvalue, *_ = adfuller(s.dropna(), autolag="AIC")
        return {"adf_stat": float(stat), "pvalue": float(pvalue)}
    except Exception:
        return {"adf_stat": np.nan, "pvalue": np.nan}


def compute_cross_correlation(series_sub: np.ndarray, series_lock: np.ndarray, max_lag: int = 14) -> pd.DataFrame:
    """计算正滞后下的相关性：corr(sub[:-lag], lock[lag:])，lag=0..max_lag。"""
    s = pd.Series(series_sub).astype(float)
    l = pd.Series(series_lock).astype(float)
    rows = []
    for lag in range(0, max_lag + 1):
        if lag == 0:
            x, y = s, l
        else:
            x, y = s.iloc[:-lag], l.iloc[lag:]
        if len(x) >= 3 and len(y) >= 3 and x.std() > 0 and y.std() > 0:
            corr = float(pd.Series(x).corr(pd.Series(y)))
        else:
            corr = np.nan
        rows.append({"lag": lag, "corr": corr})
    return pd.DataFrame(rows)


def build_html(
    daily_all: pd.DataFrame,
    granger_all: Dict[str, Dict[int, Dict[str, float]]],
    adf_all: Dict[str, Dict[str, Dict[str, float]]],
    ccf_all: Dict[str, pd.DataFrame],
    save_csv: Path,
) -> str:
    # 生成每车型的折线图（小订 vs 锁单）
    figs_html: List[str] = []
    for key in ["CM0", "CM1", "CM2", "DM0", "DM1", "LS9"]:
        df_k = daily_all[daily_all["车型分组"] == key].copy()
        if df_k.empty:
            continue
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_k["date"], y=df_k["subscription_count"], mode="lines+markers", name="小订数"))
        fig.add_trace(go.Scatter(x=df_k["date"], y=df_k["lock_count"], mode="lines+markers", name="锁单数"))
        fig.update_layout(title=f"{key}：小订 vs 锁单（日度）", xaxis_title="日期", yaxis_title="数量", template="plotly_white")
        figs_html.append(fig.to_html(include_plotlyjs="cdn", full_html=False))

    # Granger 结果表
    rows = []
    for key, gmap in granger_all.items():
        for lag, vals in gmap.items():
            rows.append({
                "车型分组": key,
                "滞后阶": lag,
                "F": vals.get("f_stat"),
                "p(F)": vals.get("p_f"),
                "p(ssr_chi2)": vals.get("p_ssr_chi2"),
            })
    granger_df = pd.DataFrame(rows)
    granger_table_html = granger_df.to_html(index=False)

    # ADF 结果表
    adf_rows = []
    for key, series_dict in adf_all.items():
        adf_rows.append({
            "车型分组": key,
            "小订_ADF": series_dict.get("subscription", {}).get("adf_stat"),
            "小订_p": series_dict.get("subscription", {}).get("pvalue"),
            "锁单_ADF": series_dict.get("lock", {}).get("adf_stat"),
            "锁单_p": series_dict.get("lock", {}).get("pvalue"),
        })
    adf_df = pd.DataFrame(adf_rows)
    adf_table_html = adf_df.to_html(index=False)

    # CCF 图（滞后相关）
    ccf_figs_html: List[str] = []
    for key, ccf_df in ccf_all.items():
        fig_ccf = go.Figure()
        fig_ccf.add_trace(go.Bar(x=ccf_df["lag"], y=ccf_df["corr"], name="cross-corr"))
        fig_ccf.update_layout(title=f"{key}：滞后相关（小订→锁单）", xaxis_title="滞后天数", yaxis_title="相关系数", template="plotly_white")
        ccf_figs_html.append(fig_ccf.to_html(include_plotlyjs="cdn", full_html=False))

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
      .fig { margin: 16px 0; }
    </style>
    """
    html = f"""
    <!doctype html>
    <html lang=\"zh-CN\">
    <head>
      <meta charset=\"utf-8\" />
      <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
      <title>Granger：小订是否导致锁单（日度）</title>
      {style}
    </head>
    <body>
      <h1>Granger：小订是否导致锁单（日度）</h1>
      <div class=\"desc\">小订窗口：[start, end]；锁单窗口：[end, end+30)，两序列按天对齐（未覆盖日记为0）。</div>
      <div class=\"links\"><a href=\"{save_csv.name}\" download>下载 CSV（日度序列）</a></div>
      <h2>ADF 平稳性检验（adf_stat / p）</h2>
      {adf_table_html}
      <h2>各车型 Granger 结果（关注 F 与 p(F)）</h2>
      {granger_table_html}
      <h2>对比折线（小订 vs 锁单）</h2>
      {''.join(f'<div class=\"fig\">{h}</div>' for h in figs_html)}
      <h2>滞后相关图（cross-correlation）</h2>
      {''.join(f'<div class=\"fig\">{h}</div>' for h in ccf_figs_html)}
    </body>
    </html>
    """
    return html


def main():
    parser = argparse.ArgumentParser(description="按天聚合并执行 Granger 检验：小订→锁单")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="数据文件路径（parquet）")
    parser.add_argument("--business_def", default=str(BUSINESS_DEF_PATH), help="业务定义JSON（包含各车型time_periods）")
    parser.add_argument("--save_csv", default=str(ROOT_DIR / "models" / "granger_subscription_lock_daily.csv"), help="CSV导出路径")
    parser.add_argument("--save_html", default=str(ROOT_DIR / "models" / "granger_subscription_lock_daily.html"), help="HTML导出路径")
    parser.add_argument("--max_lag", type=int, default=7, help="Granger 最大滞后阶（默认7天）")
    args = parser.parse_args()

    df = load_data(Path(args.data))
    periods = load_business_periods(Path(args.business_def))

    group_col = resolve_column(df, [
        "车型分组", "Product_Types", "产品类型", "车型类别", "车系", "product_types", "product_type"
    ])
    pay_col = resolve_column(df, [
        "Intention_Payment_Time", "intention_payment_time", "支付时间"
    ])
    lock_col = resolve_column(df, [
        "Lock_Time", "lock_time", "锁单时间", "锁单日期"
    ])

    # 排除车型
    exclude_set_norm = {_normalize(x) for x in ["LS7", "L7"]}
    df = df[~df[group_col].astype(str).map(_normalize).isin(exclude_set_norm)].copy()

    target_groups = ["CM0", "CM1", "CM2", "DM0", "DM1", "LS9"]

    # 聚合日度序列
    all_daily: List[pd.DataFrame] = []
    for key in target_groups:
        rng = periods.get(key)
        if not rng:
            print(f"跳过 {key}：未定义周期")
            continue
        start, end = rng.get("start"), rng.get("end")
        if pd.isna(start) or pd.isna(end):
            print(f"跳过 {key}：start/end 解析失败：{rng}")
            continue
        d = aggregate_daily_series(df, group_col, pay_col, lock_col, key, start, end, lock_window_days=30)
        all_daily.append(d)

    if not all_daily:
        raise RuntimeError("没有可用的车型日度序列")

    daily_all = pd.concat(all_daily, axis=0, ignore_index=True)

    # 保存CSV
    save_csv = Path(args.save_csv)
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    daily_all.to_csv(save_csv, index=False, encoding="utf-8-sig")
    print(f"已导出日度CSV: {save_csv}")

    # 执行 ADF 与 Granger 检验（按车型）
    granger_all: Dict[str, Dict[int, Dict[str, float]]] = {}
    adf_all: Dict[str, Dict[str, Dict[str, float]]] = {}
    ccf_all: Dict[str, pd.DataFrame] = {}
    for key in target_groups:
        df_k = daily_all[daily_all["车型分组"] == key].copy()
        if df_k.empty:
            continue
        # 使用全区间 [start, end+30) 的日序列（小订与锁单），可能包含大量 0 值，但方向上合理（小订先行）
        sub = df_k["subscription_count"].to_numpy()
        lock = df_k["lock_count"].to_numpy()
        # ADF 检验
        adf_sub = adf_test(sub)
        adf_lock = adf_test(lock)
        adf_all[key] = {"subscription": adf_sub, "lock": adf_lock}
        # Granger 检验
        res = granger_test(sub, lock, max_lag=args.max_lag)
        granger_all[key] = res
        # CCF（滞后相关）
        ccf_all[key] = compute_cross_correlation(sub, lock, max_lag=max(14, args.max_lag))

    # 构建并导出 ADF/Granger 摘要 CSV
    adf_rows = []
    for key, series_dict in adf_all.items():
        adf_rows.append({
            "车型分组": key,
            "小订_ADF": series_dict.get("subscription", {}).get("adf_stat"),
            "小订_p": series_dict.get("subscription", {}).get("pvalue"),
            "锁单_ADF": series_dict.get("lock", {}).get("adf_stat"),
            "锁单_p": series_dict.get("lock", {}).get("pvalue"),
        })
    adf_df = pd.DataFrame(adf_rows)
    adf_csv = ROOT_DIR / "models" / "adf_summary.csv"
    adf_df.to_csv(adf_csv, index=False, encoding="utf-8-sig")

    granger_rows = []
    for key, gmap in granger_all.items():
        for lag, vals in gmap.items():
            granger_rows.append({
                "车型分组": key,
                "滞后阶": lag,
                "F": vals.get("f_stat"),
                "p(F)": vals.get("p_f"),
                "p(ssr_chi2)": vals.get("p_ssr_chi2"),
            })
    granger_df = pd.DataFrame(granger_rows)
    granger_csv = ROOT_DIR / "models" / "granger_summary.csv"
    granger_df.to_csv(granger_csv, index=False, encoding="utf-8-sig")

    # 生成结论报告（聚焦滞后期=1~3）
    def build_report_html() -> str:
        style = """
        <style>
          body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif; margin: 24px; }
          h1 { font-size: 20px; margin-bottom: 12px; }
          h2 { font-size: 18px; margin-top: 20px; }
          p { color: #333; }
          ul { margin: 8px 0 16px 24px; }
          li { margin: 4px 0; }
          .links a { margin-right: 16px; }
        </style>
        """
        lines = []
        lines.append("<h2>摘要与结论（滞后期=1~3）</h2>")
        lines.append("<p>以下结论基于 ADF 平稳性检验与 Granger 因果检验（方向：小订→锁单）。ADF 若 p<0.05 视为平稳，Granger 以 F 与 p(F) 判断显著性。</p>")
        lines.append("<ul>")
        for key in target_groups:
            adf_row = adf_df[adf_df["车型分组"] == key]
            gr_k = granger_df[(granger_df["车型分组"] == key) & (granger_df["滞后阶"].between(1, 3))]
            if adf_row.empty or gr_k.empty:
                lines.append(f"<li>{key}：数据不足或未能计算。</li>")
                continue
            sub_p = adf_row.iloc[0]["小订_p"]
            lock_p = adf_row.iloc[0]["锁单_p"]
            # 选择 p(F) 最小的滞后阶
            gr_k_valid = gr_k.dropna(subset=["p(F)"])
            if gr_k_valid.empty:
                lines.append(f"<li>{key}：滞后1~3期未获得有效的 Granger 结果。</li>")
                continue
            best = gr_k_valid.loc[gr_k_valid["p(F)"].idxmin()]
            lag = int(best["滞后阶"]) if pd.notna(best["滞后阶"]) else None
            Fv = best["F"]
            Pv = best["p(F)"]
            signif = (pd.notna(Pv) and Pv < 0.05)
            # 交叉相关最高滞后
            ccf_df = ccf_all.get(key)
            corr_note = ""
            if ccf_df is not None and not ccf_df.empty:
                ccf_valid = ccf_df.dropna(subset=["corr"]).copy()
                if not ccf_valid.empty:
                    peak = ccf_valid.loc[ccf_valid["corr"].idxmax()]
                    corr_note = f"；滞后相关峰值出现在 lag={int(peak['lag'])}，corr={peak['corr']:.3f}"
            adf_note = "（两序列均平稳）" if (pd.notna(sub_p) and sub_p < 0.05 and pd.notna(lock_p) and lock_p < 0.05) else "（序列可能非平稳，需谨慎）"
            if signif and lag is not None:
                lines.append(f"<li>{key}：经过 ADF 与 Granger 检验（滞后期=1~3），显示小订在 {lag} 期滞后下对锁单具有显著预测作用（F={Fv:.2f}，p={Pv:.3g}）{adf_note}{corr_note}。</li>")
            else:
                lines.append(f"<li>{key}：滞后1~3期未观察到显著因果作用（最小 p(F)={Pv:.3g}）{adf_note}{corr_note}。</li>")
        lines.append("</ul>")
        links = f"<div class=\"links\"><a href=\"{adf_csv.name}\" download>下载 ADF Summary CSV</a><a href=\"{granger_csv.name}\" download>下载 Granger Summary CSV</a></div>"
        html = f"""
        <!doctype html>
        <html lang=\"zh-CN\">
        <head>
          <meta charset=\"utf-8\" />
          <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
          <title>Granger 报告：ADF 与因果结论</title>
          {style}
        </head>
        <body>
          <h1>Granger 报告：ADF 与因果结论</h1>
          {links}
          {''.join(lines)}
        </body>
        </html>
        """
        return html

    report_html = build_report_html()
    report_path = ROOT_DIR / "models" / "granger_subscription_lock_daily_report.html"
    report_path.write_text(report_html, encoding="utf-8")

    # 生成 HTML（原图页）
    html = build_html(daily_all, granger_all, adf_all, ccf_all, save_csv)
    save_html = Path(args.save_html)
    save_html.parent.mkdir(parents=True, exist_ok=True)
    save_html.write_text(html, encoding="utf-8")
    print(f"已保存HTML页面: {save_html}")
    print(f"已导出摘要CSV: {adf_csv} | {granger_csv}")
    print(f"已生成报告页面: {report_path}")


if __name__ == "__main__":
    main()