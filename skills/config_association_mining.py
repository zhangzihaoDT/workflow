from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

try:
    from mlxtend.frequent_patterns import apriori, association_rules
except Exception as e:
    raise RuntimeError("mlxtend 未安装，请先运行: .venv/bin/python -m pip install mlxtend") from e


DEFAULT_DIR = Path("/Users/zihao_/Documents/coding/dataset/processed")
BUSINESS_DEF_PATH = Path("/Users/zihao_/Documents/github/W35_workflow/business_definition.json")


def find_latest_transposed_csv(dir_path: Path) -> Path:
    candidates = sorted(dir_path.glob("CM2_Configuration_Details_transposed_*.csv"))
    if not candidates:
        raise FileNotFoundError("未在目录中找到匹配的 CM2_Configuration_Details_transposed_*.csv 文件")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


DEFAULT_INPUT = find_latest_transposed_csv(DEFAULT_DIR)
OUTPUT_DIR = Path("/Users/zihao_/Documents/github/W35_workflow/models")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 统一列名，去除前后空格
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _normalize(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


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


def load_battery_capacity_mapping(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    bc = data.get("battery_capacity", {})
    name_to_capacity: dict = {}
    for cap, names in bc.items():
        for n in names:
            name_to_capacity[str(n).strip()] = cap
    return name_to_capacity


def assign_battery_capacity(df: pd.DataFrame, product_col: str, mapping_path: Path) -> pd.DataFrame:
    name_to_capacity = load_battery_capacity_mapping(mapping_path)
    df[product_col] = df[product_col].astype(str).str.strip()
    df["battery_capacity"] = df[product_col].map(name_to_capacity).fillna("unknown")
    return df


def value_counts_ratio(s: pd.Series) -> pd.Series:
    vc = s.value_counts(dropna=False)
    return vc / max(1, len(s))


def one_hot_encode(df: pd.DataFrame, min_support: float, max_categories_per_col: int = 50) -> pd.DataFrame:
    n = len(df)
    items: List[pd.DataFrame] = []

    # 允许排除某些列（例如 Product Name、battery_capacity 等）
    exclude_columns = getattr(one_hot_encode, "_exclude_columns", None)
    exclude_set = set(exclude_columns or [])

    for col in df.columns:
        if col in exclude_set:
            continue
        s = df[col]
        # 处理布尔/二值
        if pd.api.types.is_bool_dtype(s):
            items.append(pd.DataFrame({f"{col}=True": s.astype(bool)}))
            continue
        # 统一字符串
        if pd.api.types.is_numeric_dtype(s):
            unique_vals = s.dropna().unique()
            # 二值数值
            if set(np.unique(unique_vals)).issubset({0, 1}):
                items.append(pd.DataFrame({f"{col}=1": (s.fillna(0) == 1)}))
                continue
            # 小基数整数 -> 视为类别
            if pd.api.types.is_integer_dtype(s) and len(unique_vals) <= 10:
                s = s.astype('Int64').astype(str)
            else:
                # 连续型 -> 分箱
                try:
                    binned = pd.qcut(s, q=3, duplicates='drop')
                    s = binned.astype(str)
                except Exception:
                    s = s.astype(str)
        else:
            s = s.astype(str)

        # 生成 one-hot items，限制每列最大类别数且过滤低支持度类别
        ratios = value_counts_ratio(s)
        # 过滤支持度过低
        keep_vals = [v for v, r in ratios.items() if r >= min_support]
        # 限制类别数量
        keep_vals = keep_vals[:max_categories_per_col]
        for val in keep_vals:
            item_col = f"{col}={val}"
            items.append(pd.DataFrame({item_col: (s == val)}))

    # 合并为布尔矩阵
    if not items:
        raise RuntimeError("无法生成任何项集，请检查输入数据格式")
    X = pd.concat(items, axis=1)
    # 转换为布尔类型并填充缺失
    X = X.fillna(False).astype(bool)
    return X


def main():
    parser = argparse.ArgumentParser(description="配置数据关联规则挖掘")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="输入CSV路径")
    parser.add_argument("--min_support", type=float, default=0.05, help="频繁项最小支持度")
    parser.add_argument("--min_confidence", type=float, default=0.6, help="规则最小置信度")
    parser.add_argument("--max_len", type=int, default=3, help="频繁项最大长度")
    parser.add_argument("--max_categories_per_col", type=int, default=50, help="每列最多编码的类别数量")
    args = parser.parse_args()

    input_path = Path(args.input)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"读取数据: {input_path}")
    df = load_data(input_path)
    print(f"原始维度: {df.shape}")

    # 解析产品列并加入电池容量特征
    try:
        product_col = resolve_column(df, [
            "Product Name", "product_name", "ProductName", "产品名称", "商品名称", "品名"
        ])
        df = assign_battery_capacity(df, product_col, BUSINESS_DEF_PATH)
        cap_counts = df["battery_capacity"].value_counts()
        print("电池容量分布：")
        for cap, cnt in cap_counts.items():
            print(f"- {cap}: {cnt}")
    except Exception as e:
        print(f"电池容量特征加入失败：{e}")

    # 在整体挖掘中移除 Product Name 以避免其主导规则
    one_hot_encode._exclude_columns = [product_col]
    X_bool = one_hot_encode(df, min_support=args.min_support, max_categories_per_col=args.max_categories_per_col)
    print(f"编码后维度: {X_bool.shape}")

    # 频繁项挖掘
    freq_items = apriori(X_bool, min_support=args.min_support, use_colnames=True, max_len=args.max_len)
    freq_items = freq_items.sort_values(["support", "itemsets"], ascending=[False, True])

    # 关联规则
    rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
    # 过滤置信度
    rules = rules[rules["confidence"] >= args.min_confidence].copy()
    # 友好显示项集
    def set_to_str(s):
        return " & ".join(sorted(list(s)))
    rules["antecedents_str"] = rules["antecedents"].apply(set_to_str)
    rules["consequents_str"] = rules["consequents"].apply(set_to_str)
    rules = rules.sort_values(["lift", "confidence"], ascending=[False, False])

    freq_path = OUTPUT_DIR / "cm2_config_frequent_itemsets.csv"
    rules_path = OUTPUT_DIR / "cm2_config_association_rules.csv"
    freq_items.to_csv(freq_path, index=False)
    # 仅导出必要列
    export_cols = [
        "antecedents_str", "consequents_str", "support", "confidence", "lift", "leverage", "conviction"
    ]
    rules[export_cols].to_csv(rules_path, index=False)
    print(f"已保存频繁项集: {freq_path}")
    print(f"已保存关联规则: {rules_path}")

    # 打印Top规则
    top_n = min(20, len(rules))
    if top_n > 0:
        print("Top 关联规则（按lift排序）：")
        for i in range(top_n):
            r = rules.iloc[i]
            print(f"[{i+1}] {r['antecedents_str']} => {r['consequents_str']} | support={r['support']:.3f}, confidence={r['confidence']:.3f}, lift={r['lift']:.3f}")
        # 按容量分组输出 TopN 规则（在分组挖掘中排除 Product Name 和 battery_capacity）
        target_caps = ["52kwh", "66kwh", "76kwh", "103kwh"]
        for cap in target_caps:
            df_cap = df[df["battery_capacity"] == cap].copy()
            if df_cap.empty:
                continue
            print(f"\n容量分组 {cap} 的 Top 规则：")
            # 为子集设置排除列
            one_hot_encode._exclude_columns = [product_col, "battery_capacity"]
            try:
                X_cap = one_hot_encode(df_cap, min_support=args.min_support, max_categories_per_col=args.max_categories_per_col)
                freq_cap = apriori(X_cap, min_support=args.min_support, use_colnames=True, max_len=args.max_len)
                if len(freq_cap) == 0:
                    print("(该容量分组无频繁项，考虑降低 min_support)")
                    continue
                rules_cap = association_rules(freq_cap, metric="lift", min_threshold=1.0)
                rules_cap = rules_cap[rules_cap["confidence"] >= args.min_confidence].copy()
                # 友好显示
                def set_to_str(s):
                    return " & ".join(sorted(list(s)))
                rules_cap["antecedents_str"] = rules_cap["antecedents"].apply(set_to_str)
                rules_cap["consequents_str"] = rules_cap["consequents"].apply(set_to_str)
                rules_cap = rules_cap.sort_values(["lift", "confidence"], ascending=[False, False])
                top_m = min(10, len(rules_cap))
                if top_m == 0:
                    print("(该容量分组无满足阈值的规则，考虑降低 min_confidence)")
                    continue
                for i in range(top_m):
                    r = rules_cap.iloc[i]
                    print(f"[{cap}-{i+1}] {r['antecedents_str']} => {r['consequents_str']} | support={r['support']:.3f}, confidence={r['confidence']:.3f}, lift={r['lift']:.3f}")
            except Exception as e:
                print(f"容量分组 {cap} 挖掘失败：{e}")
    else:
        print("规则为空，请尝试降低 min_support 或 min_confidence")


if __name__ == "__main__":
    main()