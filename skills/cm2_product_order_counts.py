from pathlib import Path
from typing import Dict, List

try:
    import pandas as pd
except Exception:
    raise RuntimeError("请先安装依赖：pip install pandas pyarrow numpy")


ORIGINAL_DIR = Path("/Users/zihao_/Documents/coding/dataset/original")


def find_latest_cm2_csv(dir_path: Path) -> Path:
    candidates = sorted(dir_path.glob("CM2_Configuration_Details_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            "未在目录中找到匹配的 CM2_Configuration_Details_*.csv 文件"
        )
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def _read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e
            continue
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        raise e if last_error is None else last_error


def _normalize(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def _resolve_column(df: pd.DataFrame, synonyms: List[str]) -> str:
    target_norm = {_normalize(n) for n in synonyms}
    mapping = { _normalize(c): c for c in df.columns }
    for key_norm, original in mapping.items():
        if key_norm in target_norm:
            return original
    # 二次匹配：包含关系
    for key_norm, original in mapping.items():
        if any(key_norm.find(n) != -1 for n in target_norm):
            return original
    raise KeyError(
        f"未找到匹配列，期望列之一：{synonyms}，实际列：{list(df.columns)}"
    )


def compute_product_order_counts(df: pd.DataFrame) -> pd.DataFrame:
    product_synonyms = [
        "Product Name", "product_name", "ProductName", "产品名称", "商品名称", "品名"
    ]
    order_synonyms = [
        "order_number", "Order Number", "orderNo", "订单号", "订单编号", "订单编码"
    ]

    product_col = _resolve_column(df, product_synonyms)
    order_col = _resolve_column(df, order_synonyms)

    work = df[[product_col, order_col]].copy()
    # 清理缺失
    work[product_col] = work[product_col].astype(str).str.strip()
    work = work[(work[product_col].notna()) & (work[product_col] != "")]
    work = work[work[order_col].notna()]

    # 去重计数（count distinct）
    result = (
        work.groupby(product_col)[order_col]
            .nunique(dropna=True)
            .reset_index(name="order_count_distinct")
            .sort_values(["order_count_distinct", product_col], ascending=[False, True])
    )
    return result


def print_markdown_table(result: pd.DataFrame, product_col: str):
    print("| Product Name | 订单数（去重 order_number）|")
    print("|--------------|---------------------------|")
    for _, row in result.iterrows():
        print(f"| {row[product_col]} | {int(row['order_count_distinct'])} |")


def main():
    latest_csv = find_latest_cm2_csv(ORIGINAL_DIR)
    df = _read_csv_with_fallback(latest_csv)
    result = compute_product_order_counts(df)
    print(f"最新文件: {latest_csv}")
    if result.empty:
        print("无统计结果（可能缺少有效的产品或订单号数据）")
        return
    product_col = result.columns[0]
    print_markdown_table(result, product_col)


if __name__ == "__main__":
    main()