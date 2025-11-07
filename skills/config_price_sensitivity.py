from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    XGB_READY = True
except Exception:
    XGB_READY = False

try:
    import shap
    SHAP_READY = True
except Exception:
    SHAP_READY = False


DEFAULT_DIR = Path("/Users/zihao_/Documents/coding/dataset/processed")
BUSINESS_DEF_PATH = Path("/Users/zihao_/Documents/github/W35_workflow/business_definition.json")
OUTPUT_DIR = Path("/Users/zihao_/Documents/github/W35_workflow/models")


def find_latest_transposed_csv(dir_path: Path) -> Path:
    candidates = sorted(dir_path.glob("CM2_Configuration_Details_transposed_*.csv"))
    if not candidates:
        raise FileNotFoundError("未在目录中找到匹配的 CM2_Configuration_Details_transposed_*.csv 文件")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
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


def build_features(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: List[str],
    min_support: float,
    max_categories_per_col: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target_col]
    X_items: List[pd.DataFrame] = []

    for col in df.columns:
        if col == target_col or col in exclude_cols:
            continue
        s = df[col]
        # 数值型：直接加入（填充缺失为中位数）
        if pd.api.types.is_numeric_dtype(s):
            s_num = s.copy()
            try:
                med = float(pd.to_numeric(s_num, errors="coerce").dropna().median())
            except Exception:
                med = 0.0
            s_num = pd.to_numeric(s_num, errors="coerce").fillna(med)
            X_items.append(pd.DataFrame({col: s_num}))
            continue
        # 非数值型：做 one-hot（限制类别数、过滤低支持度）
        s_str = s.astype(str).fillna("NA")
        ratios = value_counts_ratio(s_str)
        keep_vals = [v for v, r in ratios.items() if r >= min_support]
        keep_vals = keep_vals[:max_categories_per_col]
        for val in keep_vals:
            item_col = f"{col}={val}"
            X_items.append(pd.DataFrame({item_col: (s_str == val).astype(int)}))

    if not X_items:
        raise RuntimeError("无可用特征，请检查输入与支持度阈值")
    X = pd.concat(X_items, axis=1).fillna(0)
    return X, y


def extract_one_hot_meta(X: pd.DataFrame) -> List[Tuple[str, str, str]]:
    meta: List[Tuple[str, str, str]] = []
    for c in X.columns:
        # 仅识别基础 one-hot 列（不包含已生成的组合列）
        if "=" in c and "&" not in c:
            base, val = c.split("=", 1)
            meta.append((c, base, val))
    return meta


def build_interaction_features(
    X: pd.DataFrame,
    min_joint_support: float = 0.01,
    max_pairs: int = 1000,
    top_onehot: int = 120,
    allowed_bases: List[str] | None = None,
) -> pd.DataFrame:
    """基于 one-hot 列生成两两交互特征（AND组合），并按支持度筛选。

    - 仅跨不同原始列的组合（避免同一列不同取值的无效组合）
    - 根据单列的出现率选取前 top_onehot，再在其中枚举组合
    - 组合特征名称："<colA>&<colB>"，例如 "WHEEL=20寸&BATTERY=76kwh"
    """
    meta = extract_one_hot_meta(X)
    if allowed_bases:
        meta = [m for m in meta if m[1] in allowed_bases]
    if not meta:
        return pd.DataFrame(index=X.index)

    prevalence = pd.Series({m[0]: float(X[m[0]].mean()) for m in meta}).sort_values(ascending=False)
    top_cols = list(prevalence.head(top_onehot).index)
    base_map = {m[0]: m[1] for m in meta}

    items: List[pd.DataFrame] = []
    pairs_added = 0
    n = len(top_cols)
    for i in range(n):
        ci = top_cols[i]
        for j in range(i + 1, n):
            cj = top_cols[j]
            if base_map.get(ci) == base_map.get(cj):
                continue
            joint = (X[ci].astype("uint8") & X[cj].astype("uint8"))
            support = float(joint.mean())
            if support >= min_joint_support:
                name = f"{ci}&{cj}"
                items.append(pd.DataFrame({name: joint.astype(int)}))
                pairs_added += 1
                if pairs_added >= max_pairs:
                    break
        if pairs_added >= max_pairs:
            break

    if items:
        return pd.concat(items, axis=1)
    return pd.DataFrame(index=X.index)


def linear_regression_sensitivity(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        # 兼容旧版 sklearn：RMSE = sqrt(MSE)
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }
    print(f"线性回归评估: r2={metrics['r2']:.3f}, mae={metrics['mae']:.2f}, rmse={metrics['rmse']:.2f}")
    coefs = pd.DataFrame({
        "feature": X.columns,
        "coef": lr.coef_,
        "abs_coef": np.abs(lr.coef_),
    }).sort_values("abs_coef", ascending=False)
    return coefs

def lasso_sensitivity(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lasso = LassoCV(alphas=np.logspace(-4, 1, 20), cv=5, random_state=42, n_jobs=4)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }
    print(f"Lasso 评估: r2={metrics['r2']:.3f}, mae={metrics['mae']:.2f}, rmse={metrics['rmse']:.2f}, alpha={lasso.alpha_:.4f}")
    coefs = pd.DataFrame({
        "feature": X.columns,
        "coef": lasso.coef_,
        "abs_coef": np.abs(lasso.coef_),
    }).sort_values("abs_coef", ascending=False)
    return coefs

def ridge_sensitivity(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ridge = RidgeCV(alphas=np.logspace(-4, 3, 30), cv=5)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }
    alpha_val = getattr(ridge, "alpha_", None)
    alpha_str = f", alpha={alpha_val:.4f}" if alpha_val is not None else ""
    print(f"Ridge 评估: r2={metrics['r2']:.3f}, mae={metrics['mae']:.2f}, rmse={metrics['rmse']:.2f}{alpha_str}")
    coefs = pd.DataFrame({
        "feature": X.columns,
        "coef": ridge.coef_,
        "abs_coef": np.abs(ridge.coef_),
    }).sort_values("abs_coef", ascending=False)
    return coefs


def xgb_sensitivity(X: pd.DataFrame, y: pd.Series):
    if not XGB_READY:
        print("XGBoost 未安装，跳过 XGB 评估。请先安装：pip install xgboost")
        return pd.DataFrame(columns=["feature", "importance"]), None, None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        # 兼容旧版 sklearn：RMSE = sqrt(MSE)
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }
    print(f"XGBoost 评估: r2={metrics['r2']:.3f}, mae={metrics['mae']:.2f}, rmse={metrics['rmse']:.2f}")
    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": xgb.feature_importances_,
    }).sort_values("importance", ascending=False)
    return importances, xgb, X_train, X_test, y_test

def xgb_shap_summary(model: "XGBRegressor", X_test: pd.DataFrame, top_n: int, out_path: Path) -> pd.DataFrame:
    if not SHAP_READY:
        print("SHAP 未安装，跳过 XGBoost SHAP 分析。请先安装：pip install shap")
        return pd.DataFrame(columns=["feature", "mean_abs_shap"]) 
    n_sample = min(len(X_test), 5000)
    X_sample = X_test.iloc[:n_sample]
    try:
        # 优先使用 Booster 构建 TreeExplainer，避免 base_score 类型兼容问题
        explainer = shap.TreeExplainer(model.get_booster())
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        # 回退到通用 Explainer，使用 model.predict 与样本作为 masker
        explainer = shap.Explainer(model.predict, X_sample)
        shap_values = explainer(X_sample).values
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_df = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    top_df = shap_df.head(top_n)
    top_df.to_csv(out_path, index=False)
    print(f"已保存 XGBoost SHAP Top{top_n} 特征: {out_path}")
    print("XGBoost SHAP Top 特征（mean|SHAP|排序）：")
    for i in range(len(top_df)):
        r = top_df.iloc[i]
        print(f"[SHAP-{i+1}] {r['feature']} | mean_abs_shap={r['mean_abs_shap']:.4f}")
    return shap_df


def main():
    parser = argparse.ArgumentParser(description="配置项对开票价格的敏感度分析（线性回归 + XGBoost）")
    parser.add_argument("--input", default=str(find_latest_transposed_csv(DEFAULT_DIR)), help="输入CSV路径")
    parser.add_argument("--min_support", type=float, default=0.03, help="类别值最小支持度（one-hot 过滤）")
    parser.add_argument("--max_categories_per_col", type=int, default=50, help="每列最多编码的类别数量")
    parser.add_argument("--save_top", type=int, default=30, help="导出 TopN 特征数量")
    # 交互特征相关参数
    parser.add_argument("--enable_interactions", action="store_true", help="启用配置组合（两两交互）敏感度分析")
    parser.add_argument("--min_joint_support", type=float, default=0.01, help="组合最小支持度（在样本中联合出现的占比阈值）")
    parser.add_argument("--max_interaction_pairs", type=int, default=1000, help="最大组合对数量上限")
    parser.add_argument("--interaction_top_onehot", type=int, default=120, help="参与组合枚举的 one-hot 单列数量上限（按出现率排序）")
    parser.add_argument("--interaction_bases", type=str, default="", help="限定参与组合的原始列（逗号分隔），为空表示不限定")
    args = parser.parse_args()

    input_path = Path(args.input)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"读取数据: {input_path}")
    df = load_data(input_path)
    print(f"原始维度: {df.shape}")

    # 解析列名
    product_col = resolve_column(df, [
        "Product Name", "product_name", "ProductName", "产品名称", "商品名称", "品名"
    ])
    price_col = resolve_column(df, [
        "开票价格", "invoice_price", "price", "开票单价", "价格"
    ])

    # 加入电池容量，可选是否排除（默认保留）
    try:
        df = assign_battery_capacity(df, product_col, BUSINESS_DEF_PATH)
    except Exception as e:
        print(f"电池容量映射失败（不影响继续）：{e}")

    # 构建特征：排除 Product Name 与目标列
    exclude_cols = [product_col]
    X, y = build_features(
        df,
        target_col=price_col,
        exclude_cols=exclude_cols,
        min_support=args.min_support,
        max_categories_per_col=args.max_categories_per_col,
    )
    print(f"基础特征维度: {X.shape}, 目标样本量: {y.shape}")

    # 构建交互特征（配置组合）
    X_interacts = pd.DataFrame(index=X.index)
    allowed_bases = [s.strip() for s in args.interaction_bases.split(",") if s.strip()] or None
    if args.enable_interactions:
        X_interacts = build_interaction_features(
            X,
            min_joint_support=args.min_joint_support,
            max_pairs=args.max_interaction_pairs,
            top_onehot=args.interaction_top_onehot,
            allowed_bases=allowed_bases,
        )
        if not X_interacts.empty:
            print(f"生成交互特征（配置组合）列数: {X_interacts.shape[1]}，示例: {list(X_interacts.columns[:5])}")
        else:
            print("未生成交互特征（可能受支持度或限定条件约束）")
    X_aug = pd.concat([X, X_interacts], axis=1)
    print(f"最终特征维度（含组合）: {X_aug.shape}")

    # 线性回归敏感度
    lr_coefs = linear_regression_sensitivity(X_aug, y)
    top_lr = lr_coefs.head(args.save_top)
    lr_out = OUTPUT_DIR / "config_price_sensitivity_linear.csv"
    top_lr.to_csv(lr_out, index=False)
    print(f"已保存线性回归 Top{args.save_top} 特征: {lr_out}")
    print("线性回归 Top 特征（coef 绝对值排序）：")
    for i in range(len(top_lr)):
        r = top_lr.iloc[i]
        print(f"[LR-{i+1}] {r['feature']} | coef={r['coef']:.2f}")

    # Lasso 敏感度
    lasso_coefs = lasso_sensitivity(X_aug, y)
    top_lasso = lasso_coefs.head(args.save_top)
    lasso_out = OUTPUT_DIR / "config_price_sensitivity_lasso.csv"
    top_lasso.to_csv(lasso_out, index=False)
    print(f"已保存 Lasso Top{args.save_top} 特征: {lasso_out}")
    print("Lasso Top 特征（coef 绝对值排序）：")
    for i in range(len(top_lasso)):
        r = top_lasso.iloc[i]
        print(f"[LASSO-{i+1}] {r['feature']} | coef={r['coef']:.2f}")

    # Ridge 敏感度
    ridge_coefs = ridge_sensitivity(X_aug, y)
    top_ridge = ridge_coefs.head(args.save_top)
    ridge_out = OUTPUT_DIR / "config_price_sensitivity_ridge.csv"
    top_ridge.to_csv(ridge_out, index=False)
    print(f"已保存 Ridge Top{args.save_top} 特征: {ridge_out}")
    print("Ridge Top 特征（coef 绝对值排序）：")
    for i in range(len(top_ridge)):
        r = top_ridge.iloc[i]
        print(f"[RIDGE-{i+1}] {r['feature']} | coef={r['coef']:.2f}")

    # XGBoost 敏感度
    xgb_impt, xgb_model, X_train, X_test, y_test = xgb_sensitivity(X_aug, y)
    if xgb_model is not None and not xgb_impt.empty:
        top_xgb = xgb_impt.head(args.save_top)
        xgb_out = OUTPUT_DIR / "config_price_sensitivity_xgb.csv"
        top_xgb.to_csv(xgb_out, index=False)
        print(f"已保存 XGBoost Top{args.save_top} 特征: {xgb_out}")
        print("XGBoost Top 特征（importance 排序）：")
        for i in range(len(top_xgb)):
            r = top_xgb.iloc[i]
            print(f"[XGB-{i+1}] {r['feature']} | importance={r['importance']:.4f}")
        # SHAP 分析
        shap_out = OUTPUT_DIR / "config_price_sensitivity_xgb_shap.csv"
        _ = xgb_shap_summary(xgb_model, X_test, args.save_top, shap_out)


if __name__ == "__main__":
    main()