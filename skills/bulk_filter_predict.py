from __future__ import annotations

from pathlib import Path
import sys
import argparse
import pandas as pd
import numpy as np
import joblib

# 引入构建数据集与模型路径
try:
    from skills.intention_order_binary_model import build_dataset, MODEL_PATH
except Exception:
    from intention_order_binary_model import build_dataset, MODEL_PATH


# 为反序列化提供自定义编码器类（训练时在__main__下定义）
class TargetFreqEncoder:
    def __init__(self, cols: list[str], alpha: float = 10.0):
        self.cols = cols
        self.alpha = alpha
        self.global_mean_ = None
        self.mappings_ = {}
        self.freqs_ = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None):
        # 该类在训练脚本中已完成拟合并被序列化；此处无需重新计算
        return self

    def transform(self, X: pd.DataFrame):
        # 使用已保存的 self.mappings_ 和 self.freqs_ 进行映射，生成 <col>_te 与 <col>_fe
        X = X.copy()
        for col in getattr(self, "cols", []):
            s = X[col].astype(str).fillna("<UNK>") if col in X.columns else pd.Series(["<UNK>"] * len(X))
            te_map = getattr(self, "mappings_", {}).get(col, {})
            fe_map = getattr(self, "freqs_", {}).get(col, {})
            global_mean = getattr(self, "global_mean_", 0.5)
            X[f"{col}_te"] = s.map(te_map).fillna(global_mean)
            X[f"{col}_fe"] = s.map(fe_map).fillna(0.0)
        return X


DEFAULT_OUTPUT_DIR = Path("/Users/zihao_/Documents/github/W35_workflow/models")


def main():
    parser = argparse.ArgumentParser(description="按日期与车型分组筛选并批量预测")
    parser.add_argument("--date", required=False, default="2025-11-04", help="筛选的 Intention_Payment_Time 日期，如 2025-11-04")
    parser.add_argument("--group", required=False, default="LS9", help="筛选的 车型分组，如 CM2/LS9 等")
    parser.add_argument("--output", required=False, help="输出CSV路径；默认保存到 models/filtered_{group}_{date}.csv")
    args = parser.parse_args()

    # 解析筛选条件
    target_date = pd.to_datetime(args.date).date()
    target_group = str(args.group)

    # 构建包含特征工程的数据集
    df = build_dataset()
    if "Intention_Payment_Time" not in df.columns:
        raise RuntimeError("数据集中缺少 Intention_Payment_Time 列")
    if "车型分组" not in df.columns:
        raise RuntimeError("数据集中缺少 车型分组 列")

    # 筛选
    date_mask = df["Intention_Payment_Time"].dt.date == target_date
    group_mask = df["车型分组"].astype(str) == target_group
    df_sel = df[date_mask & group_mask].copy()

    # 导出 CSV（原始筛选结果，便于审阅与复用）
    if args.output:
        output_path = Path(args.output)
    else:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe_date = pd.to_datetime(target_date).strftime("%Y-%m-%d")
        output_path = DEFAULT_OUTPUT_DIR / f"filtered_{target_group}_{safe_date}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_sel.to_csv(output_path, index=False)
    print(f"筛选后数据已保存到: {output_path}")

    if df_sel.empty:
        print("筛选结果为空，无法进行预测。")
        return

    # 加载模型与特征设置
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    cat_cols = saved.get("feature_cols_categorical", [])
    num_cols = saved.get("feature_cols_numeric", [])
    te_sources = saved.get("te_source_cols", [])

    # 组装预测所需列
    missing = [c for c in (cat_cols + num_cols + te_sources) if c not in df_sel.columns]
    if missing:
        print("警告：以下预测所需列缺失，将以缺失处理后插补：", ", ".join(missing))
        for c in missing:
            # 数值列填充为 NaN（由管道插补），类别/TE来源填充为占位
            if c in num_cols:
                df_sel[c] = np.nan
            else:
                df_sel[c] = "<UNK>"

    X_new = df_sel[cat_cols + num_cols + te_sources].copy()
    probs = model.predict_proba(X_new)[:, 1]
    predicted_ratio = float(np.mean(probs))
    print(f"预测购买比例: {predicted_ratio:.2%}")


if __name__ == "__main__":
    main()