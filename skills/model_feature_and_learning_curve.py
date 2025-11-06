from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.metrics import make_scorer, roc_auc_score

# 复用训练脚本中的数据构建逻辑
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
        if y is None:
            raise ValueError("TargetFreqEncoder requires y for target encoding.")
        y = np.asarray(y).astype(float)
        self.global_mean_ = float(np.mean(y)) if y.size > 0 else 0.5
        n_total = float(len(y)) if len(y) > 0 else 1.0
        for col in self.cols:
            s = X[col].astype(str).fillna("<UNK>") if col in X.columns else pd.Series(["<UNK>"] * len(X))
            df_cat = pd.DataFrame({col: s, "y": y})
            grp = df_cat.groupby(col)
            counts = grp.size()
            means = grp["y"].mean()
            te = (counts * means + self.alpha * self.global_mean_) / (counts + self.alpha)
            fe = counts / n_total
            self.mappings_[col] = te.to_dict()
            self.freqs_[col] = fe.to_dict()
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.cols:
            s = X[col].astype(str).fillna("<UNK>") if col in X.columns else pd.Series(["<UNK>"] * len(X))
            te_map = self.mappings_.get(col, {})
            fe_map = self.freqs_.get(col, {})
            X[f"{col}_te"] = s.map(te_map).fillna(self.global_mean_)
            X[f"{col}_fe"] = s.map(fe_map).fillna(0.0)
        return X


OUTPUT_DIR = Path("/Users/zihao_/Documents/github/W35_workflow/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_saved() -> Dict[str, Any]:
    saved = joblib.load(MODEL_PATH)
    return saved


def extract_feature_names(pipeline) -> List[str]:
    # pipeline: Pipeline(te -> preprocessor -> clf)
    pre = pipeline.named_steps.get("preprocessor")
    try:
        feature_names = list(pre.get_feature_names_out())
    except Exception:
        # 兜底：无法获取就返回索引形式
        n_features = pipeline.named_steps["clf"].n_features_in_ if hasattr(pipeline.named_steps["clf"], "n_features_in_") else 0
        feature_names = [f"f{i}" for i in range(n_features)]
    return feature_names


def compute_feature_importance(pipeline) -> pd.DataFrame:
    clf = pipeline.named_steps.get("clf")
    names = extract_feature_names(pipeline)

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        df_imp = pd.DataFrame({"feature": names, "importance": importances})
        df_imp = df_imp.sort_values("importance", ascending=False)
        return df_imp
    elif hasattr(clf, "coef_"):
        coef = clf.coef_.ravel()
        df_imp = pd.DataFrame({"feature": names, "importance": np.abs(coef), "coef": coef})
        df_imp = df_imp.sort_values("importance", ascending=False)
        return df_imp
    else:
        # 无法直接提供重要性
        return pd.DataFrame({"feature": names, "importance": np.nan})


def plot_feature_importance(df_imp: pd.DataFrame, top_n: int = 20, out_path: Path | None = None):
    df_top = df_imp.head(top_n)
    plt.figure(figsize=(10, max(4, 0.4 * len(df_top))))
    plt.barh(df_top["feature"][::-1], df_top["importance"][::-1], color="#005783")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importance")
    plt.tight_layout()
    if out_path is None:
        out_path = OUTPUT_DIR / "feature_importance_top20.png"
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_learning_curve(pipeline, X: pd.DataFrame, y: np.ndarray, out_path: Path | None = None):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 8)
    scorer = "roc_auc"

    sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y,
        cv=cv,
        scoring=scorer,
        train_sizes=train_sizes,
        n_jobs=-1,
        shuffle=True,
        random_state=42,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    # 绘制AUC学习曲线
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, train_mean, "-o", color="#27AD00", label="Train AUC")
    plt.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#27AD00")
    plt.plot(sizes, val_mean, "-o", color="#005783", label="Validation AUC")
    plt.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#005783")
    plt.xlabel("Training samples")
    plt.ylabel("ROC AUC")
    plt.title("Learning Curve (ROC AUC)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    if out_path is None:
        out_path = OUTPUT_DIR / "learning_curve_auc.png"
    plt.savefig(out_path, dpi=160)
    plt.close()

    # 保存原始数据
    df_curve = pd.DataFrame({
        "train_size": sizes,
        "train_auc_mean": train_mean,
        "train_auc_std": train_std,
        "val_auc_mean": val_mean,
        "val_auc_std": val_std,
        "train_error": 1.0 - train_mean,
        "val_error": 1.0 - val_mean,
    })
    df_curve.to_csv(OUTPUT_DIR / "learning_curve_auc.csv", index=False)


def main():
    saved = load_saved()
    pipeline = saved["model"]
    best_model_name = saved.get("best_model_name", "unknown")
    cat_cols = saved.get("feature_cols_categorical", [])
    num_cols = saved.get("feature_cols_numeric", [])
    te_sources = saved.get("te_source_cols", [])

    # 特征重要性
    df_imp = compute_feature_importance(pipeline)
    df_imp.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    plot_feature_importance(df_imp, top_n=20)
    print("特征重要性已保存：", OUTPUT_DIR / "feature_importance.csv")
    print("特征重要性Top20图已保存：", OUTPUT_DIR / "feature_importance_top20.png")

    # 学习曲线：使用与训练一致的特征构造
    df = build_dataset()
    available_cols = [c for c in te_sources if c in df.columns]
    X = df[cat_cols + num_cols + available_cols].copy()
    y = df["purchase"].astype(int).values

    plot_learning_curve(pipeline, X, y)
    print("学习曲线图已保存：", OUTPUT_DIR / "learning_curve_auc.png")
    print("学习曲线数据已保存：", OUTPUT_DIR / "learning_curve_auc.csv")
    print("最佳模型:", best_model_name)


if __name__ == "__main__":
    main()