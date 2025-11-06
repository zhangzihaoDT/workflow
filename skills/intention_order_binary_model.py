from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import sys
import os
from typing import List, Dict, Any

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# Optional models
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

class TargetFreqEncoder(BaseEstimator, TransformerMixin):
    """Custom encoder for target mean (with smoothing) and frequency encoding.

    - For each column in cols, creates two numeric features: `<col>_te` and `<col>_fe`.
    - Target encoding uses smoothing: (count*cat_mean + alpha*global_mean)/(count+alpha).
    - Frequency encoding is normalized count: count/total.
    - Unknown categories map to global_mean for TE and 0 for FE.
    """

    def __init__(self, cols: list[str], alpha: float = 10.0):
        self.cols = cols
        self.alpha = alpha
        self.global_mean_ = None
        self.mappings_: dict[str, dict[str, float]] = {}
        self.freqs_: dict[str, dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None):
        if y is None:
            raise ValueError("TargetFreqEncoder requires y for target encoding.")
        y = np.asarray(y).astype(float)
        self.global_mean_ = float(np.mean(y)) if y.size > 0 else 0.5
        n_total = float(len(y)) if len(y) > 0 else 1.0

        for col in self.cols:
            s = X[col].astype(str).fillna("<UNK>") if col in X.columns else pd.Series(["<UNK>"] * len(X))
            # Compute category stats
            df_cat = pd.DataFrame({col: s, "y": y})
            grp = df_cat.groupby(col)
            counts = grp.size()
            means = grp["y"].mean()
            # Smoothing target mean
            te = (counts * means + self.alpha * self.global_mean_) / (counts + self.alpha)
            # Frequency encoding
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


PARQUET_PATH = Path("/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet")
BUSINESS_DEF_PATH = Path("/Users/zihao_/Documents/github/W35_workflow/business_definition.json")
MODEL_DIR = Path("/Users/zihao_/Documents/github/W35_workflow/models")
MODEL_PATH = MODEL_DIR / "intention_binary_model.joblib"


def load_business_ranges(path: Path) -> List[tuple[datetime, datetime]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ranges: List[tuple[datetime, datetime]] = []
    for _, v in data.items():
        try:
            start = datetime.fromisoformat(v["start"]) if isinstance(v["start"], str) else pd.to_datetime(v["start"]).to_pydatetime()
            end = datetime.fromisoformat(v["end"]) if isinstance(v["end"], str) else pd.to_datetime(v["end"]).to_pydatetime()
            ranges.append((start, end))
        except Exception:
            continue
    return ranges


def within_ranges(ts: pd.Series, ranges: List[tuple[datetime, datetime]]) -> pd.Series:
    mask = pd.Series(False, index=ts.index)
    for start, end in ranges:
        mask = mask | ((ts >= start) & (ts <= end))
    return mask


def build_dataset() -> pd.DataFrame:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"文件不存在: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)

    # 解析时间列
    for col in ["Intention_Payment_Time", "Lock_Time", "first_touch_time", "first_assign_time", "Invoice_Upload_Time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 过滤：Intention_Payment_Time 不为空
    df = df[df["Intention_Payment_Time"].notna()].copy()

    # 过滤：Intention_Payment_Time 在业务定义范围内
    ranges = load_business_ranges(BUSINESS_DEF_PATH)
    if ranges:
        pay_ts = df["Intention_Payment_Time"].dt.to_pydatetime()
        mask = within_ranges(pay_ts, ranges)
        df = df[mask].copy()

    # 主键：Order Number 去重
    if "Order Number" in df.columns:
        df = df.drop_duplicates(subset=["Order Number"]).copy()

    # 标签：是否有 Lock_Time
    df["purchase"] = (df["Lock_Time"].notna()).astype(int)

    # 老用户特征：参考 ab_comparison_analysis 的思路
    # 组合键：优先使用身份证+手机号，否则仅身份证
    if "Buyer Identity No" in df.columns:
        if "Buyer Cell Phone" in df.columns:
            df["buyer_key"] = df["Buyer Identity No"].astype(str) + "_" + df["Buyer Cell Phone"].astype(str)
        else:
            df["buyer_key"] = df["Buyer Identity No"].astype(str)
    else:
        df["buyer_key"] = np.nan

    # 更精细“老用户”标记：必须存在早于当前订单的小订或开票记录
    df["is_repeat_buyer"] = 0
    df["cumulative_order_count"] = np.nan
    df["last_invoice_gap_days"] = np.nan
    if "buyer_key" in df.columns:
        # 为稳定计算，按小订时间排序后逐组处理
        df_sorted = df.sort_values(["buyer_key", "Intention_Payment_Time"]).copy()
        repeat_flags = pd.Series(0, index=df_sorted.index)
        cum_counts = pd.Series(np.nan, index=df_sorted.index)
        last_inv_gap = pd.Series(np.nan, index=df_sorted.index)

        for key, grp in df_sorted.groupby("buyer_key", sort=False):
            if pd.isna(key):
                continue
            # 组内累计订单数（整体）
            cum_counts.loc[grp.index] = float(len(grp))

            # 先计算前序最大小订时间与前序最大全部开票时间
            prev_pay_max = grp["Intention_Payment_Time"].shift(1)
            prev_pay_max = prev_pay_max.cummax()

            prev_invoice = grp["Invoice_Upload_Time"].copy()
            prev_invoice_max = prev_invoice.shift(1)
            prev_invoice_max = prev_invoice_max.cummax()

            # 是否存在严格早于当前的小订或开票
            has_prior_pay = prev_pay_max.notna() & (prev_pay_max < grp["Intention_Payment_Time"])
            has_prior_invoice = prev_invoice_max.notna() & (prev_invoice_max < grp["Intention_Payment_Time"])
            repeat_flags.loc[grp.index] = (has_prior_pay | has_prior_invoice).astype(int)

            # 最近一次开票距当前的时间差（仅考虑早于当前的小订的开票）
            valid_prev_inv = prev_invoice_max.where(prev_invoice_max < grp["Intention_Payment_Time"])  # 仅保留早于当前的小订时间的开票
            gaps = (grp["Intention_Payment_Time"] - valid_prev_inv).dt.total_seconds() / 86400.0
            last_inv_gap.loc[grp.index] = gaps

        # 映射回原 DataFrame
        df.loc[repeat_flags.index, "is_repeat_buyer"] = repeat_flags.values
        df.loc[cum_counts.index, "cumulative_order_count"] = cum_counts.values
        df.loc[last_inv_gap.index, "last_invoice_gap_days"] = last_inv_gap.values

    # 特征：城市等级、性别、年龄、时间间隔（以天为单位）
    # Assumption: 小订时间使用 Intention_Payment_Time（用户需求处列名可能误写为 intention_refund_time）
    df["interval_touch_to_pay_days"] = (df["Intention_Payment_Time"] - df["first_touch_time"]).dt.total_seconds() / 86400.0
    df["interval_assign_to_pay_days"] = (df["Intention_Payment_Time"] - df["first_assign_time"]).dt.total_seconds() / 86400.0

    # 新增特征：预售发布会时间与小订时间间隔（按车型分组映射 business_definition.json 的 time_periods.start）
    try:
        with open(BUSINESS_DEF_PATH, "r", encoding="utf-8") as f:
            bd = json.load(f)
        tp = bd.get("time_periods", {}) if isinstance(bd, dict) else {}
        start_map = {k: pd.to_datetime(v.get("start"), errors="coerce") for k, v in tp.items() if isinstance(v, dict)}
    except Exception:
        start_map = {}

    df["interval_presale_to_pay_days"] = np.nan
    if "车型分组" in df.columns and len(start_map) > 0:
        df["_presale_start"] = df["车型分组"].map(start_map)
        gap = (df["Intention_Payment_Time"] - df["_presale_start"]).dt.total_seconds() / 86400.0
        # 仅保留非负间隔，负值视为未在预售后发生，置为缺失
        df["interval_presale_to_pay_days"] = gap.where(gap >= 0, np.nan)
        df = df.drop(columns=["_presale_start"])

    return df


def train_model(df: pd.DataFrame) -> Dict[str, Any]:
    # 将 License City 改为目标/频次编码来源列，保留 order_gender 进行 One-Hot
    feature_cols_categorical = ["order_gender"]
    feature_cols_numeric = [
        "buyer_age",
        "interval_touch_to_pay_days",
        "interval_assign_to_pay_days",
        "is_repeat_buyer",
        "cumulative_order_count",
        "last_invoice_gap_days",
        "interval_presale_to_pay_days",
    ]

    # 原始编码来源列（目标/频次编码）
    # 增加 License City 到目标/频次编码来源列
    te_source_cols = ["first_main_channel_group", "Parent Region Name", "License City"]

    # 注意：需要包含原始来源列以供编码器使用
    available_cols = [c for c in te_source_cols if c in df.columns]
    # 目标/频次编码后生成的数值列名（用于进入数值预处理）
    te_feature_cols = [f"{c}_te" for c in available_cols] + [f"{c}_fe" for c in available_cols]
    X = df[feature_cols_categorical + feature_cols_numeric + available_cols].copy()
    y = df["purchase"].astype(int).values

    # 训练/测试划分（分层保证正负样本比例）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 预处理与模型
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols_numeric + te_feature_cols),
            ("cat", categorical_transformer, feature_cols_categorical),
        ]
    )

    # 基础模型：Logistic
    models: Dict[str, Any] = {}
    # Pipeline: Target/Freq encoder -> Preprocessor -> Classifier
    te_encoder = TargetFreqEncoder(cols=available_cols, alpha=10.0)
    models["logistic"] = Pipeline(steps=[
        ("te", te_encoder),
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    # 随机森林
    models["rf"] = Pipeline(steps=[
        ("te", te_encoder),
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )),
    ])

    # XGBoost（可选）
    if HAS_XGB:
        # 处理类别不平衡
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0
        models["xgb"] = Pipeline(steps=[
            ("te", te_encoder),
            ("preprocessor", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
            )),
        ])

    # LightGBM（可选）
    if HAS_LGBM:
        models["lgbm"] = Pipeline(steps=[
            ("te", te_encoder),
            ("preprocessor", preprocessor),
            ("clf", LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )),
        ])

    # 逐模型训练与评估
    results = {}
    best_name = None
    best_auc = -1.0
    best_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=4)
        results[name] = {
            "auc": float(auc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = model

    # 保存最佳模型
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": best_model,
        "best_model_name": best_name,
        "feature_cols_categorical": feature_cols_categorical,
        "feature_cols_numeric": feature_cols_numeric,
        "te_source_cols": available_cols,
        "results": results,
    }, MODEL_PATH)

    return {
        "best_model_name": best_name,
        "auc": float(best_auc),
        "all_results": results,
        "model_path": str(MODEL_PATH),
    }


def predict_new(sample: Dict[str, Any]) -> float:
    """根据新客户特征返回购买概率（1 的概率）。"""
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    cat_cols = saved["feature_cols_categorical"]
    num_cols = saved["feature_cols_numeric"]
    te_sources = saved.get("te_source_cols", [])

    # 构造单样本 DataFrame
    row = {}
    for c in cat_cols:
        row[c] = sample.get(c, None)
    for c in num_cols:
        default_val = 0.0 if c in ("is_repeat_buyer", "cumulative_order_count") else np.nan
        row[c] = sample.get(c, default_val)
    # ensure raw TE source columns are present for encoder
    for c in te_sources:
        row[c] = sample.get(c, "<UNK>")
    X_new = pd.DataFrame([row])
    proba = float(model.predict_proba(X_new)[:, 1][0])
    return proba


def main():
    df = build_dataset()
    stats = train_model(df)
    print("模型训练完成：")
    print(f"最佳模型: {stats['best_model_name']}")
    print(f"最佳AUC: {stats['auc']:.4f}")
    print("各模型评估：")
    for name, r in stats["all_results"].items():
        print(f"- {name}: AUC={r['auc']:.4f}")
    print(f"模型已保存到: {stats['model_path']}")

    # 如果以参数传入 JSON（新客户特征），则输出预测概率
    if len(sys.argv) > 1:
        try:
            payload = json.loads(sys.argv[1])
            proba = predict_new(payload)
            print("新客户购买概率:", f"{proba:.4f}")
        except Exception as e:
            print("解析输入或预测失败:", e)


if __name__ == "__main__":
    main()