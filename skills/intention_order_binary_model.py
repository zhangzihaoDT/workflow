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

    # 定金金额特征：根据车型分组映射（用户提供：LS9=5000，CM2=2000）
    # 为提高鲁棒性，允许包含型谱文本（如包含“LS9”、“CM2”均可识别）
    df["deposit_amount"] = np.nan
    if "车型分组" in df.columns:
        def _map_deposit(x: Any) -> float | np.nan:
            s = str(x).upper() if pd.notna(x) else ""
            if "LS9" in s:
                return 5000.0
            if "CM2" in s:
                return 2000.0
            return np.nan
        df["deposit_amount"] = df["车型分组"].apply(_map_deposit)

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
        "deposit_amount",
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


def _get_feature_names_from_pipeline(pipeline: Pipeline) -> List[str]:
    names: List[str] = []
    try:
        pre = pipeline.named_steps.get("preprocessor")
        if hasattr(pre, "get_feature_names_out"):
            raw = list(pre.get_feature_names_out())
            # 清理前缀，如 "num__feature" 或 "cat__feature"
            names = [str(n).split("__")[-1] for n in raw]
    except Exception:
        names = []
    return names


def analyze_deposit_influence(df: pd.DataFrame) -> Dict[str, Any]:
    """对比分析定金金额对锁单率的主效应。

    输出三种设置的AUC：
    - base: 不含定金金额的原特征
    - deposit_only: 仅使用定金金额
    - full: 在原特征上加入定金金额

    并从 full 的逻辑回归中提取系数排名，查看 deposit_amount 的相对权重。
    结果会保存到 models/deposit_effect_report.md。
    """

    # 准备标签
    y = df["purchase"].astype(int).values

    # 定义基础列（不含定金）
    base_feature_cols_categorical = ["order_gender"] if "order_gender" in df.columns else []
    base_feature_cols_numeric = [
        c for c in [
            "buyer_age",
            "interval_touch_to_pay_days",
            "interval_assign_to_pay_days",
            "is_repeat_buyer",
            "cumulative_order_count",
            "last_invoice_gap_days",
            "interval_presale_to_pay_days",
        ] if c in df.columns
    ]

    te_source_cols = [c for c in ["first_main_channel_group", "Parent Region Name", "License City"] if c in df.columns]
    te_feature_cols = [f"{c}_te" for c in te_source_cols] + [f"{c}_fe" for c in te_source_cols]

    # 组装不同设置的 X
    def _make_X(cols_cat: List[str], cols_num: List[str], te_cols: List[str]) -> pd.DataFrame:
        return df[cols_cat + cols_num + te_cols].copy()

    # 训练/测试划分
    X_base = _make_X(base_feature_cols_categorical, base_feature_cols_numeric, te_source_cols)
    X_full = X_base.copy()
    if "deposit_amount" in df.columns:
        X_full["deposit_amount"] = df["deposit_amount"].values
    X_dep_only = df[["deposit_amount"]].copy() if "deposit_amount" in df.columns else pd.DataFrame({"deposit_amount": [np.nan] * len(df)})

    Xb_tr, Xb_te, y_tr, y_te = train_test_split(X_base, y, test_size=0.2, random_state=42, stratify=y)
    Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(X_full, y, test_size=0.2, random_state=42, stratify=y)
    Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(X_dep_only, y, test_size=0.2, random_state=42, stratify=y)

    # 通用预处理器
    num_base = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_base = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre_base = ColumnTransformer(
        transformers=[
            ("num", num_base, base_feature_cols_numeric + [f for f in te_feature_cols if f in (te_feature_cols)]),
            ("cat", cat_base, base_feature_cols_categorical),
        ]
    )
    te_enc = TargetFreqEncoder(cols=te_source_cols, alpha=10.0)

    # base: 不含定金
    pipe_base_log = Pipeline(steps=[
        ("te", te_enc),
        ("preprocessor", pre_base),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    # full: 加入定金（在数值列里）
    base_plus_dep_num = base_feature_cols_numeric + ["deposit_amount"]
    pre_full = ColumnTransformer(
        transformers=[
            ("num", num_base, base_plus_dep_num + [f for f in te_feature_cols]),
            ("cat", cat_base, base_feature_cols_categorical),
        ]
    )
    pipe_full_log = Pipeline(steps=[
        ("te", te_enc),
        ("preprocessor", pre_full),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    # deposit_only
    pre_dep_only = ColumnTransformer(transformers=[("num", num_base, ["deposit_amount"])])
    pipe_dep_only_log = Pipeline(steps=[
        ("preprocessor", pre_dep_only),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    # 训练与评估
    pipe_base_log.fit(Xb_tr, y_tr)
    pipe_full_log.fit(Xf_tr, yf_tr)
    pipe_dep_only_log.fit(Xd_tr, yd_tr)

    auc_base = float(roc_auc_score(y_te, pipe_base_log.predict_proba(Xb_te)[:, 1]))
    auc_full = float(roc_auc_score(yf_te, pipe_full_log.predict_proba(Xf_te)[:, 1]))
    auc_dep_only = float(roc_auc_score(yd_te, pipe_dep_only_log.predict_proba(Xd_te)[:, 1]))

    # 提取 full 的特征权重
    feat_names_full = _get_feature_names_from_pipeline(pipe_full_log)
    coefs_full = []
    try:
        coef = pipe_full_log.named_steps["clf"].coef_.ravel()
        coefs_full = list(zip(feat_names_full, coef))
        coefs_full_sorted = sorted(coefs_full, key=lambda x: abs(x[1]), reverse=True)
    except Exception:
        coefs_full_sorted = []

    dep_coef = None
    for n, w in coefs_full_sorted:
        if n == "deposit_amount":
            dep_coef = float(w)
            break

    # 生成报告
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MODEL_DIR / "deposit_effect_report.md"
    lines = []
    lines.append("# 定金金额对锁单率影响分析\n")
    lines.append(f"- base AUC（不含定金）: {auc_base:.4f}\n")
    lines.append(f"- deposit-only AUC（仅定金）: {auc_dep_only:.4f}\n")
    lines.append(f"- full AUC（加入定金）: {auc_full:.4f}\n")
    if dep_coef is not None:
        lines.append(f"- full 逻辑回归中 `deposit_amount` 系数: {dep_coef:.6f}\n")
    if coefs_full_sorted:
        topk = coefs_full_sorted[:15]
        lines.append("\n## full 模型特征权重Top15（按绝对值）\n")
        for n, w in topk:
            lines.append(f"- {n}: {w:.6f}\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return {
        "auc_base": auc_base,
        "auc_deposit_only": auc_dep_only,
        "auc_full": auc_full,
        "deposit_coef_full": dep_coef,
        "report_path": str(report_path),
    }


# ===== PSM（倾向得分匹配）针对定金金额（LS9=5000 vs CM2=2000）=====
def build_dataset_ls9_cm2() -> pd.DataFrame:
    """在通用构建的基础上，仅保留车型分组包含 LS9 或 CM2 的样本。其他筛选保持不变。"""
    df = build_dataset()
    if "车型分组" in df.columns:
        s = df["车型分组"].astype(str).str.upper()
        mask = s.str.contains("LS9") | s.str.contains("CM2")
        df = df[mask].copy()
    # 确保存在 deposit_amount 且不缺失
    if "deposit_amount" in df.columns:
        df = df[df["deposit_amount"].notna()].copy()
    # 定义处理变量：T=1(LS9/5000)，T=0(CM2/2000)
    df["treat_high_deposit"] = (df["deposit_amount"] >= 5000).astype(int)
    return df


def _compute_smd(X_t: np.ndarray, X_c: np.ndarray) -> np.ndarray:
    """标准化均值差 SMD，适用于数值或One-Hot特征矩阵。"""
    mt = np.nanmean(X_t, axis=0)
    mc = np.nanmean(X_c, axis=0)
    st = np.nanvar(X_t, axis=0, ddof=1)
    sc = np.nanvar(X_c, axis=0, ddof=1)
    pooled = np.sqrt((st + sc) / 2.0)
    # 避免除零
    pooled = np.where(pooled == 0, 1e-8, pooled)
    smd = (mt - mc) / pooled
    return np.abs(smd)


def psm_deposit_analysis() -> Dict[str, Any]:
    """执行基于Logistic的倾向得分匹配（1:1，caliper=0.05），并输出SMD与ATT报告。"""
    df = build_dataset_ls9_cm2()

    # 处理与协变量选择：不包含治疗相关的变量（deposit_amount、车型分组）
    y = df["purchase"].astype(int).values
    treat = df["treat_high_deposit"].astype(int).values

    base_feature_cols_categorical = [c for c in ["order_gender"] if c in df.columns]
    base_feature_cols_numeric = [
        c for c in [
            "buyer_age",
            "interval_touch_to_pay_days",
            "interval_assign_to_pay_days",
            "is_repeat_buyer",
            "cumulative_order_count",
            "last_invoice_gap_days",
            "interval_presale_to_pay_days",
        ] if c in df.columns
    ]
    te_source_cols = [c for c in ["first_main_channel_group", "Parent Region Name", "License City"] if c in df.columns]

    X_cov = df[base_feature_cols_categorical + base_feature_cols_numeric + te_source_cols].copy()

    # 预处理：与主流程一致（目标/频次编码 + 数值标准化 + 类别One-Hot）
    num_base = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_base = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    te_enc = TargetFreqEncoder(cols=te_source_cols, alpha=10.0)

    # 构造预处理用于Logit和SMD提取特征矩阵
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_base, base_feature_cols_numeric + [f"{c}_te" for c in te_source_cols] + [f"{c}_fe" for c in te_source_cols]),
            ("cat", cat_base, base_feature_cols_categorical),
        ]
    )

    # Step1: Logistic 倾向得分（稳健可解释）
    ps_model = Pipeline(steps=[
        ("te", te_enc),
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    ps_model.fit(X_cov, treat)
    propensity = ps_model.predict_proba(X_cov)[:, 1]

    # 提取协变量矩阵用于SMD
    X_trans = ps_model.named_steps["preprocessor"].transform(
        ps_model.named_steps["te"].transform(X_cov)
    )
    # 将稀疏矩阵转为数组
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    # Step2: 最近邻1:1匹配（caliper=0.05），不放回
    caliper = 0.05
    idx_t = np.where(treat == 1)[0]
    idx_c = np.where(treat == 0)[0]
    ps_t = propensity[idx_t]
    ps_c = propensity[idx_c]

    # 按倾向得分排序，双指针贪心匹配
    order_t = np.argsort(ps_t)
    order_c = np.argsort(ps_c)
    ps_t_sorted = ps_t[order_t]
    ps_c_sorted = ps_c[order_c]
    idx_t_sorted = idx_t[order_t]
    idx_c_sorted = idx_c[order_c]

    matched_pairs: list[tuple[int, int]] = []
    j = 0
    used_c = set()
    for i in range(len(ps_t_sorted)):
        pt = ps_t_sorted[i]
        # 推进控制组指针到接近pt的位置
        while j < len(ps_c_sorted) - 1 and ps_c_sorted[j] < pt:
            j += 1
        # 候选控制：j与j-1中更近者
        candidates = []
        if j < len(ps_c_sorted):
            candidates.append(j)
        if j - 1 >= 0:
            candidates.append(j - 1)
        # 选择距离最小且在caliper内且未使用的控制
        best_c = None
        best_d = None
        for cj in candidates:
            if cj < 0 or cj >= len(ps_c_sorted):
                continue
            if cj in used_c:
                continue
            d = abs(ps_c_sorted[cj] - pt)
            if d <= caliper and (best_d is None or d < best_d):
                best_d = d
                best_c = cj
        if best_c is not None:
            used_c.add(best_c)
            matched_pairs.append((idx_t_sorted[i], idx_c_sorted[best_c]))

    # Step3: 计算匹配前后SMD，检验阈值
    X_t = X_trans[idx_t, :]
    X_c = X_trans[idx_c, :]
    smd_before = _compute_smd(X_t, X_c)

    if matched_pairs:
        mt_idx = np.array([p[0] for p in matched_pairs])
        mc_idx = np.array([p[1] for p in matched_pairs])
        X_t_matched = X_trans[mt_idx, :]
        X_c_matched = X_trans[mc_idx, :]
        smd_after = _compute_smd(X_t_matched, X_c_matched)
    else:
        smd_after = np.array([])

    smd_before_max = float(np.nanmax(smd_before)) if smd_before.size > 0 else np.nan
    smd_after_max = float(np.nanmax(smd_after)) if smd_after.size > 0 else np.nan
    smd_after_prop_under_01 = float(np.mean(smd_after < 0.1)) if smd_after.size > 0 else 0.0

    # Step4: 估计ATT（配对差的平均），并给出95%CI
    att = np.nan
    ci_low = np.nan
    ci_high = np.nan
    n_pairs = len(matched_pairs)
    if n_pairs > 0:
        y_t = y[mt_idx]
        y_c = y[mc_idx]
        diffs = y_t - y_c
        att = float(np.mean(diffs))
        sd = float(np.std(diffs, ddof=1)) if n_pairs > 1 else 0.0
        se = sd / np.sqrt(n_pairs) if n_pairs > 1 else 0.0
        ci_low = att - 1.96 * se
        ci_high = att + 1.96 * se

    # 输出报告
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report_path = MODEL_DIR / "psm_deposit_report.md"
    lines = []
    lines.append("# PSM：定金金额（5000 vs 2000）对锁单率的因果影响\n")
    lines.append("\n## Step1 倾向得分(Logistic)\n")
    lines.append(f"- 样本量：T(高定金)={len(idx_t)}，C(低定金)={len(idx_c)}\n")
    # 评估倾向模型的可分性（AUC of treat prediction）
    try:
        auc_ps = roc_auc_score(treat, propensity)
        lines.append(f"- 倾向模型AUC（T预测）: {auc_ps:.4f}\n")
    except Exception:
        lines.append("- 倾向模型AUC（T预测）: 计算失败\n")

    lines.append("\n## Step2 最近邻匹配（1:1，caliper=0.05）\n")
    lines.append(f"- 匹配成功对数：{n_pairs}\n")
    lines.append(f"- caliper: 0.05\n")

    lines.append("\n## Step3 平衡检验（SMD）\n")
    lines.append(f"- 匹配前最大SMD: {smd_before_max:.4f}\n")
    lines.append(f"- 匹配后最大SMD: {smd_after_max:.4f}\n")
    lines.append(f"- 匹配后SMD<0.1的比例: {smd_after_prop_under_01:.3f}\n")

    lines.append("\n## Step4 ATT（平均处理效应-已处理）\n")
    lines.append(f"- ATT（锁单率差）：{att:.4f}\n")
    lines.append(f"- 95%CI: [{ci_low:.4f}, {ci_high:.4f}]\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return {
        "n_treated": int(len(idx_t)),
        "n_control": int(len(idx_c)),
        "n_pairs": int(n_pairs),
        "smd_before_max": smd_before_max,
        "smd_after_max": smd_after_max,
        "smd_after_prop_under_01": smd_after_prop_under_01,
        "att": att,
        "att_ci": (ci_low, ci_high),
        "report_path": str(report_path),
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
    # 先输出定金影响的对比分析
    try:
        dep_stats = analyze_deposit_influence(df)
        print("定金对锁单率的影响（AUC对比）：")
        print(f"- base（不含定金）: {dep_stats['auc_base']:.4f}")
        print(f"- deposit-only（仅定金）: {dep_stats['auc_deposit_only']:.4f}")
        print(f"- full（加入定金）: {dep_stats['auc_full']:.4f}")
        if dep_stats.get("deposit_coef_full") is not None:
            print(f"full 模型中定金系数: {dep_stats['deposit_coef_full']:.6f}")
        print(f"报告已保存到: {dep_stats['report_path']}")
    except Exception as e:
        print("定金影响分析失败:", e)

    # 执行PSM因果分析（LS9/CM2专用）
    try:
        psm_stats = psm_deposit_analysis()
        print("\nPSM（倾向得分匹配）分析：")
        print(f"- 样本量：T={psm_stats['n_treated']}, C={psm_stats['n_control']}, 配对={psm_stats['n_pairs']}")
        print(f"- SMD前最大={psm_stats['smd_before_max']:.4f}，后最大={psm_stats['smd_after_max']:.4f}，后SMD<0.1比例={psm_stats['smd_after_prop_under_01']:.3f}")
        ci = psm_stats['att_ci']
        print(f"- ATT={psm_stats['att']:.4f}，95%CI=[{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"PSM报告已保存到: {psm_stats['report_path']}")
    except Exception as e:
        print("PSM分析失败:", e)

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