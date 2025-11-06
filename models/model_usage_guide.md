# 意向订单二分类模型使用手册

本手册说明如何训练、评估与使用意向订单二分类模型，并提供特征解释与学习曲线用于了解模型行为与样本量敏感性。

## 模型概览

- 任务：预测小订是否会锁单（`purchase` 标签：`Lock_Time` 是否非空）。
- 主要特征：
  - 类别：`order_gender`（One-Hot）
  - 目标/频次编码来源列：`License City`、`first_main_channel_group`、`Parent Region Name`（生成 `<col>_te` 与 `<col>_fe` 数值特征）
  - 数值：`buyer_age`、`interval_touch_to_pay_days`、`interval_assign_to_pay_days`、`is_repeat_buyer`、`cumulative_order_count`、`last_invoice_gap_days`、`interval_presale_to_pay_days`
- 目标/频次编码来源列：`License City`、`first_main_channel_group`、`Parent Region Name`（生成 `<col>_te` 与 `<col>_fe` 数值特征）
- 数据过滤：`Intention_Payment_Time` 落在 `business_definition.json` 业务定义范围内；订单按 `Order Number` 去重。
- 最佳模型：XGBoost（AUC ≈ 0.7229）。

## 环境准备

- Python ≥ 3.10
- 依赖：`scikit-learn`、`pandas`、`numpy`、`xgboost`、`lightgbm`、`matplotlib`、`joblib`
- 数据文件：`/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet`
- 业务定义：`/Users/zihao_/Documents/github/W35_workflow/business_definition.json`

## 训练与评估

1. 激活环境：
   ```bash
   source .venv/bin/activate
   ```
2. 运行训练脚本（包含多模型对比与保存最佳模型）：
   ```bash
   python skills/intention_order_binary_model.py
   ```
3. 输出内容：
   - 各模型 AUC 对比（Logistic、RF、XGB、LGBM）
   - 最佳模型保存：`models/intention_binary_model.joblib`

## 特征解释与学习曲线

1. 生成特征重要性与学习曲线：
   ```bash
   python skills/model_feature_and_learning_curve.py
   ```
2. 输出文件：
   - `models/feature_importance.csv`：特征重要性表（树模型为 `feature_importances_`，Logistic 为 |coef|）
   - `models/feature_importance_top20.png`：Top20 特征重要性条形图
   - `models/learning_curve_auc.png`：AUC 学习曲线（训练/验证）
   - `models/learning_curve_auc.csv`：学习曲线原始数据（含误差 `1-AUC`）

## 推理使用

### 命令行预测（单样本）

```bash
python skills/intention_order_binary_model.py '{
  "License City":"上海",
  "order_gender":"F",
  "buyer_age":28,
  "interval_touch_to_pay_days":3.5,
  "interval_assign_to_pay_days":1.2,
  "is_repeat_buyer":0,
  "cumulative_order_count":1,
  "last_invoice_gap_days":null,
  "first_main_channel_group":"线上直销",
  "Parent Region Name":"上海",
  "车型分组":"CM2",
  "Intention_Payment_Time":"2025-08-20"
}'
```

- 说明：
  - 目标/频次编码需要原始来源列：`License City`、`first_main_channel_group`、`Parent Region Name`。
  - `interval_presale_to_pay_days` 将在管道中基于 `车型分组` 与业务定义自动计算（训练阶段已计算；推理阶段未强制，但建议提供以提高一致性）。
  - 未提供的数值特征默认：`is_repeat_buyer`、`cumulative_order_count` 为 0；其他为缺失，由插补处理。

### 程序化使用

```python
import joblib, pandas as pd
from pathlib import Path

MODEL_PATH = Path("/Users/zihao_/Documents/github/W35_workflow/models/intention_binary_model.joblib")
saved = joblib.load(MODEL_PATH)
model = saved["model"]
cat_cols = saved["feature_cols_categorical"]
num_cols = saved["feature_cols_numeric"]
te_sources = saved.get("te_source_cols", [])

row = {"license_city_level":"T2","order_gender":"F","buyer_age":28,
       "interval_touch_to_pay_days":3.5,"interval_assign_to_pay_days":1.2,
       "is_repeat_buyer":0,"cumulative_order_count":1,"last_invoice_gap_days":None,
       "first_main_channel_group":"线上直销","Parent Region Name":"上海","License City":"上海"}
X = pd.DataFrame([row], columns=cat_cols+num_cols+te_sources)
proba = float(model.predict_proba(X)[:,1][0])
print(proba)
```

## 常见问题

- 字体警告：生成中文图时可能提示缺少字体，属 Matplotlib 默认字体提示，不影响图文件生成。
- 目标编码泄漏：当前为整集计算（含平滑）；如需更严谨验证可引入折内计算的 KFold 目标编码。
- AUC 波动：建议采用时间感知或按预售批次分层的验证方案评估稳定性。

## 进一步优化建议

- 超参数搜索：网格/贝叶斯优化 XGBoost/LightGBM 超参数与编码平滑系数 `alpha`。
- 特征工程：渠道/区域更细粒度编码、交互项、更多历史行为（近期小订频率、退订历史等）。
- 评估扩展：PR-AUC、校准曲线、分组稳定性（按门店/渠道/地区分组）。
