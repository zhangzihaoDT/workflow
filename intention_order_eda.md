# 意向订单 EDA（CM2 筛选）

## 数据与筛选条件
- **数据文件**: /Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet
- **生成时间**: 2025-11-04T16:26:39
- **筛选条件**: `Intention_Payment_Time` 不为空 且 `车型分组` = `CM2`
- **主键**: `Order Number`（进行去重）

## 规模与去重
- 原始数据规模: 350294 行 × 30 列
- 筛选后行数: 59812
- 主键去重后行数: 59812
- 去重移除的重复行数: 0

## 缺失情况（pre_vehicle_model_type）
- 缺失数量: 47148
- 缺失比例: 78.83%

## 值分布 Top-10（非缺失）
- 增程: 7797
- 纯电: 4867
