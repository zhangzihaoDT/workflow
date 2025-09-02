# 订单趋势线监测报告

## 报告概览
- **生成时间**: 2025-09-02T09:55:52.875643
- **数据文件**: /Users/zihao_/Documents/coding/dataset/formatted/order_observation_data.parquet
- **分析类型**: 订单观察数据基本描述性分析

---

# 数据基本信息

## 数据概览
- **数据形状**: 38,407 行 × 23 列
- **内存使用**: 20.12 MB
- **数据完整性**: 91.53%
- **重复行数**: 0

## 数据类型分布
- **数值列**: 4 个
- **分类列**: 18 个
- **日期列**: 0 个

## 字段信息详情

### 全部字段列表

| 序号 | 字段名称 | 数据类型 | 非空数量 | 缺失率 | 唯一值数量 |
|------|----------|----------|----------|--------|------------|
| 1 | Order Number | object | 38,407 | 0.00% | 38,407 |
| 2 | Intention_Payment_Time | object | 38,407 | 0.00% | 23 |
| 3 | 车型分组 | category | 38,407 | 0.00% | 7 |
| 4 | pre_vehicle_model_type | category | 38,407 | 0.00% | 3 |
| 5 | Product Name | category | 38,407 | 0.00% | 34 |
| 6 | Order_Create_Time | object | 38,407 | 0.00% | 31 |
| 7 | Lock_Time | object | 38,407 | 0.00% | 33 |
| 8 | intention_refund_time | object | 38,407 | 0.00% | 21 |
| 9 | Actual_Refund_Time | object | 38,407 | 0.00% | 32 |
| 10 | sales_loyalty_type | category | 38,407 | 0.00% | 9 |
| 11 | order_gender | category | 38,407 | 0.00% | 3 |
| 12 | buyer_age | Int16 | 34,253 | 10.82% | 148 |
| 13 | Province Name | category | 38,407 | 0.00% | 30 |
| 14 | License City | category | 38,407 | 0.00% | 335 |
| 15 | license_city_level | category | 38,407 | 0.00% | 5 |
| 16 | Parent Region Name | category | 38,407 | 0.00% | 10 |
| 17 | first_middle_channel_name | category | 38,407 | 0.00% | 20 |
| 18 | Invoice_Upload_Time | object | 38,407 | 0.00% | 32 |
| 19 | first_assign_time | object | 38,407 | 0.00% | 530 |
| 20 | Order Number 不同计数 | float64 | 38,407 | 0.00% | 1 |
| 21 | 折扣率 | float64 | 3,058 | 92.04% | 427 |
| 22 | 原始价格 | float64 | 38,407 | 0.00% | 99 |
| 23 | 开票价格 | float64 | 3,058 | 92.04% | 297 |

### 数值列统计信息

| 字段名称 | 最小值 | 最大值 | 平均值 | 标准差 |
|----------|--------|--------|--------|--------|
| Order Number 不同计数 | 1.00 | 1.00 | 1.00 | 0.00 |
| 折扣率 | 0.00 | 0.98 | 0.12 | 0.09 |
| 原始价格 | 210,700.00 | 615,600.00 | 292,936.53 | 20,486.92 |
| 开票价格 | 5,000.00 | 469,600.00 | 218,035.99 | 29,361.50 |

### 分类列详情

Order Number, Intention_Payment_Time, 车型分组, pre_vehicle_model_type, Product Name, Order_Create_Time, Lock_Time, intention_refund_time, Actual_Refund_Time, sales_loyalty_type, order_gender, Province Name, License City, license_city_level, Parent Region Name, first_middle_channel_name, Invoice_Upload_Time, first_assign_time

## 数据质量分析

### 缺失值分析

| 字段名称 | 缺失数量 | 缺失比例 |
|----------|----------|----------|
| buyer_age | 4,154 | 10.82% |
| 折扣率 | 35,349 | 92.04% |
| 开票价格 | 35,349 | 92.04% |

### 重复数据检查

✅ 未发现重复数据

## 数据质量评估

- **整体评级**: 良好 ⚠️
- **完整性得分**: 91.53%
