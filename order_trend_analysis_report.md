# 订单趋势线监测报告

## 报告概览
- **生成时间**: 2025-09-01T16:57:12.174469
- **数据文件**: /Users/zihao_/Documents/coding/dataset/formatted/order_observation_data.parquet
- **分析类型**: 订单观察数据基本描述性分析

---

# 数据基本信息

## 数据概览
- **数据形状**: 125,308 行 × 20 列
- **内存使用**: 9.48 MB
- **数据完整性**: 73.39%
- **重复行数**: 0

## 数据类型分布
- **数值列**: 1 个
- **分类列**: 11 个
- **日期列**: 7 个

## 字段信息详情

### 全部字段列表

| 序号 | 字段名称 | 数据类型 | 非空数量 | 缺失率 | 唯一值数量 |
|------|----------|----------|----------|--------|------------|
| 1 | 日(Intention Payment Time) | datetime64[ns] | 79,688 | 36.41% | 20 |
| 2 | 车型分组 | category | 125,308 | 0.00% | 7 |
| 3 | pre_vehicle_model_type | category | 28,400 | 77.34% | 2 |
| 4 | Product Name | category | 125,308 | 0.00% | 33 |
| 5 | 日(Order Create Time) | datetime64[ns] | 125,308 | 0.00% | 29 |
| 6 | 日(Lock Time) | datetime64[ns] | 15,772 | 87.41% | 29 |
| 7 | 日(intention_refund_time) | datetime64[ns] | 11,564 | 90.77% | 17 |
| 8 | 日(Actual Refund Time) | datetime64[ns] | 13,208 | 89.46% | 28 |
| 9 | sales_loyalty_type | category | 125,220 | 0.07% | 8 |
| 10 | order_gender | category | 125,304 | 0.00% | 3 |
| 11 | buyer_age | Int16 | 110,396 | 11.90% | 138 |
| 12 | Province Name | category | 125,308 | 0.00% | 30 |
| 13 | License City | category | 125,308 | 0.00% | 332 |
| 14 | license_city_level | category | 124,172 | 0.91% | 4 |
| 15 | Parent Region Name | category | 125,308 | 0.00% | 10 |
| 16 | first_middle_channel_name | category | 125,308 | 0.00% | 20 |
| 17 | DATE([Invoice Upload Time]) | datetime64[ns] | 10,144 | 91.90% | 29 |
| 18 | DATE([first_assign_time]) | datetime64[ns] | 125,304 | 0.00% | 506 |
| 19 | 度量名称 | category | 125,308 | 0.00% | 4 |
| 20 | 度量值 | float64 | 67,726 | 45.95% | 736 |

### 数值列统计信息

| 字段名称 | 最小值 | 最大值 | 平均值 | 标准差 |
|----------|--------|--------|--------|--------|
| 度量值 | 0.00 | 615,600.00 | 143,377.57 | 144,875.79 |

### 分类列详情

车型分组, pre_vehicle_model_type, Product Name, sales_loyalty_type, order_gender, Province Name, License City, license_city_level, Parent Region Name, first_middle_channel_name, 度量名称

### 日期列详情

日(Intention Payment Time), 日(Order Create Time), 日(Lock Time), 日(intention_refund_time), 日(Actual Refund Time), DATE([Invoice Upload Time]), DATE([first_assign_time])

## 数据质量分析

### 缺失值分析

| 字段名称 | 缺失数量 | 缺失比例 |
|----------|----------|----------|
| 日(Intention Payment Time) | 45,620 | 36.41% |
| pre_vehicle_model_type | 96,908 | 77.34% |
| 日(Lock Time) | 109,536 | 87.41% |
| 日(intention_refund_time) | 113,744 | 90.77% |
| 日(Actual Refund Time) | 112,100 | 89.46% |
| sales_loyalty_type | 88 | 0.07% |
| order_gender | 4 | 0.00% |
| buyer_age | 14,912 | 11.90% |
| license_city_level | 1,136 | 0.91% |
| DATE([Invoice Upload Time]) | 115,164 | 91.90% |
| DATE([first_assign_time]) | 4 | 0.00% |
| 度量值 | 57,582 | 45.95% |

### 重复数据检查

✅ 未发现重复数据

## 数据质量评估

- **整体评级**: 较差 ❌
- **完整性得分**: 73.39%
