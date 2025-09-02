# 订单趋势线监测报告

## 报告概览
- **生成时间**: 2025-09-02T17:21:02.563626
- **数据文件**: /Users/zihao_/Documents/coding/dataset/formatted/order_observation_data_merged.parquet
- **分析类型**: 订单观察数据基本描述性分析

---

# 数据基本信息

## 数据概览
- **数据形状**: 300,156 行 × 23 列
- **内存使用**: 196.39 MB
- **数据完整性**: 74.83%
- **重复行数**: 0

## 数据类型分布
- **数值列**: 5 个
- **分类列**: 11 个
- **日期列**: 7 个

## 字段信息详情

### 全部字段列表

| 序号 | 字段名称 | 数据类型 | 非空数量 | 缺失率 | 唯一值数量 |
|------|----------|----------|----------|--------|------------|
| 1 | Order Number | object | 300,156 | 0.00% | 300,156 |
| 2 | 日(Intention Payment Time) | datetime64[ns] | 125,853 | 58.07% | 422 |
| 3 | 车型分组 | object | 300,156 | 0.00% | 7 |
| 4 | pre_vehicle_model_type | object | 19,217 | 93.60% | 2 |
| 5 | Product Name | object | 300,156 | 0.00% | 74 |
| 6 | 日(Order Create Time) | datetime64[ns] | 300,156 | 0.00% | 764 |
| 7 | 日(Lock Time) | datetime64[ns] | 132,964 | 55.70% | 763 |
| 8 | 日(intention_refund_time) | datetime64[ns] | 76,868 | 74.39% | 745 |
| 9 | 日(Actual Refund Time) | datetime64[ns] | 96,986 | 67.69% | 762 |
| 10 | sales_loyalty_type | object | 300,103 | 0.02% | 10 |
| 11 | order_gender | category | 300,155 | 0.00% | 3 |
| 12 | buyer_age | Int32 | 188,420 | 37.23% | 381 |
| 13 | Province Name | object | 300,156 | 0.00% | 30 |
| 14 | License City | object | 300,156 | 0.00% | 364 |
| 15 | license_city_level | category | 298,338 | 0.61% | 4 |
| 16 | Parent Region Name | object | 300,156 | 0.00% | 11 |
| 17 | first_middle_channel_name | object | 300,156 | 0.00% | 21 |
| 18 | DATE([Invoice Upload Time]) | datetime64[ns] | 108,362 | 63.90% | 754 |
| 19 | DATE([first_assign_time]) | datetime64[ns] | 300,155 | 0.00% | 1,303 |
| 20 | 平均值 Origin Amount | int64 | 300,156 | 0.00% | 361 |
| 21 | 平均值 开票价格 | float64 | 108,362 | 63.90% | 3,382 |
| 22 | 平均值 折扣率 | float64 | 108,362 | 63.90% | 9,920 |
| 23 | Order Number 不同计数 | int64 | 300,156 | 0.00% | 1 |

### 数值列统计信息

| 字段名称 | 最小值 | 最大值 | 平均值 | 标准差 |
|----------|--------|--------|--------|--------|
| buyer_age | -7974 | 2025 | 59.45 | 237.63 |
| 平均值 Origin Amount | 184000 | 1026599 | 276,470.21 | 36,596.84 |
| 平均值 开票价格 | -11,200.00 | 600,000.00 | 226,838.53 | 39,129.04 |
| 平均值 折扣率 | 0.00 | 1.05 | 0.13 | 0.08 |
| Order Number 不同计数 | 1 | 1 | 1.00 | 0.00 |

### 分类列详情

Order Number, 车型分组, pre_vehicle_model_type, Product Name, sales_loyalty_type, order_gender, Province Name, License City, license_city_level, Parent Region Name, first_middle_channel_name

### 日期列详情

日(Intention Payment Time), 日(Order Create Time), 日(Lock Time), 日(intention_refund_time), 日(Actual Refund Time), DATE([Invoice Upload Time]), DATE([first_assign_time])

## 数据质量分析

### 缺失值分析

| 字段名称 | 缺失数量 | 缺失比例 |
|----------|----------|----------|
| 日(Intention Payment Time) | 174,303 | 58.07% |
| pre_vehicle_model_type | 280,939 | 93.60% |
| 日(Lock Time) | 167,192 | 55.70% |
| 日(intention_refund_time) | 223,288 | 74.39% |
| 日(Actual Refund Time) | 203,170 | 67.69% |
| sales_loyalty_type | 53 | 0.02% |
| order_gender | 1 | 0.00% |
| buyer_age | 111,736 | 37.23% |
| license_city_level | 1,818 | 0.61% |
| DATE([Invoice Upload Time]) | 191,794 | 63.90% |
| DATE([first_assign_time]) | 1 | 0.00% |
| 平均值 开票价格 | 191,794 | 63.90% |
| 平均值 折扣率 | 191,794 | 63.90% |

### 重复数据检查

✅ 未发现重复数据

## 数据质量评估

- **整体评级**: 较差 ❌
- **完整性得分**: 74.83%
