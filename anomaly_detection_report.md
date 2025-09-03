# 异常检测报告

## 数据概览
- **数据文件**: /Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet
- **生成时间**: 2025-09-03T14:05:30.961051
- **文件大小**: 2.99 MB

## 数据基本信息
- **数据形状**: 126658 行 × 21 列
- **数据完整性**: 86.09%
- **重复行数**: 0

## 数据类型分布
- **数值列**: 3 个
- **分类列**: 13 个  
- **日期列**: 5 个

## 列信息详情
### 数值列
Intention Payment Time 小时, buyer_age, Order Number 不同计数

### 分类列
车型分组, pre_vehicle_model_type, pre_vehicle_model, Order Number, first_middle_channel_name, order_gender, first_big_channel_name, first_small_channel_name, Parent Region Name, License Province, license_city_level, License City, Has_Intention_Payment

### 日期列
Intention_Payment_Time, intention_refund_time, first_assign_time, Lock_Time, first_touch_time

## 异常检测结果

### 缺失值异常

| 列名 | 缺失数量 | 缺失比例 |
|------|----------|----------|
| pre_vehicle_model_type | 109019 | 86.07% |
| pre_vehicle_model | 110076 | 86.91% |
| intention_refund_time | 49456 | 39.05% |
| Lock_Time | 97201 | 76.74% |
| buyer_age | 2740 | 2.16% |
| first_touch_time | 766 | 0.60% |
| license_city_level | 681 | 0.54% |

### 重复数据检查
✅ 未发现重复数据

## 数据质量评估
- **整体评级**: 一般 ⚠️
- **完整性得分**: 86.09%
