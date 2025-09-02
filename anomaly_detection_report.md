# 异常检测报告

## 数据概览
- **数据文件**: /Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet
- **生成时间**: 2025-09-02T09:10:35.242867
- **文件大小**: 2.97 MB

## 数据基本信息
- **数据形状**: 125772 行 × 21 列
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
| pre_vehicle_model_type | 108470 | 86.24% |
| pre_vehicle_model | 109483 | 87.05% |
| intention_refund_time | 48930 | 38.90% |
| Lock_Time | 96315 | 76.58% |
| buyer_age | 2730 | 2.17% |
| first_touch_time | 762 | 0.61% |
| license_city_level | 672 | 0.53% |

### 重复数据检查
✅ 未发现重复数据

## 数据质量评估
- **整体评级**: 一般 ⚠️
- **完整性得分**: 86.09%
