# 异常检测报告

## 数据概览
- **数据文件**: /Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet
- **生成时间**: 2025-09-10T15:42:53.501004
- **文件大小**: 7.03 MB

## 数据基本信息
- **数据形状**: 144299 行 × 26 列
- **数据完整性**: 88.41%
- **重复行数**: 0

## 数据类型分布
- **数值列**: 5 个
- **分类列**: 16 个  
- **日期列**: 5 个

## 列信息详情
### 数值列
Store Agent Phone, Buyer Cell Phone, Intention Payment Time 小时, buyer_age, Order Number 不同计数

### 分类列
车型分组, pre_vehicle_model_type, pre_vehicle_model, Order Number, Store Agent Name, Store Agent Id, Buyer Identity No, first_middle_channel_name, order_gender, first_big_channel_name, first_small_channel_name, Parent Region Name, License Province, license_city_level, License City, Has_Intention_Payment

### 日期列
Intention_Payment_Time, intention_refund_time, first_assign_time, Lock_Time, first_touch_time

## 异常检测结果

### 缺失值异常

| 列名 | 缺失数量 | 缺失比例 |
|------|----------|----------|
| pre_vehicle_model_type | 123044 | 85.27% |
| pre_vehicle_model | 124433 | 86.23% |
| Store Agent Name | 40 | 0.03% |
| Store Agent Id | 40 | 0.03% |
| Store Agent Phone | 40 | 0.03% |
| Buyer Identity No | 3259 | 2.26% |
| intention_refund_time | 64786 | 44.90% |
| Lock_Time | 114829 | 79.58% |
| buyer_age | 2884 | 2.00% |
| first_touch_time | 801 | 0.56% |
| license_city_level | 788 | 0.55% |

### 重复数据检查
✅ 未发现重复数据

## 数据质量评估
- **整体评级**: 一般 ⚠️
- **完整性得分**: 88.41%
