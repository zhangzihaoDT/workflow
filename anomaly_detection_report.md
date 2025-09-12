# 异常检测报告

## 数据概览
- **数据文件**: /Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet
- **生成时间**: 2025-09-12T08:59:50.353204
- **文件大小**: 14.27 MB

## 数据基本信息
- **数据形状**: 339073 行 × 26 列
- **数据完整性**: 84.44%
- **重复行数**: 0

## 数据类型分布
- **数值列**: 5 个
- **分类列**: 15 个  
- **日期列**: 6 个

## 列信息详情
### 数值列
Store Agent Phone, Buyer Cell Phone, Intention Payment Time 小时, buyer_age, Order Number 不同计数

### 分类列
车型分组, pre_vehicle_model_type, Order Number, Store Agent Name, Store Agent Id, Buyer Identity No, first_middle_channel_name, order_gender, first_big_channel_name, first_small_channel_name, Parent Region Name, License Province, license_city_level, License City, Product Name

### 日期列
Order_Create_Time, Intention_Payment_Time, intention_refund_time, first_assign_time, Lock_Time, first_touch_time

## 异常检测结果

### 缺失值异常

| 列名 | 缺失数量 | 缺失比例 |
|------|----------|----------|
| pre_vehicle_model_type | 314653 | 92.80% |
| Store Agent Name | 161 | 0.05% |
| Store Agent Id | 161 | 0.05% |
| Store Agent Phone | 161 | 0.05% |
| Buyer Identity No | 127388 | 37.57% |
| Intention_Payment_Time | 179400 | 52.91% |
| intention_refund_time | 257142 | 75.84% |
| Intention Payment Time 小时 | 179400 | 52.91% |
| Lock_Time | 199197 | 58.75% |
| buyer_age | 110735 | 32.66% |
| first_touch_time | 1628 | 0.48% |
| license_city_level | 2117 | 0.62% |

### 重复数据检查
✅ 未发现重复数据

## 数据质量评估
- **整体评级**: 一般 ⚠️
- **完整性得分**: 84.44%
