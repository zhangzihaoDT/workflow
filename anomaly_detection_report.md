# 异常检测报告

## 数据概览
- **数据文件**: /Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet
- **生成时间**: 2025-09-18T09:56:50.450992
- **文件大小**: 16.28 MB

## 数据基本信息
- **数据形状**: 340654 行 × 30 列
- **数据完整性**: 82.08%
- **重复行数**: 0

## 数据类型分布
- **数值列**: 6 个
- **分类列**: 16 个  
- **日期列**: 8 个

## 列信息详情
### 数值列
Store Agent Phone, Buyer Cell Phone, Intention Payment Time 小时, buyer_age, 开票价格, Order Number 不同计数

### 分类列
Order Number, 车型分组, Store City, Store Name, Store Code, pre_vehicle_model_type, Store Agent Name, Store Agent Id, Buyer Identity No, first_middle_channel_name, order_gender, Parent Region Name, License Province, license_city_level, License City, Product Name

### 日期列
store_create_date, Invoice_Upload_Time, Order_Create_Time, Intention_Payment_Time, intention_refund_time, first_assign_time, Lock_Time, first_touch_time

## 异常检测结果

### 缺失值异常

| 列名 | 缺失数量 | 缺失比例 |
|------|----------|----------|
| Store City | 622 | 0.18% |
| pre_vehicle_model_type | 316261 | 92.84% |
| Store Agent Name | 158 | 0.05% |
| Store Agent Id | 158 | 0.05% |
| Store Agent Phone | 158 | 0.05% |
| Buyer Identity No | 128936 | 37.85% |
| Invoice_Upload_Time | 230322 | 67.61% |
| Intention_Payment_Time | 180938 | 53.11% |
| intention_refund_time | 251054 | 73.70% |
| Intention Payment Time 小时 | 180938 | 53.11% |
| Lock_Time | 195200 | 57.30% |
| buyer_age | 112026 | 32.89% |
| first_touch_time | 1697 | 0.50% |
| license_city_level | 2123 | 0.62% |
| 开票价格 | 230322 | 67.61% |

### 重复数据检查
✅ 未发现重复数据

## 数据质量评估
- **整体评级**: 一般 ⚠️
- **完整性得分**: 82.08%
