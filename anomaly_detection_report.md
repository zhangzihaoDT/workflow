# 异常检测报告

## 数据概览
- **数据文件**: /Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet
- **生成时间**: 2025-10-20T18:07:49.022867
- **文件大小**: 16.56 MB

## 数据基本信息
- **数据形状**: 346714 行 × 30 列
- **数据完整性**: 82.36%
- **重复行数**: 0

## 数据类型分布
- **数值列**: 6 个
- **分类列**: 16 个  
- **日期列**: 8 个

## 列信息详情
### 数值列
Buyer Cell Phone, buyer_age, Store Agent Phone, Intention Payment Time 小时, 开票价格, Order Number 不同计数

### 分类列
Buyer Identity No, first_middle_channel_name, License City, License Province, license_city_level, Order Number, order_gender, Parent Region Name, pre_vehicle_model_type, Product Name, Store Agent Id, Store Agent Name, Store City, Store Code, Store Name, 车型分组

### 日期列
first_assign_time, first_touch_time, Intention_Payment_Time, intention_refund_time, Lock_Time, Order_Create_Time, Invoice_Upload_Time, store_create_date

## 异常检测结果

### 缺失值异常

| 列名 | 缺失数量 | 缺失比例 |
|------|----------|----------|
| Buyer Identity No | 134654 | 38.84% |
| buyer_age | 117281 | 33.83% |
| first_touch_time | 1948 | 0.56% |
| Intention_Payment_Time | 186804 | 53.88% |
| intention_refund_time | 243232 | 70.15% |
| Lock_Time | 185265 | 53.43% |
| Invoice_Upload_Time | 226707 | 65.39% |
| license_city_level | 2146 | 0.62% |
| pre_vehicle_model_type | 322330 | 92.97% |
| Store Agent Id | 136 | 0.04% |
| Store Agent Name | 136 | 0.04% |
| Store Agent Phone | 136 | 0.04% |
| Store City | 621 | 0.18% |
| Intention Payment Time 小时 | 186804 | 53.88% |
| 开票价格 | 226707 | 65.39% |

### 重复数据检查
✅ 未发现重复数据

## 数据质量评估
- **整体评级**: 一般 ⚠️
- **完整性得分**: 82.36%
