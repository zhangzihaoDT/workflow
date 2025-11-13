# Business Daily Metrics 字段清单

## 数据概览
- **数据文件**: /Users/zihao_/Documents/coding/dataset/formatted/business_daily_metrics.parquet
- **生成时间**: 2025-11-13T10:43:29
- **文件大小**: 0.12 MB

## 数据基本信息
- **数据形状**: 802 行 × 24 列
- **数据完整性**: 87.40%
- **重复行数**: 0

## 数据类型分布
- **数值列**: 23 个
- **分类列**: 0 个  
- **日期列**: 1 个

## 列信息详情
### 数值列
有效线索数, 抖音战队线索数, 下发线索数, 有效试驾数, 试驾锁单数, 锁单数, 小订数, 小订留存锁单数, 本品牌人群总资产资产, 本品牌日新增, 本品牌日流失, 本品牌人群留存, MG4小订数, 试驾锁单占比, 小订留存占比, 线索转化率, 抖音线索占比, 锁单数_7日均值, 有效试驾数_7日均值, 试驾锁单占比_7日均值, 小订留存占比_7日均值, 线索转化率_7日均值, 抖音线索占比_7日均值

### 日期列
date


### 缺失值异常

| 列名 | 缺失数量 | 缺失比例 |
|------|----------|----------|
| MG4小订数 | 785 | 97.88% |
| 小订留存占比_7日均值 | 510 | 63.59% |
| 小订留存锁单数 | 348 | 43.39% |
| 小订留存占比 | 348 | 43.39% |
| 小订数 | 338 | 42.14% |
| 有效试驾数_7日均值 | 29 | 3.62% |
| 试驾锁单占比_7日均值 | 21 | 2.62% |
| 锁单数_7日均值 | 13 | 1.62% |
| 线索转化率_7日均值 | 13 | 1.62% |
| 有效试驾数 | 6 | 0.75% |
| 抖音线索占比_7日均值 | 6 | 0.75% |
| 试驾锁单数 | 3 | 0.37% |
| 试驾锁单占比 | 3 | 0.37% |
| 锁单数 | 1 | 0.12% |
| 线索转化率 | 1 | 0.12% |

## 字段列表
- `date`
- `有效线索数`
- `抖音战队线索数`
- `下发线索数`
- `有效试驾数`
- `试驾锁单数`
- `锁单数`
- `小订数`
- `小订留存锁单数`
- `本品牌人群总资产资产`
- `本品牌日新增`
- `本品牌日流失`
- `本品牌人群留存`
- `MG4小订数`
- `试驾锁单占比`
- `小订留存占比`
- `线索转化率`
- `抖音线索占比`
- `锁单数_7日均值`
- `有效试驾数_7日均值`
- `试驾锁单占比_7日均值`
- `小订留存占比_7日均值`
- `线索转化率_7日均值`
- `抖音线索占比_7日均值`

# 附：意向订单分析字段清单

## 数据概览
- **数据文件**: /Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet
- **生成时间**: 2025-11-13T10:43:29
- **文件大小**: 0.94 MB

## 数据基本信息
- **数据形状**: 17592 行 × 30 列
- **数据完整性**: 80.53%
- **重复行数**: 0

## 数据类型分布
- **数值列**: 7 个
- **分类列**: 15 个  
- **日期列**: 8 个

## 列信息详情
### 数值列
pre_vehicle_model_type, Store Agent Phone, Buyer Cell Phone, Intention Payment Time 小时, buyer_age, 开票价格, Order Number 不同计数

### 分类列
Order Number, 车型分组, Store City, Store Name, Store Code, Store Agent Name, Store Agent Id, Buyer Identity No, first_main_channel_group, order_gender, Parent Region Name, License Province, license_city_level, License City, Product Name

### 日期列
store_create_date, Invoice_Upload_Time, Order_Create_Time, Intention_Payment_Time, intention_refund_time, first_assign_time, Lock_Time, first_touch_time


### 缺失值异常

| 列名 | 缺失数量 | 缺失比例 |
|------|----------|----------|
| pre_vehicle_model_type | 17592 | 100.00% |
| Invoice_Upload_Time | 17086 | 97.12% |
| 开票价格 | 17086 | 97.12% |
| intention_refund_time | 16857 | 95.82% |
| Lock_Time | 13239 | 75.26% |
| Intention Payment Time 小时 | 7607 | 43.24% |
| Buyer Identity No | 4542 | 25.82% |
| Intention_Payment_Time | 4433 | 25.20% |
| buyer_age | 4087 | 23.23% |
| license_city_level | 156 | 0.89% |
| Store Agent Name | 30 | 0.17% |
| Store Agent Id | 30 | 0.17% |
| Store Agent Phone | 30 | 0.17% |
| first_touch_time | 6 | 0.03% |

## 字段列表
- `Order Number`
- `车型分组`
- `Store City`
- `Store Name`
- `Store Code`
- `store_create_date`
- `pre_vehicle_model_type`
- `Store Agent Name`
- `Store Agent Id`
- `Store Agent Phone`
- `Buyer Cell Phone`
- `Buyer Identity No`
- `Invoice_Upload_Time`
- `Order_Create_Time`
- `Intention_Payment_Time`
- `intention_refund_time`
- `first_main_channel_group`
- `first_assign_time`
- `Intention Payment Time 小时`
- `Lock_Time`
- `buyer_age`
- `order_gender`
- `first_touch_time`
- `Parent Region Name`
- `License Province`
- `license_city_level`
- `License City`
- `Product Name`
- `开票价格`
- `Order Number 不同计数`

# 附：CM2 配置明细字段清单（最新）

## 数据概览
- **数据文件**: /Users/zihao_/Documents/coding/dataset/processed/CM2_Configuration_Details_transposed_20251107_123143.csv
- **生成时间**: 2025-11-13T10:43:29
- **文件大小**: 3.47 MB

## 数据基本信息
- **数据形状**: 27952 行 × 14 列
- **数据完整性**: 96.60%
- **重复行数**: 0

## 数据类型分布
- **数值列**: 3 个
- **分类列**: 9 个  
- **日期列**: 2 个

## 列信息详情
### 数值列
OP-FRIDGE, OP-LuxGift, 开票价格

### 分类列
order_number, EXCOLOR, INCOLOR, OP-LASER, OP-SW, WHEEL, Is Staff, Product Name, Product_Types

### 日期列
invoice_time, lock_time


### 缺失值异常

| 列名 | 缺失数量 | 缺失比例 |
|------|----------|----------|
| invoice_time | 13259 | 47.43% |
| OP-FRIDGE | 25 | 0.09% |
| OP-LASER | 25 | 0.09% |

## 字段列表
- `order_number`
- `EXCOLOR`
- `INCOLOR`
- `OP-FRIDGE`
- `OP-LASER`
- `OP-LuxGift`
- `OP-SW`
- `WHEEL`
- `invoice_time`
- `lock_time`
- `Is Staff`
- `Product Name`
- `Product_Types`
- `开票价格`
