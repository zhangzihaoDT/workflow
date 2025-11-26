# Business Daily Metrics 字段清单

## 数据概览
- **数据文件**: /Users/zihao_/Documents/coding/dataset/formatted/business_daily_metrics.parquet
- **生成时间**: 2025-11-26T15:32:05
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
- **生成时间**: 2025-11-26T15:32:05
- **文件大小**: 26.90 MB

## 数据基本信息
- **数据形状**: 419490 行 × 36 列
- **数据完整性**: 76.09%
- **重复行数**: 0

## 数据类型分布
- **数值列**: 5 个
- **分类列**: 21 个  
- **日期列**: 10 个

## 列信息详情
### 数值列
Intention Payment Time 小时, owner_age, buyer_age, 开票价格, Order Number 不同计数

### 分类列
Order Number, 车型分组, Store City, Store Name, Store Code, pre_vehicle_model_type, Store Agent Name, Store Agent Id, Store Agent Phone, first_main_channel_group, Owner Cell Phone, Owner Identity No, owner_gender, Buyer Cell Phone, Buyer Identity No, order_gender, Parent Region Name, License Province, license_city_level, License City, Product Name

### 日期列
store_create_date, Deposit_Payment_Time, Invoice_Upload_Time, Order_Create_Time, Intention_Payment_Time, intention_refund_time, deposit_refund_time, first_assign_time, Lock_Time, first_touch_time


### 缺失值异常

| 列名 | 缺失数量 | 缺失比例 |
|------|----------|----------|
| pre_vehicle_model_type | 395128 | 94.19% |
| deposit_refund_time | 384130 | 91.57% |
| intention_refund_time | 290188 | 69.18% |
| 开票价格 | 277566 | 66.17% |
| Invoice_Upload_Time | 277566 | 66.17% |
| Intention Payment Time 小时 | 225763 | 53.82% |
| Intention_Payment_Time | 225763 | 53.82% |
| Lock_Time | 223344 | 53.24% |
| Deposit_Payment_Time | 206736 | 49.28% |
| owner_age | 204996 | 48.87% |
| Owner Identity No | 196343 | 46.81% |
| owner_gender | 188174 | 44.86% |
| Owner Cell Phone | 177536 | 42.32% |
| Buyer Identity No | 165105 | 39.36% |
| buyer_age | 157515 | 37.55% |
| License Province | 6638 | 1.58% |
| first_touch_time | 2766 | 0.66% |
| license_city_level | 2642 | 0.63% |
| Store City | 1522 | 0.36% |
| Store Agent Phone | 207 | 0.05% |
| Store Agent Id | 207 | 0.05% |
| Store Agent Name | 207 | 0.05% |
| Parent Region Name | 136 | 0.03% |
| Store Name | 136 | 0.03% |
| store_create_date | 136 | 0.03% |
| Store Code | 134 | 0.03% |
| License City | 71 | 0.02% |
| Buyer Cell Phone | 1 | 0.00% |
| order_gender | 1 | 0.00% |

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
- `Deposit_Payment_Time`
- `Invoice_Upload_Time`
- `Order_Create_Time`
- `Intention_Payment_Time`
- `intention_refund_time`
- `deposit_refund_time`
- `first_main_channel_group`
- `first_assign_time`
- `Intention Payment Time 小时`
- `Lock_Time`
- `Owner Cell Phone`
- `Owner Identity No`
- `owner_age`
- `owner_gender`
- `Buyer Cell Phone`
- `Buyer Identity No`
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
- **数据文件**: /Users/zihao_/Documents/coding/dataset/processed/CM2_Configuration_Details_transposed_20251121_164043.csv
- **生成时间**: 2025-11-26T15:32:05
- **文件大小**: 3.73 MB

## 数据基本信息
- **数据形状**: 29870 行 × 14 列
- **数据完整性**: 97.12%
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
| invoice_time | 11992 | 40.15% |
| OP-FRIDGE | 26 | 0.09% |
| OP-LASER | 26 | 0.09% |

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
