# Business Daily Metrics 字段清单

## 数据概览

- **数据文件**: /Users/zihao\_/Documents/coding/dataset/formatted/business_daily_metrics.parquet
- **生成时间**: 2025-11-04T16:22:18
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

有效线索数, 抖音战队线索数, 下发线索数, 有效试驾数, 试驾锁单数, 锁单数, 小订数, 小订留存锁单数, 本品牌人群总资产资产, 本品牌日新增, 本品牌日流失, 本品牌人群留存, MG4 小订数, 试驾锁单占比, 小订留存占比, 线索转化率, 抖音线索占比, 锁单数\_7 日均值, 有效试驾数\_7 日均值, 试驾锁单占比\_7 日均值, 小订留存占比\_7 日均值, 线索转化率\_7 日均值, 抖音线索占比\_7 日均值

### 日期列

date

### 缺失值异常

| 列名                   | 缺失数量 | 缺失比例 |
| ---------------------- | -------- | -------- |
| MG4 小订数             | 785      | 97.88%   |
| 小订留存占比\_7 日均值 | 510      | 63.59%   |
| 小订留存锁单数         | 348      | 43.39%   |
| 小订留存占比           | 348      | 43.39%   |
| 小订数                 | 338      | 42.14%   |
| 有效试驾数\_7 日均值   | 29       | 3.62%    |
| 试驾锁单占比\_7 日均值 | 21       | 2.62%    |
| 锁单数\_7 日均值       | 13       | 1.62%    |
| 线索转化率\_7 日均值   | 13       | 1.62%    |
| 有效试驾数             | 6        | 0.75%    |
| 抖音线索占比\_7 日均值 | 6        | 0.75%    |
| 试驾锁单数             | 3        | 0.37%    |
| 试驾锁单占比           | 3        | 0.37%    |
| 锁单数                 | 1        | 0.12%    |
| 线索转化率             | 1        | 0.12%    |

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

- **数据文件**: /Users/zihao\_/Documents/coding/dataset/formatted/intention_order_analysis.parquet
- **生成时间**: 2025-11-04T16:22:19
- **文件大小**: 16.74 MB

## 数据基本信息

- **数据形状**: 350294 行 × 30 列
- **数据完整性**: 82.44%
- **重复行数**: 0

## 数据类型分布

- **数值列**: 6 个
- **分类列**: 16 个
- **日期列**: 8 个

## 列信息详情

### 数值列

Buyer Cell Phone, buyer_age, Store Agent Phone, Intention Payment Time 小时, 开票价格, Order Number 不同计数

### 分类列

Buyer Identity No, first_main_channel_group, License City, License Province, license_city_level, Order Number, order_gender, Parent Region Name, pre_vehicle_model_type, Product Name, Store Agent Id, Store Agent Name, Store City, Store Code, Store Name, 车型分组

### 日期列

first_assign_time, first_touch_time, Intention_Payment_Time, intention_refund_time, Lock_Time, Order_Create_Time, Invoice_Upload_Time, store_create_date

### 缺失值异常

| 列名                        | 缺失数量 | 缺失比例 |
| --------------------------- | -------- | -------- |
| pre_vehicle_model_type      | 325910   | 93.04%   |
| intention_refund_time       | 239922   | 68.49%   |
| Invoice_Upload_Time         | 224027   | 63.95%   |
| 开票价格                    | 224027   | 63.95%   |
| Intention_Payment_Time      | 190280   | 54.32%   |
| Intention Payment Time 小时 | 190280   | 54.32%   |
| Lock_Time                   | 185577   | 52.98%   |
| Buyer Identity No           | 138000   | 39.40%   |
| buyer_age                   | 122152   | 34.87%   |
| license_city_level          | 2153     | 0.61%    |
| first_touch_time            | 2029     | 0.58%    |
| Store City                  | 621      | 0.18%    |
| Store Agent Id              | 148      | 0.04%    |
| Store Agent Name            | 148      | 0.04%    |
| Store Agent Phone           | 148      | 0.04%    |

## 字段列表

- `Buyer Cell Phone`
- `Buyer Identity No`
- `buyer_age`
- `first_assign_time`
- `first_touch_time`
- `Intention_Payment_Time`
- `intention_refund_time`
- `Lock_Time`
- `Order_Create_Time`
- `Invoice_Upload_Time`
- `store_create_date`
- `first_main_channel_group`
- `License City`
- `License Province`
- `license_city_level`
- `Order Number`
- `order_gender`
- `Parent Region Name`
- `pre_vehicle_model_type`
- `Product Name`
- `Store Agent Id`
- `Store Agent Name`
- `Store Agent Phone`
- `Store City`
- `Store Code`
- `Store Name`
- `Intention Payment Time 小时`
- `车型分组`
- `开票价格`
- `Order Number 不同计数`
