# 异常检测报告

## 数据概览
- **数据文件**: /Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet
- **生成时间**: 2025-08-25T13:25:03.326268
- **文件大小**: 2.96 MB

## 数据基本信息
- **数据形状**: 138207 行 × 18 列
- **数据完整性**: 95.59%
- **重复行数**: 0

## 数据类型分布
- **数值列**: 3 个
- **分类列**: 11 个  
- **日期列**: 4 个

## 列信息详情
### 数值列
Intention Payment Time 小时, buyer_age, Order Number 不同计数

### 分类列
车型分组, Order Number, first_middle_channel_name, order_gender, first_big_channel_name, first_small_channel_name, Parent Region Name, License Province, license_city_level, License City, Has_Intention_Payment

### 日期列
Intention_Payment_Time, first_assign_time, Lock_Time, first_touch_time

## 异常检测结果

### 缺失值异常

| 列名 | 缺失数量 | 缺失比例 |
|------|----------|----------|
| Lock_Time | 99014 | 71.64% |
| buyer_age | 3764 | 2.72% |
| order_gender | 1 | 0.00% |
| first_touch_time | 892 | 0.65% |
| Parent Region Name | 99 | 0.07% |
| License Province | 5101 | 3.69% |
| license_city_level | 712 | 0.52% |
| License City | 46 | 0.03% |

### 重复数据检查
✅ 未发现重复数据

## 数据质量评估
- **整体评级**: 优秀 ✅
- **完整性得分**: 95.59%
