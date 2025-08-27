# 异常检测报告

## 数据概览
- **数据文件**: /Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet
- **生成时间**: 2025-08-27T10:41:04.669466
- **文件大小**: 2.80 MB

## 数据基本信息
- **数据形状**: 119143 行 × 19 列
- **数据完整性**: 93.92%
- **重复行数**: 0

## 数据类型分布
- **数值列**: 3 个
- **分类列**: 11 个  
- **日期列**: 5 个

## 列信息详情
### 数值列
Intention Payment Time 小时, buyer_age, Order Number 不同计数

### 分类列
车型分组, Order Number, first_middle_channel_name, order_gender, first_big_channel_name, first_small_channel_name, Parent Region Name, License Province, license_city_level, License City, Has_Intention_Payment

### 日期列
Intention_Payment_Time, intention_refund_time, first_assign_time, Lock_Time, first_touch_time

## 异常检测结果

### 缺失值异常

| 列名 | 缺失数量 | 缺失比例 |
|------|----------|----------|
| intention_refund_time | 43901 | 36.85% |
| first_assign_time | 1 | 0.00% |
| Lock_Time | 89689 | 75.28% |
| buyer_age | 2660 | 2.23% |
| first_touch_time | 744 | 0.62% |
| license_city_level | 628 | 0.53% |

### 重复数据检查
✅ 未发现重复数据

## 数据质量评估
- **整体评级**: 良好 ⚠️
- **完整性得分**: 93.92%
