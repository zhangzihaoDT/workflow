# W35 异常检测工作流 - 综合分析报告

## 报告概览
- **生成时间**: 2025-10-20T18:07:49.380549
- **工作流版本**: W35 Anomaly Detection Workflow
- **分析范围**: 数据质量检测 + 结构异常分析 + 销售代理分析 + 时间间隔分析

---

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


---



---



---



---

## 综合结论

### 数据质量状况
从数据质量检测结果来看，数据的基本完整性和一致性情况。

### 结构异常状况
从结构检查结果来看，CM2车型相对于历史车型在地区分布、渠道结构、人群结构方面的变化情况。

### 销售代理分析状况
从销售代理分析结果来看，不同车型在预售周期中来自Store Agent的订单比例情况。

### 重复买家分析状况
从重复买家分析结果来看，不同车型在预售周期中重复购买（同一身份证号码对应多个订单）的情况。

### 时间间隔分析状况
从时间间隔分析结果来看，不同车型在支付到退款、支付到分配等关键业务流程的时间效率表现。

### 建议措施
1. **数据质量方面**: 根据异常检测结果，对发现的数据质量问题进行相应处理
2. **结构异常方面**: 对发现的结构异常进行深入分析，确定是否为正常的业务变化或需要关注的异常情况
3. **销售代理方面**: 根据销售代理订单比例分析结果，评估各车型的销售渠道效果
4. **重复买家方面**: 根据重复买家订单比例分析结果，评估客户忠诚度和复购行为模式
5. **时间间隔分析方面**: 根据时间间隔分析结果，优化业务流程效率，关注异常时间间隔模式
6. **持续监控**: 建议定期运行此工作流，持续监控数据质量和结构变化

---

*本报告由 W35 异常检测工作流自动生成，整合了数据质量检测、结构异常分析、销售代理分析和时间间隔分析的综合结果*
