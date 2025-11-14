# 定金金额对锁单率影响分析
- base AUC（不含定金）: 0.6399
- deposit-only AUC（仅定金）: 0.5169
- full AUC（加入定金）: 0.6416
- full 逻辑回归中 `deposit_amount` 系数: -0.061284

## full 模型特征权重Top15（按绝对值）
- first_main_channel_group_te: 0.315945
- interval_presale_to_pay_days: 0.251100
- License City_te: 0.239231
- order_gender_女: 0.200484
- order_gender_默认未知: -0.158755
- order_gender_男: -0.082792
- Parent Region Name_te: 0.062415
- deposit_amount: -0.061284
- License City_fe: -0.055842
- interval_assign_to_pay_days: -0.038396
- is_repeat_buyer: -0.038050
- interval_touch_to_pay_days: 0.027068
- Parent Region Name_fe: 0.018560
- first_main_channel_group_fe: -0.017391
- last_invoice_gap_days: 0.013590
