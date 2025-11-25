# 定金金额对锁单率影响分析
- base AUC（不含定金）: 0.6358
- deposit-only AUC（仅定金）: 0.5049
- full AUC（加入定金）: 0.6358
- full 逻辑回归中 `deposit_amount` 系数: 0.049775

## full 模型特征权重Top15（按绝对值）
- first_main_channel_group_te: 0.301311
- License City_te: 0.235100
- interval_presale_to_pay_days: 0.234242
- order_gender_默认未知: -0.221604
- order_gender_女: 0.203914
- Parent Region Name_te: 0.122793
- License City_fe: -0.059547
- deposit_amount: 0.049775
- order_gender_男: -0.040724
- is_repeat_buyer: -0.038766
- cumulative_order_count: -0.035925
- Parent Region Name_fe: 0.029759
- interval_touch_to_pay_days: 0.027333
- interval_assign_to_pay_days: -0.024694
- buyer_age: 0.014785
