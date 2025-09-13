import pandas as pd
from datetime import datetime

# 数据路径
DATA_PATH = "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"

# 加载数据
print("正在加载数据...")
df = pd.read_parquet(DATA_PATH)
print(f"数据加载完成，共{len(df)}条记录")

# 查看数据结构
print("\n数据列名:")
print(df.columns.tolist())

# 查看车型分组的唯一值
print("\n车型分组唯一值:")
print(df['车型分组'].unique())

# 筛选条件
print("\n开始筛选数据...")

# 条件1: 车型分组=CM2
cm2_data = df[df['车型分组'] == 'CM2']
print(f"车型分组=CM2的订单数: {len(cm2_data)}")

# 条件2: Intention_Payment_Time=2025-08-15
target_date = '2025-08-15'
cm2_target_date = cm2_data[cm2_data['Intention_Payment_Time'].dt.date == pd.to_datetime(target_date).date()]
print(f"车型分组=CM2且Intention_Payment_Time=2025-08-15的订单数: {len(cm2_target_date)}")

# 条件3: lock_time<2025-09-11
lock_cutoff = '2025-09-11'
# 检查是否有Lock_Time列
if 'Lock_Time' in cm2_target_date.columns:
    # 筛选有Lock_Time且Lock_Time<2025-09-11的订单
    final_result = cm2_target_date[
        (cm2_target_date['Lock_Time'].notna()) & 
        (cm2_target_date['Lock_Time'].dt.date < pd.to_datetime(lock_cutoff).date())
    ]
    print(f"最终满足所有条件的订单数: {len(final_result)}")
    
    # 显示详细信息
    print("\n详细统计:")
    print(f"- 车型分组=CM2: {len(cm2_data)}")
    print(f"- 车型分组=CM2 且 Intention_Payment_Time=2025-08-15: {len(cm2_target_date)}")
    print(f"- 上述条件 且 有Lock_Time: {len(cm2_target_date[cm2_target_date['Lock_Time'].notna()])}")
    print(f"- 上述条件 且 Lock_Time<2025-09-11: {len(final_result)}")
    
    # 查看Lock_Time的分布
    if len(cm2_target_date[cm2_target_date['Lock_Time'].notna()]) > 0:
        print("\nLock_Time分布:")
        lock_times = cm2_target_date[cm2_target_date['Lock_Time'].notna()]['Lock_Time'].dt.date
        print(lock_times.value_counts().sort_index())
else:
    print("数据中没有Lock_Time列")
    print("可用的时间相关列:")
    time_cols = [col for col in cm2_target_date.columns if 'time' in col.lower() or 'date' in col.lower()]
    print(time_cols)

print("\n验证完成!")