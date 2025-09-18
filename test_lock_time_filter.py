import pandas as pd
import sys
sys.path.append('skills')
from store_analysis import load_and_analyze_data

def test_lock_time_functionality():
    """测试Lock_Time范围控制功能"""
    print("=== 测试Lock_Time范围控制功能 ===\n")
    
    # 加载数据
    df = load_and_analyze_data()
    if df is None:
        print("数据加载失败")
        return
    
    print(f"原始数据总量: {len(df)} 条记录")
    
    # 检查Lock_Time列是否存在
    if 'Lock_Time' not in df.columns:
        print("数据中不包含Lock_Time列")
        return
    
    # 转换Lock_Time为datetime
    df['Lock_Time'] = pd.to_datetime(df['Lock_Time'], errors='coerce')
    
    # 统计Lock_Time的基本信息
    lock_data = df['Lock_Time'].dropna()
    print(f"包含Lock_Time的记录数: {len(lock_data)}")
    print(f"Lock_Time范围: {lock_data.min()} 至 {lock_data.max()}")
    
    # 测试不同的Lock_Time范围筛选
    test_cases = [
        ("2023-01-01", "2023-12-31", "2023年全年"),
        ("2024-01-01", "2024-06-30", "2024年上半年"),
        ("2024-07-01", "2024-12-31", "2024年下半年"),
    ]
    
    for start_date, end_date, description in test_cases:
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # 应用筛选
            filtered_mask = (df['Lock_Time'] >= start_dt) & (df['Lock_Time'] <= end_dt)
            filtered_df = df[filtered_mask]
            
            print(f"\n{description} ({start_date} 至 {end_date}):")
            print(f"  筛选后记录数: {len(filtered_df)}")
            print(f"  包含Lock_Time的记录: {filtered_df['Lock_Time'].notna().sum()}")
            print(f"  涉及店铺数: {filtered_df['Store Name'].nunique()}")
            print(f"  涉及城市数: {filtered_df['Store City'].nunique()}")
            
        except Exception as e:
            print(f"测试 {description} 时出错: {e}")

if __name__ == "__main__":
    test_lock_time_functionality()
