#!/usr/bin/env python3
"""
测试年龄统计功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from order_trend_monitor import OrderTrendMonitor

def test_age_statistics():
    """测试年龄统计功能"""
    print("🧪 开始测试年龄统计功能...")
    
    # 初始化监控器
    monitor = OrderTrendMonitor()
    
    # 测试参数
    selected_vehicles = ["CM2", "CM1"]
    start_date = "2025-08-15"
    end_date = ""
    product_categories = []
    lock_start_date = "2025-09-10"
    lock_end_date = ""
    lock_n_days = 30
    age_include_unknown = True
    weekend_lock_filter = "全部"
    include_repeat_buyers = True
    include_repeat_buyers_combo = True
    
    try:
        # 测试年龄统计计算函数
        print("📊 测试 calculate_age_statistics 函数...")
        age_stats_data = monitor.calculate_age_statistics(
            selected_vehicles, start_date, end_date, product_categories, 
            lock_start_date, lock_end_date, lock_n_days, age_include_unknown, 
            weekend_lock_filter, include_repeat_buyers, include_repeat_buyers_combo
        )
        
        print(f"✅ 年龄统计数据计算成功！")
        print(f"📈 数据形状: {age_stats_data.shape}")
        print(f"📋 列名: {list(age_stats_data.columns)}")
        print(f"🔍 前5行数据:")
        print(age_stats_data.head())
        
        # 检查必要的列是否存在
        expected_columns = ['车型', '平均年龄', '中位数', '方差']
        missing_columns = [col for col in expected_columns if col not in age_stats_data.columns]
        
        if missing_columns:
            print(f"❌ 缺少必要的列: {missing_columns}")
            return False
        else:
            print(f"✅ 所有必要的列都存在: {expected_columns}")
        
        # 检查数据类型
        print(f"📊 数据类型:")
        for col in age_stats_data.columns:
            print(f"  {col}: {age_stats_data[col].dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ 年龄统计功能测试失败: {str(e)}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_age_statistics()
    if success:
        print("\n🎉 年龄统计功能测试通过！")
    else:
        print("\n💥 年龄统计功能测试失败！")
        sys.exit(1)