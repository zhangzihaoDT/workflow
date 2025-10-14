#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Parent Region平均年龄功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from order_trend_monitor import OrderTrendMonitor

def test_parent_region_avg_age():
    """测试Parent Region平均年龄功能"""
    print("开始测试Parent Region平均年龄功能...")
    
    # 初始化监控器
    monitor = OrderTrendMonitor()
    
    # 设置测试参数
    selected_vehicles = ["CM1", "CM2"]
    start_date = "2024-09-01"
    end_date = "2024-12-31"
    lock_start_date = "2024-09-10"
    lock_end_date = "2024-12-31"
    lock_n_days = 30
    product_types = []
    weekend_lock_filter = "全部"
    
    print(f"测试参数:")
    print(f"  车型: {selected_vehicles}")
    print(f"  小订时间范围: {start_date} 到 {end_date}")
    print(f"  锁单时间范围: {lock_start_date} 到 {lock_end_date}")
    print(f"  锁单后天数: {lock_n_days}")
    
    # 调用prepare_region_lock_data函数
    try:
        region_data = monitor.prepare_region_lock_data(
            selected_vehicles=selected_vehicles,
            start_date=start_date,
            end_date=end_date,
            lock_start_date=lock_start_date,
            lock_end_date=lock_end_date,
            lock_n_days=lock_n_days,
            product_types=product_types,
            weekend_lock_filter=weekend_lock_filter
        )
        
        print(f"\n✅ prepare_region_lock_data 执行成功")
        print(f"   数据形状: {region_data.shape}")
        print(f"   列名: {list(region_data.columns)}")
        
        # 检查是否包含平均年龄列
        if '平均年龄' in region_data.columns:
            print(f"✅ 成功添加'平均年龄'列")
            print(f"   平均年龄数据样例:")
            for i, row in region_data.head().iterrows():
                print(f"     {row['父级区域']}: {row['平均年龄']}")
        else:
            print(f"❌ 未找到'平均年龄'列")
            return False
        
        # 调用create_region_lock_table函数
        table_data = monitor.create_region_lock_table(region_data)
        
        print(f"\n✅ create_region_lock_table 执行成功")
        print(f"   表格数据形状: {table_data.shape}")
        print(f"   表格列名: {list(table_data.columns)}")
        
        # 检查表格是否包含平均年龄列
        if '平均年龄' in table_data.columns:
            print(f"✅ 表格成功包含'平均年龄'列")
            print(f"   表格平均年龄数据样例:")
            for i, row in table_data.head().iterrows():
                print(f"     {row['父级区域']}: {row['平均年龄']}")
        else:
            print(f"❌ 表格未包含'平均年龄'列")
            return False
        
        print(f"\n🎉 Parent Region平均年龄功能测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_parent_region_avg_age()
    if success:
        print("\n✅ 所有测试通过")
    else:
        print("\n❌ 测试失败")
        sys.exit(1)