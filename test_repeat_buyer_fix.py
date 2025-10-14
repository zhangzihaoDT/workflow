#!/usr/bin/env python3
"""
测试重复买家筛选功能修复
验证所有相关函数是否能正确使用 'Buyer Identity No' 和 'Buyer Cell Phone' 列名
"""

import sys
import os
sys.path.append('/Users/zihao_/Documents/github/W35_workflow/skills')

from order_trend_monitor import OrderTrendMonitor
import pandas as pd

def test_repeat_buyer_filtering():
    """测试重复买家筛选功能的修复"""
    print("开始测试重复买家筛选功能修复...")
    
    # 初始化监控器
    monitor = OrderTrendMonitor()
    
    # 测试参数
    selected_vehicles = ["CM1", "CM2"]
    
    # 测试 prepare_product_name_lock_data
    try:
        print("\n测试 prepare_product_name_lock_data...")
        result = monitor.prepare_product_name_lock_data(
            selected_vehicles=selected_vehicles,
            start_date="2025-08-15",
            end_date="2025-12-31",
            product_types=["增程", "纯电"],
            lock_start_date="2025-09-10",
            lock_end_date="2025-12-31",
            lock_n_days=30,
            weekend_lock_filter="全部",
            include_repeat_buyers=True,
            include_repeat_buyers_combo=True
        )
        print(f"✅ prepare_product_name_lock_data 测试成功，返回数据形状: {result.shape}")
    except Exception as e:
        print(f"❌ prepare_product_name_lock_data 测试失败: {e}")
    
    # 测试 prepare_channel_lock_data
    try:
        print("\n测试 prepare_channel_lock_data...")
        result = monitor.prepare_channel_lock_data(
            selected_vehicles=selected_vehicles,
            start_date="2025-08-15",
            end_date="2025-12-31"
        )
        print(f"✅ prepare_channel_lock_data 测试成功，返回数据形状: {result.shape}")
    except Exception as e:
        print(f"❌ prepare_channel_lock_data 测试失败: {e}")
    
    # 测试 prepare_age_lock_data
    try:
        print("\n测试 prepare_age_lock_data...")
        result = monitor.prepare_age_lock_data(
            selected_vehicles=selected_vehicles,
            start_date="2025-08-15",
            end_date="2025-12-31",
            product_types=["增程", "纯电"],
            lock_start_date="2025-09-10",
            lock_end_date="2025-12-31",
            lock_n_days=30,
            include_unknown=True,
            weekend_lock_filter="全部",
            include_repeat_buyers=True,
            include_repeat_buyers_combo=True
        )
        print(f"✅ prepare_age_lock_data 测试成功，返回数据形状: {result.shape}")
    except Exception as e:
        print(f"❌ prepare_age_lock_data 测试失败: {e}")
    
    # 测试 prepare_gender_lock_data
    try:
        print("\n测试 prepare_gender_lock_data...")
        result = monitor.prepare_gender_lock_data(
            selected_vehicles=selected_vehicles,
            start_date="2025-08-15",
            end_date="2025-12-31",
            product_types=["增程", "纯电"],
            lock_start_date="2025-09-10",
            lock_end_date="2025-12-31",
            lock_n_days=30,
            include_unknown=True,
            weekend_lock_filter="全部",
            include_repeat_buyers=True,
            include_repeat_buyers_combo=True
        )
        print(f"✅ prepare_gender_lock_data 测试成功，返回数据形状: {result.shape}")
    except Exception as e:
        print(f"❌ prepare_gender_lock_data 测试失败: {e}")
    
    # 测试 prepare_region_lock_data
    try:
        print("\n测试 prepare_region_lock_data...")
        result = monitor.prepare_region_lock_data(
            selected_vehicles=selected_vehicles,
            start_date="2025-08-15",
            end_date="2025-12-31",
            product_types=["增程", "纯电"],
            lock_start_date="2025-09-10",
            lock_end_date="2025-12-31",
            lock_n_days=30,
            include_unknown=True,
            include_virtual=True,
            include_fac=True,
            weekend_lock_filter="全部",
            include_repeat_buyers=True,
            include_repeat_buyers_combo=True
        )
        print(f"✅ prepare_region_lock_data 测试成功，返回数据形状: {result.shape}")
    except Exception as e:
        print(f"❌ prepare_region_lock_data 测试失败: {e}")
    
    # 测试 prepare_city_lock_data
    try:
        print("\n测试 prepare_city_lock_data...")
        result = monitor.prepare_city_lock_data(
            selected_vehicles=selected_vehicles,
            order_start_date="2025-08-15",
            order_end_date="2025-12-31",
            lock_start_date="2025-09-10",
            lock_end_date="2025-12-31",
            lock_n_days=30,
            product_types=["增程", "纯电"],
            weekend_lock_filter="全部",
            min_lock_count=100,
            max_lock_count=1000,
            include_repeat_buyers=True,
            include_repeat_buyers_combo=True
        )
        print(f"✅ prepare_city_lock_data 测试成功，返回数据形状: {result.shape}")
    except Exception as e:
        print(f"❌ prepare_city_lock_data 测试失败: {e}")
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_repeat_buyer_filtering()