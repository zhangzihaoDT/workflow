#!/usr/bin/env python3
"""
测试新的复购用户筛选功能
验证"仅复购用户"和"排除复购用户"两种模式是否正常工作
同时输出特定条件的订单清单
"""

import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ab_comparison_analysis import ABComparisonAnalyzer

def test_new_repeat_buyer_feature():
    """测试新的复购用户筛选功能"""
    print("🧪 开始测试新的复购用户筛选功能...")
    
    # 初始化分析器
    analyzer = ABComparisonAnalyzer()
    
    # 设置测试参数
    test_start_date = "2024-01-01"
    test_end_date = "2024-12-31"
    test_lock_start_date = "2024-06-01"
    
    print(f"📅 测试时间范围: {test_start_date} 到 {test_end_date}")
    print(f"🔒 锁单开始日期: {test_lock_start_date}")
    
    # 测试1: 默认情况（不筛选复购用户）
    print("\n📊 测试1: 默认情况（不筛选复购用户）")
    sample_default = analyzer.filter_sample(
        start_date=test_start_date,
        end_date=test_end_date,
        lock_start_date=test_lock_start_date,
        repeat_buyer_only=False,
        exclude_repeat_buyer=False
    )
    print(f"默认样本大小: {len(sample_default)} 条记录")
    
    # 测试2: 仅复购用户
    print("\n📊 测试2: 仅复购用户")
    sample_repeat_only = analyzer.filter_sample(
        start_date=test_start_date,
        end_date=test_end_date,
        lock_start_date=test_lock_start_date,
        repeat_buyer_only=True,
        exclude_repeat_buyer=False
    )
    print(f"仅复购用户样本大小: {len(sample_repeat_only)} 条记录")
    
    # 测试3: 排除复购用户
    print("\n📊 测试3: 排除复购用户")
    sample_exclude_repeat = analyzer.filter_sample(
        start_date=test_start_date,
        end_date=test_end_date,
        lock_start_date=test_lock_start_date,
        repeat_buyer_only=False,
        exclude_repeat_buyer=True
    )
    print(f"排除复购用户样本大小: {len(sample_exclude_repeat)} 条记录")
    
    # 测试4: 验证逻辑一致性
    print("\n🔍 测试4: 验证逻辑一致性")
    expected_total = len(sample_repeat_only) + len(sample_exclude_repeat)
    actual_total = len(sample_default)
    
    print(f"仅复购用户 + 排除复购用户 = {len(sample_repeat_only)} + {len(sample_exclude_repeat)} = {expected_total}")
    print(f"默认样本总数 = {actual_total}")
    
    if expected_total == actual_total:
        print("✅ 逻辑一致性测试通过！")
    else:
        print(f"❌ 逻辑一致性测试失败！差异: {abs(expected_total - actual_total)} 条记录")
    
    # 测试5: 验证互斥性
    print("\n🔍 测试5: 验证互斥性（同时设置两个选项）")
    sample_both = analyzer.filter_sample(
        start_date=test_start_date,
        end_date=test_end_date,
        lock_start_date=test_lock_start_date,
        repeat_buyer_only=True,
        exclude_repeat_buyer=True
    )
    print(f"同时设置两个选项的样本大小: {len(sample_both)} 条记录")
    
    if len(sample_both) == 0:
        print("✅ 互斥性测试通过！同时设置两个选项时返回空结果")
    else:
        print("❌ 互斥性测试失败！同时设置两个选项时应该返回空结果")
    
    # 测试6: 检查复购用户识别
    print("\n🔍 测试6: 检查复购用户识别详情")
    if len(sample_repeat_only) > 0:
        # 检查仅复购用户样本中的用户是否确实有多个订单
        repeat_buyers = sample_repeat_only.groupby('Buyer Identity No').size()
        multi_order_users = repeat_buyers[repeat_buyers > 1]
        print(f"仅复购用户样本中有多个订单的用户数: {len(multi_order_users)}")
        print(f"仅复购用户样本中的总用户数: {len(repeat_buyers)}")
        
        if len(multi_order_users) > 0:
            print("✅ 复购用户识别正确！")
        else:
            print("⚠️  复购用户识别可能有问题")
    
    print("\n🎉 新的复购用户筛选功能测试完成！")
    
    return {
        'default_count': len(sample_default),
        'repeat_only_count': len(sample_repeat_only),
        'exclude_repeat_count': len(sample_exclude_repeat),
        'both_options_count': len(sample_both),
        'logic_consistent': expected_total == actual_total,
        'mutually_exclusive': len(sample_both) == 0
    }

def get_specific_order_list():
    """
    获取特定条件的订单清单：
    - 小定时间范围为空（不以小定时间做筛选条件）
    - 锁单时间在2025-09-10～2025-10-14之间
    - 车型=CM2
    """
    print("\n🔍 开始筛选特定条件的订单清单...")
    
    # 初始化分析器
    analyzer = ABComparisonAnalyzer()
    
    # 筛选条件
    print("📋 筛选条件:")
    print("- 小定时间范围为空（不以小定时间做筛选条件）")
    print("- 锁单时间在2025-09-10～2025-10-14之间")
    print("- 车型=CM2")
    
    # 应用筛选条件
    filtered_data = analyzer.df.copy()
    
    # 条件1: 锁单时间在2025-09-10～2025-10-14之间
    lock_start_date = '2025-09-10'
    lock_end_date = '2025-10-14'
    
    # 确保Lock_Time是datetime类型
    filtered_data['Lock_Time'] = pd.to_datetime(filtered_data['Lock_Time'])
    
    # 筛选锁单时间范围
    filtered_data = filtered_data[
        (filtered_data['Lock_Time'] >= lock_start_date) & 
        (filtered_data['Lock_Time'] <= lock_end_date)
    ]
    print(f"✅ 锁单时间在{lock_start_date}～{lock_end_date}的订单数: {len(filtered_data)}")
    
    # 条件2: 车型=CM2
    filtered_data = filtered_data[filtered_data['车型分组'] == 'CM2']
    print(f"✅ 车型=CM2的订单数: {len(filtered_data)}")
    
    # 选择需要的列
    required_columns = ['Order Number', 'Lock_Time', 'Buyer Identity No', '车型分组', 'Invoice_Upload_Time']
    
    # 检查列是否存在
    available_columns = []
    for col in required_columns:
        if col in filtered_data.columns:
            available_columns.append(col)
        else:
            print(f"⚠️  列 '{col}' 不存在于数据中")
    
    if available_columns:
        # 提取订单清单
        order_list = filtered_data[available_columns].copy()
        
        # 按Lock_Time排序
        order_list = order_list.sort_values('Lock_Time')
        
        print(f"\n📊 符合条件的订单清单（共{len(order_list)}条）:")
        print("=" * 100)
        
        if len(order_list) > 0:
            # 显示前20条记录
            display_count = min(20, len(order_list))
            print(f"显示前{display_count}条记录:")
            print(order_list.head(display_count).to_string(index=False))
            
            if len(order_list) > 20:
                print(f"\n... 还有{len(order_list) - 20}条记录未显示")
            
            # 保存到CSV文件
            output_file = 'cm2_specific_orders.csv'
            order_list.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n💾 完整订单清单已保存到: {output_file}")
            
            # 统计信息
            print(f"\n📈 统计信息:")
            print(f"- 总订单数: {len(order_list)}")
            print(f"- 不同买家数: {order_list['Buyer Identity No'].nunique()}")
            print(f"- 锁单时间范围: {order_list['Lock_Time'].min()} 到 {order_list['Lock_Time'].max()}")
            
            # 按日期统计
            order_list['Lock_Date'] = order_list['Lock_Time'].dt.date
            daily_counts = order_list['Lock_Date'].value_counts().sort_index()
            print(f"\n📅 按日期统计:")
            for date, count in daily_counts.items():
                print(f"  {date}: {count}条订单")
                
        else:
            print("❌ 没有找到符合条件的订单")
            
        return order_list
    else:
        print("❌ 没有找到所需的列")
        return pd.DataFrame()

def get_repeat_buyer_orders_list(reference_date="2025-09-10", lock_start_date="2025-09-10", lock_end_date="2025-10-14", vehicle_type="CM2"):
    """获取复购用户的订单清单
    
    Args:
        reference_date: 参考日期，用于判断复购用户（Invoice_Upload_Time需要早于此日期）
        lock_start_date: 锁单开始日期
        lock_end_date: 锁单结束日期
        vehicle_type: 车型筛选条件
    
    Returns:
        DataFrame: 复购用户的所有订单清单
    """
    print("🔍 开始获取复购用户的订单清单...")
    
    # 初始化分析器
    analyzer = ABComparisonAnalyzer()
    
    # 获取原始数据
    df = analyzer.df.copy()
    
    print(f"📊 原始数据总量: {len(df):,} 条记录")
    print(f"📅 参考日期: {reference_date}")
    print(f"🔒 锁单时间范围: {lock_start_date} 到 {lock_end_date}")
    print(f"🚗 车型筛选: {vehicle_type}")
    
    # 确保必要的列存在
    required_columns = ['Buyer Identity No', 'Invoice_Upload_Time', 'Order Number', 'Lock_Time', '车型分组']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ 缺少必要的列: {missing_columns}")
        return pd.DataFrame()
    
    # 数据预处理
    df['Buyer Identity No'] = df['Buyer Identity No'].fillna('').astype(str).str.strip()
    df['Invoice_Upload_Time'] = pd.to_datetime(df['Invoice_Upload_Time'], errors='coerce')
    df['Lock_Time'] = pd.to_datetime(df['Lock_Time'], errors='coerce')
    
    # 过滤掉身份证号为空的记录
    df = df[df['Buyer Identity No'] != '']
    print(f"📊 过滤空身份证号后数据量: {len(df):,} 条记录")
    
    # 复购用户识别逻辑（基于现有代码的逻辑）
    print("\n🔍 识别复购用户...")
    
    reference_datetime = pd.to_datetime(reference_date)
    repeat_buyer_ids = set()
    
    # 按身份证号分组，找出有多个订单的买家
    buyer_groups = df.groupby('Buyer Identity No')
    total_buyers = len(buyer_groups)
    print(f"� 总买家数: {total_buyers:,}")
    
    for buyer_id, group in buyer_groups:
        if len(group) > 1:  # 有多个订单
            # 检查是否有Invoice_Upload_Time且早于参考日期
            invoice_times = group['Invoice_Upload_Time'].dropna()
            if len(invoice_times) > 0:
                # 检查是否有Invoice_Upload_Time早于参考日期
                early_invoices = invoice_times[invoice_times < reference_datetime]
                if len(early_invoices) > 0:
                    repeat_buyer_ids.add(buyer_id)
    
    print(f"🔄 识别出复购用户数: {len(repeat_buyer_ids):,}")
    
    if len(repeat_buyer_ids) == 0:
        print("⚠️  未找到符合条件的复购用户")
        return pd.DataFrame()
    
    # 获取所有复购用户的订单
    all_repeat_buyer_orders = df[df['Buyer Identity No'].isin(repeat_buyer_ids)].copy()
    print(f"📋 复购用户的总订单数: {len(all_repeat_buyer_orders):,}")
    
    # 应用锁单时间和车型筛选条件，找出符合条件的复购用户
    print(f"\n🔍 应用额外筛选条件...")
    
    # 筛选锁单时间范围
    lock_start_datetime = pd.to_datetime(lock_start_date)
    lock_end_datetime = pd.to_datetime(lock_end_date)
    
    filtered_orders = all_repeat_buyer_orders[
        (all_repeat_buyer_orders['Lock_Time'] >= lock_start_datetime) & 
        (all_repeat_buyer_orders['Lock_Time'] <= lock_end_datetime)
    ]
    print(f"📋 锁单时间在{lock_start_date}～{lock_end_date}的复购用户订单数: {len(filtered_orders):,}")
    
    # 筛选车型
    filtered_orders = filtered_orders[filtered_orders['车型分组'] == vehicle_type]
    print(f"📋 车型={vehicle_type}的复购用户订单数: {len(filtered_orders):,}")
    
    if len(filtered_orders) == 0:
        print("⚠️  应用筛选条件后未找到符合条件的复购用户订单")
        return pd.DataFrame()
    
    # 获取符合筛选条件的复购用户ID
    filtered_repeat_buyer_ids = filtered_orders['Buyer Identity No'].unique()
    print(f"📊 符合所有条件的复购用户数: {len(filtered_repeat_buyer_ids):,}")
    
    # 获取这些复购用户的所有订单（包括早期订单和符合条件的订单）
    result_df = all_repeat_buyer_orders[
        all_repeat_buyer_orders['Buyer Identity No'].isin(filtered_repeat_buyer_ids)
    ].copy()
    
    # 添加订单类型标识
    result_df['订单类型'] = '早期订单'
    
    # 标记符合筛选条件的订单
    condition_mask = (
        (result_df['Lock_Time'] >= lock_start_datetime) & 
        (result_df['Lock_Time'] <= lock_end_datetime) &
        (result_df['车型分组'] == vehicle_type)
    )
    result_df.loc[condition_mask, '订单类型'] = '符合条件订单'
    
    print(f"📋 包含所有相关订单的复购用户订单数: {len(result_df):,}")
    print(f"  - 符合条件订单: {len(result_df[result_df['订单类型'] == '符合条件订单']):,}")
    print(f"  - 早期订单: {len(result_df[result_df['订单类型'] == '早期订单']):,}")
    
    # 选择输出列
    output_columns = ['Buyer Identity No', 'Order Number', 'Lock_Time', '车型分组', 'Invoice_Upload_Time', '订单类型']
    result_df = result_df[output_columns].copy()
    
    # 按买家身份证号和锁单时间排序
    result_df = result_df.sort_values(['Buyer Identity No', 'Lock_Time'])
    
    print(f"\n📋 复购用户订单清单预览（前10条）:")
    print(result_df.head(10).to_string(index=False))
    
    # 保存为CSV文件
    output_file = f'repeat_buyer_orders_list_{vehicle_type}_{lock_start_date}_to_{lock_end_date}.csv'
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 完整复购用户订单清单已保存到: {output_file}")
    
    # 统计信息
    print(f"\n📈 统计信息:")
    print(f"- 符合条件的复购用户数: {len(filtered_repeat_buyer_ids):,}")
    print(f"- 复购用户总订单数（包含早期订单）: {len(result_df):,}")
    print(f"- 符合条件的订单数: {len(result_df[result_df['订单类型'] == '符合条件订单']):,}")
    print(f"- 早期订单数: {len(result_df[result_df['订单类型'] == '早期订单']):,}")
    print(f"- 平均每个复购用户总订单数: {len(result_df) / len(filtered_repeat_buyer_ids):.2f}")
    
    # 按买家统计订单数
    buyer_order_counts = result_df['Buyer Identity No'].value_counts()
    print(f"\n📊 复购用户订单数分布（包含所有订单）:")
    order_count_dist = buyer_order_counts.value_counts().sort_index()
    for order_count, buyer_count in order_count_dist.items():
        print(f"  {order_count}个订单: {buyer_count}个买家")
    
    # 按订单类型统计
    order_type_dist = result_df['订单类型'].value_counts()
    print(f"\n📋 按订单类型统计:")
    for order_type, count in order_type_dist.items():
        print(f"  {order_type}: {count}个订单")
    
    # 按车型分组统计
    vehicle_dist = result_df['车型分组'].value_counts()
    print(f"\n🚗 按车型分组统计:")
    for vehicle, count in vehicle_dist.items():
        print(f"  {vehicle}: {count}个订单")
    
    return result_df

if __name__ == "__main__":
    # 运行复购用户筛选功能测试
    test_results = test_new_repeat_buyer_feature()
    print(f"\n� 测试结果摘要: {test_results}")
    
    # 运行订单清单筛选
    order_list = get_specific_order_list()
    
    # 运行复购用户订单清单获取（锁单时间2025-09-10～2025-10-14，车型CM2）
    repeat_buyer_orders = get_repeat_buyer_orders_list(
        reference_date="2025-09-10",
        lock_start_date="2025-09-10", 
        lock_end_date="2025-10-14",
        vehicle_type="CM2"
    )