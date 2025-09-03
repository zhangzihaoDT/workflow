import pandas as pd

# 读取数据
data = pd.read_parquet('/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet')

# 筛选CM2数据
cm2_data = data[data['车型分组'] == 'CM2']

# 筛选退订数据
cm2_refund_data = cm2_data[cm2_data['intention_refund_time'].notna()].copy()

print(f'CM2总订单数: {len(cm2_data)}')
print(f'CM2退订数: {len(cm2_refund_data)}')

if len(cm2_refund_data) > 0:
    # 获取当日日期（与订单数据保持一致，使用订单数据的最新日期）
    cm2_order_data_copy = cm2_data.copy()
    cm2_order_data_copy['date'] = cm2_order_data_copy['Intention_Payment_Time'].dt.date
    latest_date = cm2_order_data_copy['date'].max()
    
    # 筛选当日退订数据
    daily_refund = cm2_refund_data[cm2_refund_data['intention_refund_time'].dt.date == latest_date]
    
    print(f'最新退订日期: {latest_date}')
    print(f'当日退订数: {len(daily_refund)}')
    print(f'累计退订数: {len(cm2_refund_data)}')
    
    # 检查地区分布（日环比增速分析）
    if len(daily_refund) > 0:
        print('\n=== 地区当日退订日环比分析 ===')
        
        # 获取前一日退订数据
        previous_date = latest_date - pd.Timedelta(days=1)
        previous_refund = cm2_refund_data[cm2_refund_data['intention_refund_time'].dt.date == previous_date]
        
        # 计算当日和前一日各地区退订数量
        daily_region_counts = daily_refund['License City'].value_counts()
        previous_region_counts = previous_refund['License City'].value_counts() if len(previous_refund) > 0 else pd.Series()
        
        print(f'当日退订数: {len(daily_refund)}单')
        print(f'前日退订数: {len(previous_refund)}单')
        
        print('\n当日退订地区分布:')
        for region, count in daily_region_counts.head().items():
            print(f'  {region}: {count}单')
            
        print('\n前日退订地区分布:')
        for region, count in previous_region_counts.head().items():
            print(f'  {region}: {count}单')
            
        # 检查日环比增速异常
        print('\n=== 日环比异常检查 ===')
        all_regions = set(daily_region_counts.index) | set(previous_region_counts.index)
        
        for region in all_regions:
            daily_count = daily_region_counts.get(region, 0)
            previous_count = previous_region_counts.get(region, 0)
            
            # 计算日环比增速
            if previous_count > 0 and daily_count >= 1:  # 当日至少1单退订
                change_rate = (daily_count - previous_count) / previous_count
                if abs(change_rate) > 0.10:  # 10%变化幅度阈值
                    change_direction = "增长" if change_rate > 0 else "下降"
                    print(f'异常: {region} - 当日{daily_count}单, 前日{previous_count}单, 日环比增速{change_rate:.1%} ({change_direction})')
            elif daily_count >= 2 and previous_count == 0:  # 新出现的地区，当日至少2单
                print(f'新增: {region} - 当日{daily_count}单, 前日0单')
else:
    print('没有退订数据')