#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CM2车型重复买家深度分析
基于销售代理分析报告中的重复买家数据，进行更详细的特征分析
"""

import pandas as pd
import json
from datetime import datetime
from collections import Counter
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """加载订单数据"""
    data_path = "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"
    df = pd.read_parquet(data_path)
    df['Intention_Payment_Time'] = pd.to_datetime(df['Intention_Payment_Time'])
    return df

def extract_cm2_repeat_buyers(df):
    """提取CM2车型的重复买家数据"""
    # CM2预售周期：2025-08-15 到 2025-09-10
    start_date = pd.to_datetime('2025-08-15')
    end_date = pd.to_datetime('2025-09-10')
    
    # 筛选CM2预售周期的订单
    mask = (df['Intention_Payment_Time'] >= start_date) & (df['Intention_Payment_Time'] <= end_date)
    cm2_df = df[mask].copy()
    
    logger.info(f"CM2预售周期总订单数: {len(cm2_df)}")
    
    # 统计每个Buyer Identity No的订单数量
    buyer_counts = cm2_df['Buyer Identity No'].value_counts()
    
    # 找出重复买家（订单数>=2）
    repeat_buyers = buyer_counts[buyer_counts >= 2]
    repeat_buyer_ids = repeat_buyers.index.tolist()
    
    # 提取重复买家的所有订单
    repeat_buyer_orders = cm2_df[cm2_df['Buyer Identity No'].isin(repeat_buyer_ids)].copy()
    
    logger.info(f"重复买家数量: {len(repeat_buyers)}")
    logger.info(f"重复买家订单总数: {len(repeat_buyer_orders)}")
    
    return repeat_buyer_orders, repeat_buyers

def analyze_regional_distribution(repeat_buyer_orders):
    """分析重复买家的地域分布"""
    logger.info("开始分析地域分布...")
    
    # 省份分布
    province_dist = repeat_buyer_orders['License Province'].value_counts().head(10)
    
    # 城市分布
    city_dist = repeat_buyer_orders['License City'].value_counts().head(10)
    
    # 城市等级分布
    city_level_dist = repeat_buyer_orders['license_city_level'].value_counts()
    
    # 大区分布
    region_dist = repeat_buyer_orders['Parent Region Name'].value_counts()
    
    return {
        'province_distribution': province_dist,
        'city_distribution': city_dist,
        'city_level_distribution': city_level_dist,
        'region_distribution': region_dist
    }

def analyze_channel_preference(repeat_buyer_orders):
    """分析重复买家的渠道偏好"""
    logger.info("开始分析渠道偏好...")
    
    # 中渠道分布（数据中只有这个字段）
    middle_channel_dist = repeat_buyer_orders['first_middle_channel_name'].value_counts()
    
    return {
        'middle_channel_distribution': middle_channel_dist
    }

def analyze_gender_distribution(repeat_buyer_orders):
    """分析重复买家的性别分布"""
    logger.info("开始分析性别分布...")
    
    gender_dist = repeat_buyer_orders['order_gender'].value_counts()
    
    return {'gender_distribution': gender_dist}

def analyze_purchase_patterns(repeat_buyer_orders, repeat_buyers):
    """分析重复买家的购买行为模式"""
    logger.info("开始分析购买行为模式...")
    
    # 购买频次分布
    purchase_frequency = repeat_buyers.value_counts().sort_index()
    
    # 时间间隔分析（对于有多个订单的买家）
    time_intervals = []
    for buyer_id in repeat_buyers.index:
        buyer_orders = repeat_buyer_orders[repeat_buyer_orders['Buyer Identity No'] == buyer_id]
        if len(buyer_orders) >= 2:
            times = sorted(buyer_orders['Intention_Payment_Time'])
            for i in range(1, len(times)):
                interval = (times[i] - times[i-1]).days
                time_intervals.append(interval)
    
    # 车型组合分析
    vehicle_combinations = []
    for buyer_id in repeat_buyers.index:
        buyer_orders = repeat_buyer_orders[repeat_buyer_orders['Buyer Identity No'] == buyer_id]
        vehicle_types = sorted(buyer_orders['车型分组'].unique())
        if len(vehicle_types) > 1:
            vehicle_combinations.append(tuple(vehicle_types))
    
    return {
        'purchase_frequency_distribution': purchase_frequency,
        'time_intervals': time_intervals,
        'vehicle_combinations': Counter(vehicle_combinations)
    }

def analyze_sales_agent_match(repeat_buyer_orders):
    """分析重复买家中有多少满足销售代理匹配条件"""
    logger.info("开始分析重复买家与销售代理的匹配情况...")
    
    # 读取销售代理数据
    sales_info_path = "/Users/zihao_/Documents/coding/dataset/formatted/sales_info_data.json"
    try:
        with open(sales_info_path, 'r', encoding='utf-8') as f:
            sales_info_json = json.load(f)
            # 提取data部分
            sales_info_data = sales_info_json.get('data', [])
            logger.info(f"成功读取销售代理数据，共 {len(sales_info_data)} 条记录")
    except Exception as e:
        logger.warning(f"无法读取销售代理数据文件: {str(e)}")
        return {
            'total_repeat_buyer_orders': len(repeat_buyer_orders),
            'matched_orders': 0,
            'match_ratio': 0.0,
            'error': str(e)
        }
    
    # 预处理订单数据中的相关字段
    repeat_buyer_orders['clean_store_agent_name'] = repeat_buyer_orders['Store Agent Name'].fillna('').astype(str).str.strip()
    repeat_buyer_orders['clean_store_agent_id'] = repeat_buyer_orders['Store Agent Id'].fillna('').astype(str).str.strip()
    repeat_buyer_orders['clean_buyer_identity'] = repeat_buyer_orders['Buyer Identity No'].fillna('').astype(str).str.strip()
    
    # 从销售代理数据中提取销售代理信息并构建快速查找集合
    # 字段映射：Member Name -> Store Agent Name, Member Code -> Store Agent Id, Id Card -> Buyer Identity No
    sales_agents_lookup = set()  # 使用集合进行快速查找
    if isinstance(sales_info_data, list):
        for item in sales_info_data:
            if isinstance(item, dict):
                # 提取三个关键字段
                member_name = str(item.get('Member Name', '')).strip() if item.get('Member Name') else ''
                member_code = str(item.get('Member Code', '')).strip() if item.get('Member Code') else ''
                id_card = str(item.get('Id Card', '')).strip() if item.get('Id Card') else ''
                
                # 只有当三个字段都非空时才添加到查找集合
                if member_name and member_code and id_card:
                    sales_agents_lookup.add((member_name, member_code, id_card))
    
    logger.info(f"构建销售代理查找集合，共 {len(sales_agents_lookup)} 条有效记录")
    
    # 创建组合字段用于快速匹配
    agent_combos = list(zip(
        repeat_buyer_orders['clean_store_agent_name'],
        repeat_buyer_orders['clean_store_agent_id'], 
        repeat_buyer_orders['clean_buyer_identity']
    ))
    
    # 使用集合交集快速找到匹配的订单
    matched_combos = set(agent_combos) & sales_agents_lookup
    
    # 计算匹配的订单数
    matched_orders = sum(1 for combo in agent_combos if combo in matched_combos)
    
    total_orders = len(repeat_buyer_orders)
    match_ratio = matched_orders / total_orders if total_orders > 0 else 0.0
    
    logger.info(f"重复买家订单匹配结果: 总订单数 {total_orders}, 匹配订单数 {matched_orders}, 匹配比例 {match_ratio:.2%}")
    
    return {
        'total_repeat_buyer_orders': total_orders,
        'matched_orders': matched_orders,
        'match_ratio': match_ratio,
        'sales_agents_count': len(sales_agents_lookup)
    }

def generate_detailed_report(regional_analysis, channel_analysis, gender_analysis, pattern_analysis, sales_agent_analysis, repeat_buyer_orders, repeat_buyers):
    """生成详细的分析报告"""
    logger.info("生成详细分析报告...")
    
    report_lines = []
    report_lines.append("# CM2车型重复买家深度分析报告\n")
    report_lines.append(f"**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"**数据范围**: CM2车型预售周期 (2025-08-15 至 2025-09-10)\n\n")
    
    # 基础统计
    report_lines.append("## 基础统计信息\n")
    report_lines.append(f"- **重复买家数量**: {len(repeat_buyers)} 人\n")
    report_lines.append(f"- **重复买家订单总数**: {len(repeat_buyer_orders)} 单\n")
    report_lines.append(f"- **平均每人订单数**: {len(repeat_buyer_orders) / len(repeat_buyers):.2f} 单\n\n")
    
    # 地域分布分析
    report_lines.append("## 地域分布分析\n")
    report_lines.append("### 省份分布 (Top 10)\n")
    report_lines.append("| 省份 | 订单数 | 占比 |\n")
    report_lines.append("|------|--------|------|\n")
    total_orders = len(repeat_buyer_orders)
    for province, count in regional_analysis['province_distribution'].items():
        ratio = count / total_orders * 100
        report_lines.append(f"| {province} | {count} | {ratio:.2f}% |\n")
    
    report_lines.append("\n### 城市等级分布\n")
    report_lines.append("| 城市等级 | 订单数 | 占比 |\n")
    report_lines.append("|----------|--------|------|\n")
    for level, count in regional_analysis['city_level_distribution'].items():
        ratio = count / total_orders * 100
        report_lines.append(f"| {level} | {count} | {ratio:.2f}% |\n")
    
    report_lines.append("\n### 大区分布\n")
    report_lines.append("| 大区 | 订单数 | 占比 |\n")
    report_lines.append("|------|--------|------|\n")
    for region, count in regional_analysis['region_distribution'].items():
        ratio = count / total_orders * 100
        report_lines.append(f"| {region} | {count} | {ratio:.2f}% |\n")
    
    # 渠道偏好分析
    report_lines.append("\n## 渠道偏好分析\n")
    report_lines.append("### 中渠道分布\n")
    report_lines.append("| 中渠道 | 订单数 | 占比 |\n")
    report_lines.append("|--------|--------|------|\n")
    for channel, count in channel_analysis['middle_channel_distribution'].items():
        ratio = count / total_orders * 100
        report_lines.append(f"| {channel} | {count} | {ratio:.2f}% |\n")
    
    # 性别分布分析
    report_lines.append("\n## 性别分布分析\n")
    report_lines.append("| 性别 | 订单数 | 占比 |\n")
    report_lines.append("|------|--------|------|\n")
    for gender, count in gender_analysis['gender_distribution'].items():
        ratio = count / total_orders * 100
        report_lines.append(f"| {gender} | {count} | {ratio:.2f}% |\n")
    
    # 购买行为模式分析
    report_lines.append("\n## 购买行为模式分析\n")
    report_lines.append("### 购买频次分布\n")
    report_lines.append("| 订单数/人 | 买家数量 | 占比 |\n")
    report_lines.append("|-----------|----------|------|\n")
    total_buyers = len(repeat_buyers)
    for freq, count in pattern_analysis['purchase_frequency_distribution'].items():
        ratio = count / total_buyers * 100
        report_lines.append(f"| {freq} | {count} | {ratio:.2f}% |\n")
    
    # 销售代理匹配分析
    report_lines.append("\n## 销售代理匹配分析\n")
    if 'error' in sales_agent_analysis:
        report_lines.append(f"⚠️ 销售代理数据读取失败: {sales_agent_analysis['error']}\n")
    else:
        report_lines.append("### 匹配条件说明\n")
        report_lines.append("匹配条件：sales_info_data中的Member Name = Store Agent Name，且Member Code = Store Agent Id，且Id Card = Buyer Identity No\n\n")
        
        report_lines.append("### 匹配结果统计\n")
        report_lines.append("| 指标 | 数值 |\n")
        report_lines.append("|------|------|\n")
        report_lines.append(f"| 重复买家订单总数 | {sales_agent_analysis['total_repeat_buyer_orders']} |\n")
        report_lines.append(f"| 匹配的订单数 | {sales_agent_analysis['matched_orders']} |\n")
        report_lines.append(f"| 匹配比例 | {sales_agent_analysis['match_ratio']:.2%} |\n")
        report_lines.append(f"| 销售代理数据条数 | {sales_agent_analysis['sales_agents_count']} |\n")
        
        if sales_agent_analysis['matched_orders'] > 0:
            report_lines.append(f"\n✅ 发现 {sales_agent_analysis['matched_orders']} 个重复买家订单满足销售代理匹配条件\n")
        else:
            report_lines.append("\n❌ 未发现满足销售代理匹配条件的重复买家订单\n")
    
    if pattern_analysis['time_intervals']:
        avg_interval = sum(pattern_analysis['time_intervals']) / len(pattern_analysis['time_intervals'])
        report_lines.append(f"\n### 购买时间间隔\n")
        report_lines.append(f"- **平均时间间隔**: {avg_interval:.1f} 天\n")
        report_lines.append(f"- **最短间隔**: {min(pattern_analysis['time_intervals'])} 天\n")
        report_lines.append(f"- **最长间隔**: {max(pattern_analysis['time_intervals'])} 天\n")
    
    if pattern_analysis['vehicle_combinations']:
        report_lines.append("\n### 车型组合偏好\n")
        report_lines.append("| 车型组合 | 买家数量 |\n")
        report_lines.append("|----------|----------|\n")
        for combo, count in pattern_analysis['vehicle_combinations'].most_common(5):
            combo_str = ' + '.join(combo)
            report_lines.append(f"| {combo_str} | {count} |\n")
    
    # 关键洞察
    report_lines.append("\n## 关键洞察\n")
    
    # 地域洞察
    top_province = regional_analysis['province_distribution'].index[0]
    top_province_ratio = regional_analysis['province_distribution'].iloc[0] / total_orders * 100
    report_lines.append(f"1. **地域集中度**: {top_province}省是重复买家最集中的地区，占比{top_province_ratio:.1f}%\n")
    
    # 渠道洞察
    top_middle_channel = channel_analysis['middle_channel_distribution'].index[0]
    top_channel_ratio = channel_analysis['middle_channel_distribution'].iloc[0] / total_orders * 100
    report_lines.append(f"2. **渠道偏好**: {top_middle_channel}是重复买家的主要中渠道，占比{top_channel_ratio:.1f}%\n")
    
    # 购买行为洞察
    most_common_freq = pattern_analysis['purchase_frequency_distribution'].index[0]
    freq_ratio = pattern_analysis['purchase_frequency_distribution'].iloc[0] / total_buyers * 100
    report_lines.append(f"3. **购买频次**: {freq_ratio:.1f}%的重复买家购买了{most_common_freq}次\n")
    
    # 销售代理匹配洞察
    if 'error' not in sales_agent_analysis:
        match_ratio = sales_agent_analysis['match_ratio'] * 100
        if sales_agent_analysis['matched_orders'] > 0:
            report_lines.append(f"4. **销售代理匹配**: {match_ratio:.1f}%的重复买家订单满足销售代理匹配条件，共{sales_agent_analysis['matched_orders']}个订单\n")
        else:
            report_lines.append(f"4. **销售代理匹配**: 未发现满足销售代理匹配条件的重复买家订单\n")
    
    report_lines.append("\n---\n")
    report_lines.append("*本报告基于CM2车型预售周期数据生成*\n")
    
    # 保存报告
    report_content = ''.join(report_lines)
    with open('/Users/zihao_/Documents/github/W35_workflow/cm2_repeat_buyer_detailed_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info("详细分析报告已生成: cm2_repeat_buyer_detailed_report.md")
    return report_content

def main():
    """主函数"""
    try:
        # 加载数据
        df = load_data()
        
        # 提取CM2重复买家数据
        repeat_buyer_orders, repeat_buyers = extract_cm2_repeat_buyers(df)
        
        # 各维度分析
        regional_analysis = analyze_regional_distribution(repeat_buyer_orders)
        channel_analysis = analyze_channel_preference(repeat_buyer_orders)
        gender_analysis = analyze_gender_distribution(repeat_buyer_orders)
        pattern_analysis = analyze_purchase_patterns(repeat_buyer_orders, repeat_buyers)
        sales_agent_analysis = analyze_sales_agent_match(repeat_buyer_orders)
        
        # 生成详细报告
        report = generate_detailed_report(
            regional_analysis, channel_analysis, gender_analysis, 
            pattern_analysis, sales_agent_analysis, repeat_buyer_orders, repeat_buyers
        )
        
        print("\n=== CM2重复买家分析完成 ===")
        print(f"重复买家数量: {len(repeat_buyers)}")
        print(f"重复买家订单数: {len(repeat_buyer_orders)}")
        
        # 显示销售代理匹配结果
        if 'error' not in sales_agent_analysis:
            print(f"销售代理匹配订单数: {sales_agent_analysis['matched_orders']}")
            print(f"销售代理匹配比例: {sales_agent_analysis['match_ratio']:.2%}")
        else:
            print(f"销售代理匹配分析失败: {sales_agent_analysis['error']}")
        
        print("详细报告已生成: cm2_repeat_buyer_detailed_report.md")
        
    except Exception as e:
        logger.error(f"分析过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()