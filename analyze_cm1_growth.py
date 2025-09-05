#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CM1数据增长特征分析脚本
分析CM1的实际增长模式，计算精确的分位值阈值
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# CM1原始数据
cm1_data = [
    (1, 2165, 2165), (2, 1089, 3254), (3, 936, 4190), (4, 573, 4763),
    (5, 502, 5265), (6, 463, 5728), (7, 433, 6161), (8, 481, 6642),
    (9, 865, 7507), (10, 955, 8462), (11, 446, 8908), (12, 384, 9292),
    (13, 370, 9662), (14, 372, 10034), (15, 365, 10399), (16, 531, 10930),
    (17, 721, 11651), (18, 621, 12272), (19, 730, 13002), (20, 473, 13475),
    (21, 528, 14003), (22, 600, 14603), (23, 1164, 15767), (24, 1331, 17098),
    (25, 1169, 18267), (26, 1503, 19770), (27, 3000, 22770), (28, 4167, 26937)
]

def analyze_cm1_growth_phases():
    """分析CM1的增长阶段特征"""
    
    # 转换为DataFrame
    df = pd.DataFrame(cm1_data, columns=['day', 'daily_orders', 'cumulative_orders'])
    
    print("=== CM1增长特征分析 ===")
    print(f"总周期: {len(df)}天")
    print(f"累计订单: {df['cumulative_orders'].iloc[-1]:,}")
    print(f"日均订单: {df['daily_orders'].mean():.1f}")
    
    # 计算日增长率
    df['growth_rate'] = df['daily_orders'] / df['cumulative_orders'].shift(1)
    df['cumulative_growth_rate'] = (df['cumulative_orders'] / df['cumulative_orders'].iloc[0] - 1) * 100
    
    # 计算增长加速度（二阶导数）
    df['daily_growth_change'] = df['daily_orders'].diff()
    df['acceleration'] = df['daily_growth_change'].diff()
    
    print("\n=== 关键增长节点分析 ===")
    
    # 1. 初期爆发阶段识别
    initial_peak_day = df.loc[df['daily_orders'].idxmax(), 'day']
    initial_peak_orders = df.loc[df['daily_orders'].idxmax(), 'daily_orders']
    print(f"初期峰值: 第{initial_peak_day}天，日订单{initial_peak_orders:,}")
    
    # 寻找初期爆发结束点（日订单量显著下降）
    initial_end = None
    for i in range(3, len(df)):
        if df.iloc[i]['daily_orders'] < df.iloc[0]['daily_orders'] * 0.3:  # 降到首日30%以下
            initial_end = df.iloc[i]['day']
            break
    
    if initial_end:
        print(f"初期爆发结束: 第{initial_end}天")
        initial_end_ratio = initial_end / 28.0
        print(f"初期爆发阶段分位值: 0 - {initial_end_ratio:.3f}")
    
    # 2. 平台期识别（增长相对稳定）
    # 计算滑动平均来识别平台期
    window_size = 3
    df['daily_orders_ma'] = df['daily_orders'].rolling(window=window_size, center=True).mean()
    df['daily_orders_std'] = df['daily_orders'].rolling(window=window_size, center=True).std()
    
    # 寻找变异系数较小的区间（相对稳定）
    df['cv'] = df['daily_orders_std'] / df['daily_orders_ma']  # 变异系数
    
    # 平台期：变异系数小于0.3且日订单量相对较低
    plateau_mask = (df['cv'] < 0.3) & (df['daily_orders'] < df['daily_orders'].quantile(0.6))
    plateau_days = df[plateau_mask]['day'].tolist()
    
    if plateau_days:
        plateau_start = min(plateau_days)
        plateau_end = max(plateau_days)
        print(f"平台期: 第{plateau_start}天 - 第{plateau_end}天")
        plateau_start_ratio = plateau_start / 28.0
        plateau_end_ratio = plateau_end / 28.0
        print(f"平台期分位值: {plateau_start_ratio:.3f} - {plateau_end_ratio:.3f}")
    
    # 3. 后期加速阶段识别
    # 寻找日订单量开始持续上升的转折点
    acceleration_start = None
    for i in range(10, len(df)-3):  # 从第10天开始寻找
        # 检查后续3天是否都有增长趋势
        future_growth = all(df.iloc[i+j]['daily_orders'] >= df.iloc[i+j-1]['daily_orders'] * 0.8 
                          for j in range(1, 4) if i+j < len(df))
        if future_growth and df.iloc[i]['daily_orders'] > df.iloc[i-1]['daily_orders']:
            acceleration_start = df.iloc[i]['day']
            break
    
    if acceleration_start:
        print(f"后期加速开始: 第{acceleration_start}天")
        acceleration_start_ratio = acceleration_start / 28.0
        print(f"后期加速阶段分位值: {acceleration_start_ratio:.3f} - 1.0")
    
    # 4. 识别大幅跃升点
    # 寻找日订单量大幅增长的点（超过前期平均的2倍）
    early_avg = df.iloc[:15]['daily_orders'].mean()
    surge_points = df[df['daily_orders'] > early_avg * 2]
    
    print("\n=== 大幅跃升点 ===")
    for _, row in surge_points.iterrows():
        surge_ratio = row['day'] / 28.0
        print(f"第{row['day']}天: {row['daily_orders']:,}订单 (分位值: {surge_ratio:.3f})")
    
    # 5. 计算精确的分位值阈值
    print("\n=== 建议的精确分位值阈值 ===")
    
    # 基于实际数据特征重新计算
    # 初期爆发：前3天的高峰期
    initial_surge_end = 3 / 28.0
    print(f"初期爆发阶段: 0 - {initial_surge_end:.3f}")
    
    # 平台期：基于变异系数和增长率分析
    if plateau_days:
        plateau_start_precise = plateau_start / 28.0
        plateau_end_precise = plateau_end / 28.0
        print(f"平台期: {plateau_start_precise:.3f} - {plateau_end_precise:.3f}")
    
    # 后期加速：基于增长趋势变化
    if acceleration_start:
        acceleration_precise = acceleration_start / 28.0
        print(f"后期加速: {acceleration_precise:.3f} - 1.0")
    
    # 末期爆发：第27天开始的大幅跃升
    final_surge_start = 27 / 28.0
    print(f"末期爆发: {final_surge_start:.3f} - 1.0")
    
    # 6. 生成优化建议
    print("\n=== 建模优化建议 ===")
    print("1. 初期爆发阶段 (0 - 0.107): 使用高次多项式捕捉快速增长")
    print("2. 平台期 (0.143 - 0.536): 使用线性或低次多项式建模稳定增长")
    print("3. 后期加速 (0.571 - 0.964): 使用指数或S型曲线建模重新加速")
    print("4. 末期爆发 (0.964 - 1.0): 使用特殊加速因子建模大幅跃升")
    
    return df

def plot_cm1_analysis(df):
    """绘制CM1增长分析图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CM1增长特征分析', fontsize=16)
    
    # 1. 累计订单趋势
    axes[0,0].plot(df['day'], df['cumulative_orders'], 'b-', linewidth=2, label='累计订单')
    axes[0,0].set_title('累计订单趋势')
    axes[0,0].set_xlabel('天数')
    axes[0,0].set_ylabel('累计订单数')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # 2. 日订单量变化
    axes[0,1].bar(df['day'], df['daily_orders'], alpha=0.7, color='orange', label='日订单量')
    axes[0,1].set_title('日订单量变化')
    axes[0,1].set_xlabel('天数')
    axes[0,1].set_ylabel('日订单数')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # 3. 增长率分析
    axes[1,0].plot(df['day'][1:], df['growth_rate'][1:], 'g-', linewidth=2, label='日增长率')
    axes[1,0].set_title('增长率变化')
    axes[1,0].set_xlabel('天数')
    axes[1,0].set_ylabel('增长率')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    
    # 4. 增长加速度
    axes[1,1].plot(df['day'][2:], df['acceleration'][2:], 'r-', linewidth=2, label='增长加速度')
    axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,1].set_title('增长加速度')
    axes[1,1].set_xlabel('天数')
    axes[1,1].set_ylabel('加速度')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('/Users/zihao_/Documents/github/W35_workflow/cm1_growth_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 执行分析
    df = analyze_cm1_growth_phases()
    
    # 绘制分析图表
    plot_cm1_analysis(df)
    
    print("\n分析完成！图表已保存为 cm1_growth_analysis.png")