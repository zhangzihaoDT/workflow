#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CM车型预售分析脚本
分析CM0、CM1、CM2车型的预售累计小订数、小订留存锁单数和小订转化率

参考 order_trend_monitor.py 的计算逻辑
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据路径配置
DATA_PATH = "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"
BUSINESS_DEF_PATH = "/Users/zihao_/Documents/github/W35_workflow/business_definition.json"

class CMVehiclePresaleAnalyzer:
    """CM车型预售分析器"""
    
    def __init__(self):
        self.df = None
        self.business_definition = None
        self.analysis_results = {}
        
    def load_data(self):
        """加载数据"""
        try:
            logger.info("加载数据文件...")
            self.df = pd.read_parquet(DATA_PATH)
            logger.info(f"数据加载成功，共{len(self.df)}条记录")
            
            # 转换时间列
            time_columns = ['Intention_Payment_Time', 'Lock_Time', 'intention_refund_time', 'first_assign_time']
            for col in time_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
            
    def load_business_definition(self):
        """加载业务定义"""
        try:
            logger.info("加载业务定义文件...")
            with open(BUSINESS_DEF_PATH, 'r', encoding='utf-8') as f:
                self.business_definition = json.load(f)
            logger.info("业务定义加载成功")
        except Exception as e:
            logger.error(f"业务定义加载失败: {e}")
            raise
            
    def get_vehicle_period(self, vehicle: str) -> Dict[str, str]:
        """获取车型的预售周期"""
        if not self.business_definition or 'time_periods' not in self.business_definition:
            return None
        return self.business_definition['time_periods'].get(vehicle)
        
    def calculate_days_from_start(self, vehicle: str, date: datetime) -> int:
        """计算从预售开始的天数"""
        period = self.get_vehicle_period(vehicle)
        if not period:
            return 0
        start_date = pd.to_datetime(period['start'])
        return (date - start_date).days + 1
        
    def analyze_time_interval_patterns(self, vehicle: str) -> Dict[str, Any]:
        """分析Lock_Time和Intention_Payment_Time的时间差模式"""
        logger.info(f"分析{vehicle}车型的时间间隔模式...")
        
        # 获取车型数据
        vehicle_data = self.df[self.df['车型分组'] == vehicle].copy()
        if len(vehicle_data) == 0:
            logger.warning(f"{vehicle}车型无数据")
            return None
            
        # 获取预售周期
        period = self.get_vehicle_period(vehicle)
        if not period:
            logger.warning(f"{vehicle}车型无预售周期定义")
            return None
            
        start_date = pd.to_datetime(period['start'])
        end_date = pd.to_datetime(period['end'])
        
        # 筛选预售期间的数据
        presale_data = vehicle_data[
            (vehicle_data['Intention_Payment_Time'] >= start_date) & 
            (vehicle_data['Intention_Payment_Time'] <= end_date)
        ].copy()
        
        if len(presale_data) == 0:
            logger.warning(f"{vehicle}车型预售期间无数据")
            return None
            
        # 筛选有锁单时间的订单
        lock_cutoff_date = end_date + timedelta(days=1)
        locked_orders = presale_data[
            (presale_data['Lock_Time'].notna()) & 
            (presale_data['Intention_Payment_Time'].notna()) & 
            (presale_data['Lock_Time'] < pd.Timestamp(lock_cutoff_date))
        ].copy()
        
        if len(locked_orders) == 0:
            logger.warning(f"{vehicle}车型无有效锁单数据")
            return {
                'vehicle': vehicle,
                'total_presale_orders': len(presale_data),
                'locked_orders': 0,
                'time_intervals': [],
                'interval_stats': {}
            }
            
        # 计算时间差（天数）
        locked_orders['time_interval_days'] = (
            locked_orders['Lock_Time'] - locked_orders['Intention_Payment_Time']
        ).dt.days
        
        # 过滤负值和异常值（超过365天的）
        locked_orders = locked_orders[
            (locked_orders['time_interval_days'] >= 0) & 
            (locked_orders['time_interval_days'] <= 365)
        ]
        
        if len(locked_orders) == 0:
            logger.warning(f"{vehicle}车型无有效时间间隔数据")
            return {
                'vehicle': vehicle,
                'total_presale_orders': len(presale_data),
                'locked_orders': 0,
                'time_intervals': [],
                'interval_stats': {}
            }
            
        # 按时间间隔分组统计
        interval_groups = [
            (0, 0, '当天'),
            (1, 3, '1-3天'),
            (4, 7, '4-7天'),
            (8, 14, '8-14天'),
            (15, 30, '15-30天'),
            (31, 60, '31-60天'),
            (61, 365, '61天以上')
        ]
        
        interval_stats = {}
        total_presale = len(presale_data)
        total_locked = len(locked_orders)
        
        for min_days, max_days, label in interval_groups:
            if min_days == max_days:
                interval_data = locked_orders[locked_orders['time_interval_days'] == min_days]
            else:
                interval_data = locked_orders[
                    (locked_orders['time_interval_days'] >= min_days) & 
                    (locked_orders['time_interval_days'] <= max_days)
                ]
            
            count = len(interval_data)
            percentage_of_locked = (count / total_locked * 100) if total_locked > 0 else 0
            percentage_of_total = (count / total_presale * 100) if total_presale > 0 else 0
            
            interval_stats[label] = {
                'min_days': min_days,
                'max_days': max_days,
                'count': count,
                'percentage_of_locked': round(percentage_of_locked, 2),
                'percentage_of_total': round(percentage_of_total, 2)
            }
        
        # 统计信息
        time_intervals = locked_orders['time_interval_days'].tolist()
        time_intervals.sort(reverse=True)  # 降序排序
        
        stats = {
            'mean': round(locked_orders['time_interval_days'].mean(), 2),
            'median': round(locked_orders['time_interval_days'].median(), 2),
            'std': round(locked_orders['time_interval_days'].std(), 2),
            'min': int(locked_orders['time_interval_days'].min()),
            'max': int(locked_orders['time_interval_days'].max()),
            'q25': round(locked_orders['time_interval_days'].quantile(0.25), 2),
            'q75': round(locked_orders['time_interval_days'].quantile(0.75), 2)
        }
        
        result = {
            'vehicle': vehicle,
            'total_presale_orders': total_presale,
            'locked_orders': total_locked,
            'overall_lock_rate': round(total_locked / total_presale * 100, 2),
            'time_intervals': time_intervals[:100],  # 只保留前100个最大值
            'interval_stats': interval_stats,
            'descriptive_stats': stats
        }
        
        logger.info(f"{vehicle}车型时间间隔分析完成: 锁单{total_locked}单, 平均间隔{stats['mean']:.1f}天")
        return result
        
    def analyze_all_vehicles_time_intervals(self) -> Dict[str, Any]:
        """分析所有车型的时间间隔模式并进行对比"""
        logger.info("开始分析所有车型的时间间隔模式...")
        
        vehicles = ['CM0', 'CM1', 'CM2']
        all_results = {}
        
        # 分析每个车型
        for vehicle in vehicles:
            result = self.analyze_time_interval_patterns(vehicle)
            if result:
                all_results[vehicle] = result
        
        if not all_results:
            logger.warning("无有效的时间间隔分析结果")
            return {}
            
        # 生成对比分析
        comparison = {
            'vehicles_analyzed': list(all_results.keys()),
            'individual_results': all_results,
            'comparison_summary': {}
        }
        
        # 对比统计
        summary = {}
        for vehicle, data in all_results.items():
            summary[vehicle] = {
                'total_presale_orders': data['total_presale_orders'],
                'locked_orders': data['locked_orders'],
                'overall_lock_rate': data['overall_lock_rate'],
                'avg_interval_days': data['descriptive_stats']['mean'],
                'median_interval_days': data['descriptive_stats']['median'],
                'max_interval_days': data['descriptive_stats']['max'],
                'min_interval_days': data['descriptive_stats']['min']
            }
        
        comparison['comparison_summary'] = summary
        
        # 按不同时间间隔的锁单率对比
        interval_comparison = {}
        interval_labels = ['当天', '1-3天', '4-7天', '8-14天', '15-30天', '31-60天', '61天以上']
        
        for label in interval_labels:
            interval_comparison[label] = {}
            for vehicle, data in all_results.items():
                if label in data['interval_stats']:
                    interval_comparison[label][vehicle] = {
                        'count': data['interval_stats'][label]['count'],
                        'percentage_of_locked': data['interval_stats'][label]['percentage_of_locked'],
                        'percentage_of_total': data['interval_stats'][label]['percentage_of_total']
                    }
                else:
                    interval_comparison[label][vehicle] = {
                        'count': 0,
                        'percentage_of_locked': 0,
                        'percentage_of_total': 0
                    }
        
        comparison['interval_comparison'] = interval_comparison
        
        # 排序分析：按平均时间间隔排序
        sorted_by_avg_interval = sorted(
            summary.items(), 
            key=lambda x: x[1]['avg_interval_days'], 
            reverse=True
        )
        
        # 排序分析：按锁单率排序
        sorted_by_lock_rate = sorted(
            summary.items(), 
            key=lambda x: x[1]['overall_lock_rate'], 
            reverse=True
        )
        
        comparison['rankings'] = {
            'by_avg_interval_desc': [item[0] for item in sorted_by_avg_interval],
            'by_lock_rate_desc': [item[0] for item in sorted_by_lock_rate]
        }
        
        logger.info(f"时间间隔对比分析完成，分析了{len(all_results)}个车型")
        return comparison
        
    def analyze_vehicle_presale(self, vehicle: str) -> Dict[str, Any]:
        """分析单个车型的预售数据"""
        logger.info(f"分析{vehicle}车型预售数据...")
        
        # 获取车型数据
        vehicle_data = self.df[self.df['车型分组'] == vehicle].copy()
        if len(vehicle_data) == 0:
            logger.warning(f"{vehicle}车型无数据")
            return None
            
        # 获取预售周期
        period = self.get_vehicle_period(vehicle)
        if not period:
            logger.warning(f"{vehicle}车型无预售周期定义")
            return None
            
        start_date = pd.to_datetime(period['start'])
        end_date = pd.to_datetime(period['end'])
        
        # 筛选预售期间的数据
        presale_data = vehicle_data[
            (vehicle_data['Intention_Payment_Time'] >= start_date) & 
            (vehicle_data['Intention_Payment_Time'] <= end_date)
        ].copy()
        
        if len(presale_data) == 0:
            logger.warning(f"{vehicle}车型预售期间无数据")
            return None
            
        # 计算预售累计小订数
        total_presale_orders = len(presale_data)
        
        # 计算小订留存锁单数：同时含有Lock_Time、Intention_Payment_Time，且Lock_Time < 预售结束日期+30日
        # 参考order_trend_monitor.py的计算逻辑
        lock_cutoff_date = end_date + timedelta(days=1)
        lock_orders = presale_data[
            (presale_data['Lock_Time'].notna()) & 
            (presale_data['Intention_Payment_Time'].notna()) & 
            (presale_data['Lock_Time'] < pd.Timestamp(lock_cutoff_date))
        ]
        retained_locks = len(lock_orders)
        
        # 计算小订转化率
        conversion_rate = (retained_locks / total_presale_orders * 100) if total_presale_orders > 0 else 0
        
        # 按日统计预售订单数
        presale_data['date'] = presale_data['Intention_Payment_Time'].dt.date
        daily_orders = presale_data.groupby('date').size().reset_index(name='orders')
        daily_orders['date'] = pd.to_datetime(daily_orders['date'])
        daily_orders = daily_orders.sort_values('date')
        daily_orders['cumulative'] = daily_orders['orders'].cumsum()
        daily_orders['days_from_start'] = daily_orders['date'].apply(
            lambda x: self.calculate_days_from_start(vehicle, x.to_pydatetime())
        )
        
        # 按日统计锁单数（使用正确的锁单条件）
        daily_locks = []
        for _, row in daily_orders.iterrows():
            day_orders = presale_data[presale_data['date'] == row['date'].date()]
            # 使用与总体计算一致的锁单条件
            day_locks = day_orders[
                (day_orders['Lock_Time'].notna()) & 
                (day_orders['Intention_Payment_Time'].notna()) & 
                (day_orders['Lock_Time'] < pd.Timestamp(lock_cutoff_date))
            ]
            daily_locks.append({
                'date': row['date'],
                'days_from_start': row['days_from_start'],
                'orders': row['orders'],
                'cumulative_orders': row['cumulative'],
                'locks': len(day_locks),
                'conversion_rate': (len(day_locks) / row['orders'] * 100) if row['orders'] > 0 else 0
            })
        
        daily_locks_df = pd.DataFrame(daily_locks)
        
        # 计算预售周期长度
        presale_days = (end_date - start_date).days + 1
        
        result = {
            'vehicle': vehicle,
            'period': period,
            'presale_days': presale_days,
            'total_presale_orders': total_presale_orders,
            'retained_locks': retained_locks,
            'conversion_rate': round(conversion_rate, 2),
            'daily_data': daily_locks_df.to_dict('records'),
            'summary': {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'avg_daily_orders': round(total_presale_orders / presale_days, 1),
                'peak_daily_orders': daily_orders['orders'].max() if not daily_orders.empty else 0,
                'peak_date': daily_orders.loc[daily_orders['orders'].idxmax(), 'date'].strftime('%Y-%m-%d') if not daily_orders.empty else None
            }
        }
        
        logger.info(f"{vehicle}车型分析完成: 预售{total_presale_orders}单, 留存锁单{retained_locks}单, 转化率{conversion_rate:.2f}%")
        return result
        
    def analyze_all_cm_vehicles(self) -> Dict[str, Any]:
        """分析所有CM车型"""
        logger.info("开始分析所有CM车型...")
        
        cm_vehicles = ['CM0', 'CM1', 'CM2']
        results = {}
        
        for vehicle in cm_vehicles:
            result = self.analyze_vehicle_presale(vehicle)
            if result:
                results[vehicle] = result
        
        # 添加时间间隔分析
        logger.info("开始时间间隔分析...")
        time_interval_analysis = self.analyze_all_vehicles_time_intervals()
        
        # 将时间间隔分析结果添加到总结果中
        final_results = {
            'vehicle_analysis': results,
            'time_interval_analysis': time_interval_analysis
        }
                 
        self.analysis_results = final_results
        return final_results
        
    def generate_comparison_table(self) -> pd.DataFrame:
        """生成对比表格"""
        if not self.analysis_results or 'vehicle_analysis' not in self.analysis_results:
            return pd.DataFrame()
            
        vehicle_results = self.analysis_results['vehicle_analysis']
        comparison_data = []
        for vehicle, data in vehicle_results.items():
            comparison_data.append({
                '车型分组': vehicle,
                '预售周期': f"{data['summary']['start_date']} 至 {data['summary']['end_date']}",
                '预售天数': data['presale_days'],
                '预售累计小订数': data['total_presale_orders'],
                '小订留存锁单数': data['retained_locks'],
                '小订转化率(%)': data['conversion_rate'],
                '日均订单数': data['summary']['avg_daily_orders'],
                '峰值日订单数': data['summary']['peak_daily_orders'],
                '峰值日期': data['summary']['peak_date']
            })
            
        return pd.DataFrame(comparison_data)
        
    def generate_markdown_report(self) -> str:
        """生成Markdown格式的分析报告"""
        if not self.analysis_results:
            return "# CM车型预售分析报告\n\n**错误**: 无分析结果数据"
            
        report = []
        report.append("# CM车型预售分析报告")
        report.append("")
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 概览
        report.append("## 📊 分析概览")
        report.append("")
        report.append("本报告分析CM0、CM1、CM2三个车型的预售表现，重点关注以下指标：")
        report.append("- **预售累计小订数**: 预售期间的总订单数")
        report.append("- **小订留存锁单数**: 预售订单中最终锁单的数量")
        report.append("- **小订转化率**: 留存锁单数占预售订单数的比例")
        report.append("")
        
        # 对比表格
        comparison_df = self.generate_comparison_table()
        if not comparison_df.empty:
            report.append("## 📈 车型对比表")
            report.append("")
            report.append(comparison_df.to_markdown(index=False))
            report.append("")
            
        # 详细分析
        report.append("## 🔍 详细分析")
        report.append("")
        
        vehicle_results = self.analysis_results.get('vehicle_analysis', {})
        for vehicle in ['CM0', 'CM1', 'CM2']:
            if vehicle in vehicle_results:
                data = vehicle_results[vehicle]
                report.append(f"### {vehicle}车型")
                report.append("")
                report.append(f"**预售周期**: {data['summary']['start_date']} 至 {data['summary']['end_date']} ({data['presale_days']}天)")
                report.append("")
                report.append("**核心指标**:")
                report.append(f"- 预售累计小订数: **{data['total_presale_orders']:,}单**")
                report.append(f"- 小订留存锁单数: **{data['retained_locks']:,}单**")
                report.append(f"- 小订转化率: **{data['conversion_rate']:.2f}%**")
                report.append("")
                report.append("**表现特征**:")
                report.append(f"- 日均订单数: {data['summary']['avg_daily_orders']}单")
                report.append(f"- 峰值日订单数: {data['summary']['peak_daily_orders']}单")
                report.append(f"- 峰值日期: {data['summary']['peak_date']}")
                report.append("")
                
                # 每日数据表格（仅显示前10天和后5天）
                daily_data = data['daily_data']
                if daily_data:
                    report.append("**每日订单数据** (前10天):")
                    report.append("")
                    report.append("| 日期 | 第N日 | 当日订单 | 累计订单 | 当日锁单 | 当日转化率(%) |")
                    report.append("|------|-------|----------|----------|----------|---------------|")
                    
                    # 显示前10天
                    for i, day_data in enumerate(daily_data[:10]):
                        date_str = pd.to_datetime(day_data['date']).strftime('%m-%d')
                        report.append(f"| {date_str} | 第{day_data['days_from_start']}日 | {day_data['orders']} | {day_data['cumulative_orders']} | {day_data['locks']} | {day_data['conversion_rate']:.1f}% |")
                    
                    if len(daily_data) > 10:
                        report.append("| ... | ... | ... | ... | ... | ... |")
                        
                    report.append("")
            else:
                report.append(f"### {vehicle}车型")
                report.append("")
                report.append("**⚠️ 无数据或分析失败**")
                report.append("")
                
        # 时间间隔分析
        if 'time_interval_analysis' in self.analysis_results:
            time_analysis = self.analysis_results['time_interval_analysis']
            report.append("## ⏱️ 时间间隔分析")
            report.append("")
            report.append("分析预售订单从支付意向金到锁单的时间间隔模式：")
            report.append("")
            
            if 'comparison_summary' in time_analysis:
                summary = time_analysis['comparison_summary']
                report.append("### 时间间隔对比")
                report.append("")
                report.append("| 车型 | 平均间隔(天) | 中位数间隔(天) | 最大间隔(天) | 锁单率(%) |")
                report.append("|------|-------------|---------------|-------------|-----------|")
                
                for vehicle in ['CM0', 'CM1', 'CM2']:
                    if vehicle in summary:
                        data = summary[vehicle]
                        report.append(f"| {vehicle} | {data['avg_interval_days']:.1f} | {data['median_interval_days']:.1f} | {data['max_interval_days']} | {data['overall_lock_rate']:.1f}% |")
                
                report.append("")
            
            if 'interval_comparison' in time_analysis:
                interval_comp = time_analysis['interval_comparison']
                report.append("### 不同时间段锁单分布")
                report.append("")
                report.append("| 时间段 | CM0锁单数 | CM1锁单数 | CM2锁单数 |")
                report.append("|--------|----------|----------|----------|")
                
                for interval_label in ['当天', '1-3天', '4-7天', '8-14天', '15-30天', '31-60天', '61天以上']:
                    if interval_label in interval_comp:
                        data = interval_comp[interval_label]
                        cm0_count = data.get('CM0', {}).get('count', 0)
                        cm1_count = data.get('CM1', {}).get('count', 0)
                        cm2_count = data.get('CM2', {}).get('count', 0)
                        report.append(f"| {interval_label} | {cm0_count} | {cm1_count} | {cm2_count} |")
                
                report.append("")
        
        # 关键洞察
        report.append("## 💡 关键洞察")
        report.append("")
        
        vehicle_results = self.analysis_results.get('vehicle_analysis', {})
        if len(vehicle_results) >= 2:
            # 转化率对比
            conversion_rates = {v: data['conversion_rate'] for v, data in vehicle_results.items()}
            best_vehicle = max(conversion_rates.keys(), key=lambda x: conversion_rates[x])
            worst_vehicle = min(conversion_rates.keys(), key=lambda x: conversion_rates[x])
            
            report.append("**转化率表现**:")
            report.append(f"- 最高转化率: **{best_vehicle}** ({conversion_rates[best_vehicle]:.2f}%)")
            report.append(f"- 最低转化率: **{worst_vehicle}** ({conversion_rates[worst_vehicle]:.2f}%)")
            report.append("")
            
            # 订单规模对比
            order_counts = {v: data['total_presale_orders'] for v, data in vehicle_results.items()}
            max_orders_vehicle = max(order_counts.keys(), key=lambda x: order_counts[x])
            
            report.append("**订单规模**:")
            report.append(f"- 最大订单量: **{max_orders_vehicle}** ({order_counts[max_orders_vehicle]:,}单)")
            report.append("")
            
            # 时间间隔洞察
            if 'time_interval_analysis' in self.analysis_results:
                time_analysis = self.analysis_results['time_interval_analysis']
                if 'comparison_summary' in time_analysis:
                    summary = time_analysis['comparison_summary']
                    avg_intervals = {v: data['avg_interval_days'] for v, data in summary.items()}
                    fastest_vehicle = min(avg_intervals.keys(), key=lambda x: avg_intervals[x])
                    slowest_vehicle = max(avg_intervals.keys(), key=lambda x: avg_intervals[x])
                    
                    report.append("**时间间隔表现**:")
                    report.append(f"- 最快锁单: **{fastest_vehicle}** (平均{avg_intervals[fastest_vehicle]:.1f}天)")
                    report.append(f"- 最慢锁单: **{slowest_vehicle}** (平均{avg_intervals[slowest_vehicle]:.1f}天)")
                    report.append("")
            
        # 方法说明
        report.append("## 📋 计算方法说明")
        report.append("")
        report.append("**指标定义**:")
        report.append("- **预售累计小订数**: 在预售周期内（从预售开始到预售结束）支付意向金的订单总数")
        report.append("- **小订留存锁单数**: 预售订单中，在预售结束后完成锁单的订单数量")
        report.append("- **小订转化率**: 小订留存锁单数 ÷ 预售累计小订数 × 100%")
        report.append("")
        report.append("**数据来源**:")
        report.append("- 数据文件: intention_order_analysis.parquet")
        report.append("- 业务定义: business_definition.json")
        report.append("- 分析时间: 基于各车型预售周期定义")
        report.append("")
        
        return "\n".join(report)
        
    def save_report(self, filename: str = None):
        """保存分析报告"""
        if not filename:
            filename = f"cm_vehicle_presale_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
        report_content = self.generate_markdown_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        logger.info(f"分析报告已保存至: {filename}")
        return filename

def main():
    """主函数"""
    try:
        # 创建分析器
        analyzer = CMVehiclePresaleAnalyzer()
        
        # 加载数据
        analyzer.load_data()
        analyzer.load_business_definition()
        
        # 执行分析
        results = analyzer.analyze_all_cm_vehicles()
        
        if not results:
            logger.error("分析失败，无有效结果")
            return
            
        # 生成并保存报告
        report_file = analyzer.save_report()
        
        # 输出对比表格到控制台
        print("\n" + "="*80)
        print("CM车型预售分析结果")
        print("="*80)
        
        comparison_df = analyzer.generate_comparison_table()
        if not comparison_df.empty:
            print("\n车型对比表:")
            print(comparison_df.to_string(index=False))
        
        print(f"\n详细报告已保存至: {report_file}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"分析过程出错: {e}")
        raise

if __name__ == "__main__":
    main()