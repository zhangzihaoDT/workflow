#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小订订单趋势监测工具
包含订单、退订、配置、预测四个模块
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import gradio as gr
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据路径
DATA_PATH = "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"
BUSINESS_DEF_PATH = "/Users/zihao_/Documents/github/W35_workflow/business_definition.json"

class OrderTrendMonitor:
    def __init__(self):
        self.df = None
        self.business_def = None
        self.load_data()
        self.load_business_definition()
    
    def load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_parquet(DATA_PATH)
            # 确保日期列为datetime类型
            if 'Intention_Payment_Time' in self.df.columns:
                self.df['Intention_Payment_Time'] = pd.to_datetime(self.df['Intention_Payment_Time'])
            logger.info(f"数据加载成功，共{len(self.df)}条记录")
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise
    
    def load_business_definition(self):
        """加载业务定义"""
        try:
            with open(BUSINESS_DEF_PATH, 'r', encoding='utf-8') as f:
                self.business_def = json.load(f)
            logger.info("业务定义加载成功")
        except Exception as e:
            logger.error(f"业务定义加载失败: {str(e)}")
            raise
    
    def get_vehicle_groups(self) -> List[str]:
        """获取车型分组列表"""
        if '车型分组' in self.df.columns:
            return sorted(self.df['车型分组'].dropna().unique().tolist())
        return []
    
    def calculate_days_from_start(self, vehicle_group: str, date: datetime) -> int:
        """计算从预售开始的天数"""
        if vehicle_group not in self.business_def:
            return -1
        
        start_date = datetime.strptime(self.business_def[vehicle_group]['start'], '%Y-%m-%d')
        return (date - start_date).days + 1  # 第1日开始计算
    
    def prepare_daily_data(self, selected_vehicles: List[str]) -> pd.DataFrame:
        """准备每日数据"""
        if not selected_vehicles:
            return pd.DataFrame()
        
        # 筛选选中的车型
        vehicle_data = self.df[self.df['车型分组'].isin(selected_vehicles)].copy()
        
        # 按车型和日期分组统计
        daily_stats = []
        
        for vehicle in selected_vehicles:
            if vehicle not in self.business_def:
                continue
                
            vehicle_df = vehicle_data[vehicle_data['车型分组'] == vehicle].copy()
            if len(vehicle_df) == 0:
                continue
            
            # 计算该车型的最大天数（基于business_definition.json）
            start_date = datetime.strptime(self.business_def[vehicle]['start'], '%Y-%m-%d')
            end_date = datetime.strptime(self.business_def[vehicle]['end'], '%Y-%m-%d')
            max_days = (end_date - start_date).days + 1
            
            # 按日期分组统计订单数
            daily_orders = vehicle_df.groupby(vehicle_df['Intention_Payment_Time'].dt.date).size().reset_index()
            daily_orders.columns = ['date', 'daily_orders']
            daily_orders['date'] = pd.to_datetime(daily_orders['date'])
            
            # 计算从预售开始的天数
            daily_orders['days_from_start'] = daily_orders['date'].apply(
                lambda x: self.calculate_days_from_start(vehicle, x)
            )
            
            # 过滤有效天数（>=1 且 <= max_days）
            daily_orders = daily_orders[
                (daily_orders['days_from_start'] >= 1) & 
                (daily_orders['days_from_start'] <= max_days)
            ]
            
            # 计算累计订单数
            daily_orders = daily_orders.sort_values('days_from_start')
            daily_orders['cumulative_orders'] = daily_orders['daily_orders'].cumsum()
            
            # 计算环比变化
            daily_orders['daily_change_rate'] = daily_orders['daily_orders'].pct_change() * 100
            
            # 添加车型标识
            daily_orders['vehicle_group'] = vehicle
            
            daily_stats.append(daily_orders)
        
        if daily_stats:
            return pd.concat(daily_stats, ignore_index=True)
        return pd.DataFrame()
    
    def create_cumulative_chart(self, data: pd.DataFrame) -> go.Figure:
        """创建累计小订订单数对比图"""
        fig = go.Figure()
        
        if data.empty:
            fig.add_annotation(
                text="暂无数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig
        
        colors = px.colors.qualitative.Set1
        
        for i, vehicle in enumerate(data['vehicle_group'].unique()):
            vehicle_data = data[data['vehicle_group'] == vehicle]
            
            fig.add_trace(go.Scatter(
                x=vehicle_data['days_from_start'],
                y=vehicle_data['cumulative_orders'],
                mode='lines+markers',
                name=vehicle,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=6),
                hovertemplate=f'<b>{vehicle}</b><br>' +
                             '第%{x}日<br>' +
                             '累计订单: %{y}<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text="📈 累计小订订单数对比图",
                font=dict(size=18, color="#2E86AB")
            ),
            xaxis_title="第N日",
            yaxis_title="累计订单数",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_daily_chart(self, data: pd.DataFrame) -> go.Figure:
        """创建每日小订单数对比图"""
        fig = go.Figure()
        
        if data.empty:
            fig.add_annotation(
                text="暂无数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig
        
        colors = px.colors.qualitative.Set1
        
        for i, vehicle in enumerate(data['vehicle_group'].unique()):
            vehicle_data = data[data['vehicle_group'] == vehicle]
            
            fig.add_trace(go.Bar(
                x=vehicle_data['days_from_start'],
                y=vehicle_data['daily_orders'],
                name=vehicle,
                marker_color=colors[i % len(colors)],
                opacity=0.8,
                hovertemplate=f'<b>{vehicle}</b><br>' +
                             '第%{x}日<br>' +
                             '当日订单: %{y}<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text="📊 每日小订单数对比图",
                font=dict(size=18, color="#2E86AB")
            ),
            xaxis_title="第N日",
            yaxis_title="每日订单数",
            barmode='group',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_change_trend_chart(self, data: pd.DataFrame) -> go.Figure:
        """创建每日小订单数环比变化趋势图"""
        fig = go.Figure()
        
        if data.empty:
            fig.add_annotation(
                text="暂无数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig
        
        colors = px.colors.qualitative.Set1
        
        for i, vehicle in enumerate(data['vehicle_group'].unique()):
            vehicle_data = data[data['vehicle_group'] == vehicle]
            # 过滤掉第一天（无法计算环比）
            vehicle_data = vehicle_data[vehicle_data['days_from_start'] > 1]
            
            if len(vehicle_data) == 0:
                continue
            
            fig.add_trace(go.Scatter(
                x=vehicle_data['days_from_start'],
                y=vehicle_data['daily_change_rate'],
                mode='lines+markers',
                name=vehicle,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=6),
                hovertemplate=f'<b>{vehicle}</b><br>' +
                             '第%{x}日<br>' +
                             '环比变化: %{y:.1f}%<br>' +
                             '<extra></extra>'
            ))
        
        # 添加零线
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title=dict(
                text="📈 每日小订单数环比变化趋势图",
                font=dict(size=18, color="#2E86AB")
            ),
            xaxis_title="第N日",
            yaxis_title="环比变化率 (%)",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_daily_change_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建订单的日变化表格"""
        if data.empty:
            return pd.DataFrame({'提示': ['暂无数据']})
        
        # 准备表格数据
        table_data = []
        
        for vehicle in data['vehicle_group'].unique():
            vehicle_data = data[data['vehicle_group'] == vehicle].sort_values('days_from_start')
            
            for _, row in vehicle_data.iterrows():
                # 环比变化emoji标记
                if pd.isna(row['daily_change_rate']):
                    change_emoji = "➖"
                    change_text = "N/A"
                elif row['daily_change_rate'] > 0:
                    change_emoji = "📈"
                    change_text = f"+{row['daily_change_rate']:.1f}%"
                elif row['daily_change_rate'] < 0:
                    change_emoji = "📉"
                    change_text = f"{row['daily_change_rate']:.1f}%"
                else:
                    change_emoji = "➖"
                    change_text = "0.0%"
                
                table_data.append({
                    '车型': vehicle,
                    '第N日': int(row['days_from_start']),
                    '小订数': int(row['daily_orders']),
                    '累计小订数': int(row['cumulative_orders']),
                    '日环比变化': f"{change_emoji} {change_text}"
                })
        
        return pd.DataFrame(table_data)
    
    def prepare_refund_data(self, selected_vehicles: List[str]) -> pd.DataFrame:
        """准备退订数据"""
        if not selected_vehicles:
            return pd.DataFrame()
        
        # 筛选选中的车型和有退订时间的数据
        vehicle_data = self.df[
            (self.df['车型分组'].isin(selected_vehicles)) & 
            (self.df['intention_refund_time'].notna())
        ].copy()
        
        # 按车型和日期分组统计
        refund_stats = []
        
        for vehicle in selected_vehicles:
            if vehicle not in self.business_def:
                continue
                
            vehicle_df = vehicle_data[vehicle_data['车型分组'] == vehicle].copy()
            if len(vehicle_df) == 0:
                continue
            
            # 计算该车型的最大天数（基于business_definition.json）
            start_date = datetime.strptime(self.business_def[vehicle]['start'], '%Y-%m-%d')
            end_date = datetime.strptime(self.business_def[vehicle]['end'], '%Y-%m-%d')
            max_days = (end_date - start_date).days + 1
            
            # 按退订日期分组统计退订数
            daily_refunds = vehicle_df.groupby(vehicle_df['intention_refund_time'].dt.date).size().reset_index()
            daily_refunds.columns = ['refund_date', 'daily_refunds']
            daily_refunds['refund_date'] = pd.to_datetime(daily_refunds['refund_date'])
            
            # 计算从预售开始的天数
            daily_refunds['days_from_start'] = daily_refunds['refund_date'].apply(
                lambda x: self.calculate_days_from_start(vehicle, x)
            )
            
            # 过滤有效天数（>=1 且 <= max_days）
            daily_refunds = daily_refunds[
                (daily_refunds['days_from_start'] >= 1) & 
                (daily_refunds['days_from_start'] <= max_days)
            ]
            
            # 计算累计退订数
            daily_refunds = daily_refunds.sort_values('days_from_start')
            daily_refunds['cumulative_refunds'] = daily_refunds['daily_refunds'].cumsum()
            
            # 添加车型标识
            daily_refunds['vehicle_group'] = vehicle
            
            refund_stats.append(daily_refunds)
        
        if refund_stats:
            return pd.concat(refund_stats, ignore_index=True)
        return pd.DataFrame()
    
    def prepare_refund_rate_data(self, selected_vehicles: List[str]) -> pd.DataFrame:
        """准备退订率数据（累计小订退订/累计小订）"""
        if not selected_vehicles:
            return pd.DataFrame()
        
        # 获取订单数据和退订数据
        order_data = self.prepare_daily_data(selected_vehicles)
        refund_data = self.prepare_refund_data(selected_vehicles)
        
        if order_data.empty:
            return pd.DataFrame()
        
        # 合并数据计算退订率
        rate_stats = []
        
        for vehicle in selected_vehicles:
            vehicle_orders = order_data[order_data['vehicle_group'] == vehicle].copy()
            vehicle_refunds = refund_data[refund_data['vehicle_group'] == vehicle].copy() if not refund_data.empty else pd.DataFrame()
            
            if vehicle_orders.empty:
                continue
            
            # 创建完整的天数范围
            max_day = vehicle_orders['days_from_start'].max()
            all_days = pd.DataFrame({'days_from_start': range(1, max_day + 1)})
            
            # 合并订单和退订数据
            merged = all_days.merge(vehicle_orders[['days_from_start', 'cumulative_orders']], on='days_from_start', how='left')
            if not vehicle_refunds.empty:
                merged = merged.merge(vehicle_refunds[['days_from_start', 'cumulative_refunds']], on='days_from_start', how='left')
            else:
                merged['cumulative_refunds'] = 0
            
            # 填充缺失值
            merged['cumulative_orders'] = merged['cumulative_orders'].ffill().fillna(0)
            merged['cumulative_refunds'] = merged['cumulative_refunds'].ffill().fillna(0)
            
            # 计算退订率
            merged['refund_rate'] = (merged['cumulative_refunds'] / merged['cumulative_orders'] * 100).fillna(0)
            merged['vehicle_group'] = vehicle
            
            rate_stats.append(merged)
        
        if rate_stats:
            return pd.concat(rate_stats, ignore_index=True)
        return pd.DataFrame()
    
    def prepare_daily_refund_rate_data(self, selected_vehicles: List[str]) -> pd.DataFrame:
        """准备每日小订在当前观察时间范围的累计退订率数据"""
        if not selected_vehicles:
            return pd.DataFrame()
        
        # 获取当前观察时间点（以CM2为基准，固定为第20日）
        observation_day = 20
        
        # 定义各车型的最大天数（基于分析报告数据）
        vehicle_max_days = {
            'CM2': 19,   # CM2最大到第19日
            'CM0': 48,   # CM0最大到第48日
            'CM1': 27,   # CM1最大到第27日
            'DM0': 35,   # DM0最大到第35日
            'DM1': 25    # DM1最大到第25日
        }
        
        daily_rate_stats = []
        
        for vehicle in selected_vehicles:
            if vehicle not in self.business_def:
                continue
            
            vehicle_start = datetime.strptime(self.business_def[vehicle]['start'], '%Y-%m-%d')
            observation_cutoff_date = vehicle_start + timedelta(days=observation_day)
            
            # 获取车型数据
            vehicle_data = self.df[self.df['车型分组'] == vehicle].copy()
            if len(vehicle_data) == 0:
                continue
            
            # 按下单日期分组
            vehicle_data['order_date'] = vehicle_data['Intention_Payment_Time'].dt.date
            daily_orders = vehicle_data.groupby('order_date').size().reset_index(name='daily_orders')
            daily_orders['order_date'] = pd.to_datetime(daily_orders['order_date'])
            
            # 计算从预售开始的天数
            daily_orders['days_from_start'] = (daily_orders['order_date'] - pd.to_datetime(vehicle_start)).dt.days
            
            # 过滤有效天数，限制在车型的最大天数范围内
            max_days = vehicle_max_days.get(vehicle, 50)  # 默认最大50天
            daily_orders = daily_orders[
                (daily_orders['days_from_start'] >= 0) & 
                (daily_orders['days_from_start'] <= max_days)
            ]
            
            # 计算每日订单的退订率
            for _, row in daily_orders.iterrows():
                day_num = row['days_from_start']
                order_date = row['order_date'].date()
                
                # 获取当日下单的数据
                daily_order_data = vehicle_data[vehicle_data['order_date'] == order_date]
                
                # 计算这些订单中在观察截止时间点前退订的数量
                daily_refunds = 0
                if len(daily_order_data) > 0:
                    refunded_data = daily_order_data[
                        (daily_order_data['intention_refund_time'].notna()) &
                        (daily_order_data['intention_refund_time'] <= observation_cutoff_date)
                    ]
                    daily_refunds = len(refunded_data)
                
                # 计算退订率
                refund_rate = (daily_refunds / row['daily_orders'] * 100) if row['daily_orders'] > 0 else 0
                
                daily_rate_stats.append({
                    'vehicle_group': vehicle,
                    'days_from_start': day_num,
                    'daily_orders': row['daily_orders'],
                    'daily_refunds': daily_refunds,
                    'daily_refund_rate': refund_rate
                })
        
        return pd.DataFrame(daily_rate_stats)
    
    def create_cumulative_refund_chart(self, data: pd.DataFrame) -> go.Figure:
        """创建累计小订退订数对比图"""
        fig = go.Figure()
        
        if data.empty:
            fig.add_annotation(
                text="暂无退订数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig
        
        colors = px.colors.qualitative.Set1
        
        for i, vehicle in enumerate(data['vehicle_group'].unique()):
            vehicle_data = data[data['vehicle_group'] == vehicle]
            
            fig.add_trace(go.Scatter(
                x=vehicle_data['days_from_start'],
                y=vehicle_data['cumulative_refunds'],
                mode='lines+markers',
                name=vehicle,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=6),
                hovertemplate=f'<b>{vehicle}</b><br>' +
                             '第%{x}日<br>' +
                             '累计退订: %{y}单<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text="🔄 累计小订退订数对比图",
                font=dict(size=18, color="#E74C3C")
            ),
            xaxis_title="第N日",
            yaxis_title="累计退订数",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_cumulative_refund_rate_chart(self, data: pd.DataFrame) -> go.Figure:
        """创建累计小订退订率对比图"""
        fig = go.Figure()
        
        if data.empty:
            fig.add_annotation(
                text="暂无数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig
        
        colors = px.colors.qualitative.Set1
        
        for i, vehicle in enumerate(data['vehicle_group'].unique()):
            vehicle_data = data[data['vehicle_group'] == vehicle]
            
            fig.add_trace(go.Scatter(
                x=vehicle_data['days_from_start'],
                y=vehicle_data['refund_rate'],
                mode='lines+markers',
                name=vehicle,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=6),
                hovertemplate=f'<b>{vehicle}</b><br>' +
                             '第%{x}日<br>' +
                             '累计退订率: %{y:.2f}%<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text="📊 累计小订退订率对比图",
                font=dict(size=18, color="#E74C3C")
            ),
            xaxis_title="第N日",
            yaxis_title="累计退订率 (%)",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_daily_refund_rate_chart(self, data: pd.DataFrame) -> go.Figure:
        """创建每日小订在当前观察时间范围的累计退订率对比图"""
        fig = go.Figure()
        
        if data.empty:
            fig.add_annotation(
                text="暂无数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig
        
        colors = px.colors.qualitative.Set1
        
        for i, vehicle in enumerate(data['vehicle_group'].unique()):
            vehicle_data = data[data['vehicle_group'] == vehicle]
            
            fig.add_trace(go.Scatter(
                x=vehicle_data['days_from_start'],
                y=vehicle_data['daily_refund_rate'],
                mode='lines+markers',
                name=vehicle,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=6),
                hovertemplate=f'<b>{vehicle}</b><br>' +
                             '第%{x}日<br>' +
                             '当日退订率: %{y:.2f}%<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text="📈 每日小订在当前观察时间范围的累计退订率对比",
                font=dict(size=18, color="#E74C3C")
            ),
            xaxis_title="第N日",
            yaxis_title="当日订单退订率 (%)",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_daily_refund_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建每日订单累计退订情况表格"""
        if data.empty:
            return pd.DataFrame({'提示': ['暂无数据']})
        
        # 准备表格数据
        table_data = []
        
        for vehicle in data['vehicle_group'].unique():
            vehicle_data = data[data['vehicle_group'] == vehicle].sort_values('days_from_start')
            
            for _, row in vehicle_data.iterrows():
                # 退订率emoji标记
                refund_rate = row['daily_refund_rate']
                if refund_rate == 0:
                    rate_emoji = "✅"
                elif refund_rate < 10:
                    rate_emoji = "🟢"
                elif refund_rate < 20:
                    rate_emoji = "🟡"
                else:
                    rate_emoji = "🔴"
                
                refund_situation = f"{int(row['daily_orders'])}订单/{int(row['daily_refunds'])}退订({refund_rate:.1f}%)"
                
                table_data.append({
                    '车型': vehicle,
                    '第N日': int(row['days_from_start']),
                    '当日订单退订情况': f"{rate_emoji} {refund_situation}"
                })
        
        return pd.DataFrame(table_data)

# 创建监控器实例
monitor = OrderTrendMonitor()

def update_charts(selected_vehicles):
    """更新订单图表"""
    try:
        if not selected_vehicles:
            # 返回空图表
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="请选择车型",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            empty_df = pd.DataFrame({'提示': ['请选择车型']})
            return empty_fig, empty_fig, empty_fig, empty_df
        
        # 准备数据
        daily_data = monitor.prepare_daily_data(selected_vehicles)
        
        # 创建图表
        cumulative_chart = monitor.create_cumulative_chart(daily_data)
        daily_chart = monitor.create_daily_chart(daily_data)
        change_trend_chart = monitor.create_change_trend_chart(daily_data)
        daily_table = monitor.create_daily_change_table(daily_data)
        
        return cumulative_chart, daily_chart, change_trend_chart, daily_table
        
    except Exception as e:
        logger.error(f"图表更新失败: {str(e)}")
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"错误: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        error_df = pd.DataFrame({'错误': [str(e)]})
        return error_fig, error_fig, error_fig, error_df

def update_refund_charts(selected_vehicles):
    """更新退订图表"""
    try:
        if not selected_vehicles:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="请选择车型",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            empty_df = pd.DataFrame({'提示': ['请选择车型']})
            return empty_fig, empty_fig, empty_fig, empty_df
        
        # 准备退订相关数据
        refund_data = monitor.prepare_refund_data(selected_vehicles)
        refund_rate_data = monitor.prepare_refund_rate_data(selected_vehicles)
        daily_refund_rate_data = monitor.prepare_daily_refund_rate_data(selected_vehicles)
        
        # 创建图表
        cumulative_refund_chart = monitor.create_cumulative_refund_chart(refund_data)
        cumulative_refund_rate_chart = monitor.create_cumulative_refund_rate_chart(refund_rate_data)
        daily_refund_rate_chart = monitor.create_daily_refund_rate_chart(daily_refund_rate_data)
        daily_refund_table = monitor.create_daily_refund_table(daily_refund_rate_data)
        
        return cumulative_refund_chart, cumulative_refund_rate_chart, daily_refund_rate_chart, daily_refund_table
        
    except Exception as e:
        logger.error(f"退订图表更新失败: {str(e)}")
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"错误: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        error_df = pd.DataFrame({'错误': [str(e)]})
        return error_fig, error_fig, error_fig, error_df

# 获取车型列表
vehicle_groups = monitor.get_vehicle_groups()

# 创建Gradio界面
with gr.Blocks(title="小订订单趋势监测", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚗 小订订单趋势监测工具")
    gr.Markdown("监测各车型小订订单的趋势变化，支持多维度对比分析")
    
    with gr.Tabs():
        # 订单模块
        with gr.Tab("📊 订单"):
            with gr.Row():
                vehicle_selector = gr.CheckboxGroup(
                    choices=vehicle_groups,
                    label="选择车型分组",
                    value=["CM2", "CM1"] if "CM2" in vehicle_groups and "CM1" in vehicle_groups else vehicle_groups[:2],
                    interactive=True
                )
            
            with gr.Row():
                with gr.Column(scale=1):
                    cumulative_plot = gr.Plot(label="累计小订订单数对比图")
                with gr.Column(scale=1):
                    daily_plot = gr.Plot(label="每日小订单数对比图")
            
            with gr.Row():
                with gr.Column(scale=1):
                    change_trend_plot = gr.Plot(label="每日小订单数环比变化趋势图")
                with gr.Column(scale=1):
                    daily_table = gr.DataFrame(
                        label="订单日变化表格",
                        interactive=False,
                        wrap=True
                    )
        
        # 退订模块
        with gr.Tab("🔄 退订"):
            with gr.Row():
                refund_vehicle_selector = gr.CheckboxGroup(
                    choices=vehicle_groups,
                    label="选择车型分组",
                    value=["CM2", "CM1"] if "CM2" in vehicle_groups and "CM1" in vehicle_groups else vehicle_groups[:2],
                    interactive=True
                )
            
            with gr.Row():
                with gr.Column(scale=1):
                    cumulative_refund_plot = gr.Plot(label="累计小订退订数对比图")
                with gr.Column(scale=1):
                    cumulative_refund_rate_plot = gr.Plot(label="累计小订退订率对比图")
            
            with gr.Row():
                with gr.Column(scale=1):
                    daily_refund_rate_plot = gr.Plot(label="每日小订在当前观察时间范围的累计退订率对比")
                with gr.Column(scale=1):
                    daily_refund_table = gr.DataFrame(
                        label="每日订单累计退订情况表格",
                        interactive=False,
                        wrap=True
                    )
        
        # 配置模块（占位）
        with gr.Tab("⚙️ 配置"):
            gr.Markdown("### 配置模块")
            gr.Markdown("*此模块待开发*")
        
        # 预测模块（占位）
        with gr.Tab("🔮 预测"):
            gr.Markdown("### 预测模块")
            gr.Markdown("*此模块待开发*")
    
    # 绑定事件
    vehicle_selector.change(
        fn=update_charts,
        inputs=[vehicle_selector],
        outputs=[cumulative_plot, daily_plot, change_trend_plot, daily_table]
    )
    
    refund_vehicle_selector.change(
        fn=update_refund_charts,
        inputs=[refund_vehicle_selector],
        outputs=[cumulative_refund_plot, cumulative_refund_rate_plot, daily_refund_rate_plot, daily_refund_table]
    )
    
    # 页面加载时自动更新
    demo.load(
        fn=update_charts,
        inputs=[vehicle_selector],
        outputs=[cumulative_plot, daily_plot, change_trend_plot, daily_table]
    )
    
    demo.load(
        fn=update_refund_charts,
        inputs=[refund_vehicle_selector],
        outputs=[cumulative_refund_plot, cumulative_refund_rate_plot, daily_refund_rate_plot, daily_refund_table]
    )
    
    with gr.Accordion("📖 使用说明", open=False):
        gr.Markdown("""
        ### 功能说明
        
        **订单模块**包含四个核心图表：
        1. **累计小订订单数对比图**: 展示各车型从预售开始的累计订单趋势
        2. **每日小订单数对比图**: 对比各车型每日的订单量
        3. **每日小订单数环比变化趋势图**: 显示订单量的日环比变化率
        4. **订单日变化表格**: 详细的数据表格，包含emoji标记的变化趋势
        
        ### 使用方法
        1. 在车型选择器中勾选要对比的车型（默认选择CM2和CM1）
        2. 图表会自动更新显示选中车型的数据
        3. 所有图表基于各车型预售开始日期计算第N日数据
        
        ### 数据说明
        - X轴表示从预售开始的第N日
        - 基于business_definition.json中定义的各车型预售开始时间
        - 环比变化率 = (当日订单数 - 前一日订单数) / 前一日订单数 × 100%
        """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True
    )