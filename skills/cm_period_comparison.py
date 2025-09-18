#!/usr/bin/env python3
"""
CM时期对比分析脚本
对比CM0、CM1、CM2时期结束后5天（包含end日）的有效线索数和本品牌人群总资产
"""

import pandas as pd
import numpy as np
import gradio as gr
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CMPeriodComparison:
    def __init__(self, data_path, business_def_path):
        """
        初始化分析器
        
        Args:
            data_path: 业务数据文件路径
            business_def_path: 业务定义文件路径
        """
        self.data_path = data_path
        self.business_def_path = business_def_path
        self.data = None
        self.business_def = None
        self.analysis_results = {}
        
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        
        # 加载业务数据
        self.data = pd.read_parquet(self.data_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        print(f"业务数据加载完成，共 {len(self.data)} 行")
        
        # 加载业务定义
        with open(self.business_def_path, 'r', encoding='utf-8') as f:
            self.business_def = json.load(f)
        print("业务定义加载完成")
        
        # 显示数据基本信息
        print(f"数据日期范围: {self.data['date'].min()} 到 {self.data['date'].max()}")
        print(f"关键指标列: 有效线索数, 本品牌人群总资产资产")
        
    def get_period_analysis_dates(self, period_name):
        """
        获取指定时期的分析日期范围（end日期后5天，包含end日）
        
        Args:
            period_name: 时期名称 (CM0, CM1, CM2)
            
        Returns:
            tuple: (start_date, end_date)
        """
        period_info = self.business_def['time_periods'][period_name]
        end_date = datetime.strptime(period_info['end'], '%Y-%m-%d')
        
        # 分析期间：end日期后5天（包含end日）
        analysis_start = end_date
        analysis_end = end_date + timedelta(days=4)
        
        return analysis_start, analysis_end
        
    def analyze_period(self, period_name):
        """
        分析指定时期的数据
        
        Args:
            period_name: 时期名称
            
        Returns:
            dict: 分析结果
        """
        start_date, end_date = self.get_period_analysis_dates(period_name)
        
        # 筛选数据
        period_data = self.data[
            (self.data['date'] >= start_date) & 
            (self.data['date'] <= end_date)
        ].copy()
        
        if len(period_data) == 0:
            print(f"警告: {period_name} 时期 ({start_date.date()} 到 {end_date.date()}) 没有数据")
            return {
                'period': period_name,
                'start_date': start_date.date(),
                'end_date': end_date.date(),
                'data_count': 0,
                'total_leads': 0,
                'avg_daily_leads': 0,
                'total_brand_assets': 0,
                'avg_daily_brand_assets': 0,
                'daily_data': pd.DataFrame()
            }
        
        # 计算指标
        total_leads = period_data['有效线索数'].sum()
        avg_daily_leads = period_data['有效线索数'].mean()
        total_brand_assets = period_data['本品牌人群总资产资产'].sum()
        avg_daily_brand_assets = period_data['本品牌人群总资产资产'].mean()
        
        result = {
            'period': period_name,
            'start_date': start_date.date(),
            'end_date': end_date.date(),
            'data_count': len(period_data),
            'total_leads': total_leads,
            'avg_daily_leads': avg_daily_leads,
            'total_brand_assets': total_brand_assets,
            'avg_daily_brand_assets': avg_daily_brand_assets,
            'daily_data': period_data[['date', '有效线索数', '本品牌人群总资产资产']].copy()
        }
        
        print(f"\n{period_name} 时期分析结果:")
        print(f"  分析期间: {start_date.date()} 到 {end_date.date()}")
        print(f"  数据天数: {len(period_data)} 天")
        print(f"  总有效线索数: {total_leads:,.0f}")
        print(f"  日均有效线索数: {avg_daily_leads:,.1f}")
        print(f"  总本品牌人群资产: {total_brand_assets:,.0f}")
        print(f"  日均本品牌人群资产: {avg_daily_brand_assets:,.1f}")
        
        return result
        
    def run_comparison_analysis(self):
        """运行对比分析"""
        print("开始CM时期对比分析...")
        
        # 分析各个时期
        periods = ['CM0', 'CM1', 'CM2']
        for period in periods:
            self.analysis_results[period] = self.analyze_period(period)
            
        # 生成对比报告
        self.generate_comparison_report()
        
        # 准备图表数据
        return self.prepare_chart_data()
        
    def generate_comparison_report(self):
        """生成对比报告"""
        print("\n" + "="*60)
        print("CM时期对比分析报告")
        print("="*60)
        
        # 创建对比表格
        comparison_data = []
        for period in ['CM0', 'CM1', 'CM2']:
            result = self.analysis_results[period]
            comparison_data.append({
                '时期': period,
                '分析期间': f"{result['start_date']} 到 {result['end_date']}",
                '数据天数': result['data_count'],
                '总有效线索数': f"{result['total_leads']:,.0f}",
                '日均有效线索数': f"{result['avg_daily_leads']:,.1f}",
                '总本品牌人群资产': f"{result['total_brand_assets']:,.0f}",
                '日均本品牌人群资产': f"{result['avg_daily_brand_assets']:,.1f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n对比汇总表:")
        print(comparison_df.to_string(index=False))
        
        # 计算增长率
        print("\n增长率分析:")
        if self.analysis_results['CM0']['data_count'] > 0:
            # CM1 vs CM0
            if self.analysis_results['CM1']['data_count'] > 0:
                leads_growth_cm1 = ((self.analysis_results['CM1']['avg_daily_leads'] - 
                                   self.analysis_results['CM0']['avg_daily_leads']) / 
                                  self.analysis_results['CM0']['avg_daily_leads'] * 100)
                assets_growth_cm1 = ((self.analysis_results['CM1']['avg_daily_brand_assets'] - 
                                     self.analysis_results['CM0']['avg_daily_brand_assets']) / 
                                    self.analysis_results['CM0']['avg_daily_brand_assets'] * 100)
                print(f"CM1 vs CM0:")
                print(f"  日均有效线索数增长率: {leads_growth_cm1:+.1f}%")
                print(f"  日均本品牌人群资产增长率: {assets_growth_cm1:+.1f}%")
            
            # CM2 vs CM0
            if self.analysis_results['CM2']['data_count'] > 0:
                leads_growth_cm2 = ((self.analysis_results['CM2']['avg_daily_leads'] - 
                                   self.analysis_results['CM0']['avg_daily_leads']) / 
                                  self.analysis_results['CM0']['avg_daily_leads'] * 100)
                assets_growth_cm2 = ((self.analysis_results['CM2']['avg_daily_brand_assets'] - 
                                     self.analysis_results['CM0']['avg_daily_brand_assets']) / 
                                    self.analysis_results['CM0']['avg_daily_brand_assets'] * 100)
                print(f"CM2 vs CM0:")
                print(f"  日均有效线索数增长率: {leads_growth_cm2:+.1f}%")
                print(f"  日均本品牌人群资产增长率: {assets_growth_cm2:+.1f}%")
        
        # CM2 vs CM1
        if (self.analysis_results['CM1']['data_count'] > 0 and 
            self.analysis_results['CM2']['data_count'] > 0):
            leads_growth_cm2_cm1 = ((self.analysis_results['CM2']['avg_daily_leads'] - 
                                   self.analysis_results['CM1']['avg_daily_leads']) / 
                                  self.analysis_results['CM1']['avg_daily_leads'] * 100)
            assets_growth_cm2_cm1 = ((self.analysis_results['CM2']['avg_daily_brand_assets'] - 
                                     self.analysis_results['CM1']['avg_daily_brand_assets']) / 
                                    self.analysis_results['CM1']['avg_daily_brand_assets'] * 100)
            print(f"CM2 vs CM1:")
            print(f"  日均有效线索数增长率: {leads_growth_cm2_cm1:+.1f}%")
            print(f"  日均本品牌人群资产增长率: {assets_growth_cm2_cm1:+.1f}%")
        
    def prepare_chart_data(self):
        """准备图表数据"""
        chart_data = {}
        
        # 准备日均有效线索数数据
        avg_leads_data = []
        for period in ['CM0', 'CM1', 'CM2']:
            if self.analysis_results[period]['data_count'] > 0:
                avg_leads_data.append({
                    'period': period,
                    'value': self.analysis_results[period]['avg_daily_leads']
                })
        chart_data['avg_leads'] = pd.DataFrame(avg_leads_data)
        
        # 准备日均本品牌人群资产数据
        avg_assets_data = []
        for period in ['CM0', 'CM1', 'CM2']:
            if self.analysis_results[period]['data_count'] > 0:
                avg_assets_data.append({
                    'period': period,
                    'value': self.analysis_results[period]['avg_daily_brand_assets']
                })
        chart_data['avg_assets'] = pd.DataFrame(avg_assets_data)
        
        # 准备总有效线索数数据
        total_leads_data = []
        for period in ['CM0', 'CM1', 'CM2']:
            if self.analysis_results[period]['data_count'] > 0:
                total_leads_data.append({
                    'period': period,
                    'value': self.analysis_results[period]['total_leads']
                })
        chart_data['total_leads'] = pd.DataFrame(total_leads_data)
        
        # 准备总本品牌人群资产数据
        total_assets_data = []
        for period in ['CM0', 'CM1', 'CM2']:
            if self.analysis_results[period]['data_count'] > 0:
                total_assets_data.append({
                    'period': period,
                    'value': self.analysis_results[period]['total_brand_assets']
                })
        chart_data['total_assets'] = pd.DataFrame(total_assets_data)
        
        return chart_data


def run_analysis():
    """运行分析并返回结果"""
    # 文件路径
    data_path = '/Users/zihao_/Documents/coding/dataset/formatted/business_daily_metrics.parquet'
    business_def_path = '/Users/zihao_/Documents/github/W35_workflow/business_definition.json'
    
    # 创建分析器
    analyzer = CMPeriodComparison(data_path, business_def_path)
    
    try:
        # 加载数据
        analyzer.load_data()
        
        # 运行分析
        chart_data = analyzer.run_comparison_analysis()
        
        # 生成汇总报告文本
        summary_text = generate_summary_text(analyzer.analysis_results)
        
        return (
            summary_text,
            chart_data['avg_leads'],
            chart_data['avg_assets'], 
            chart_data['total_leads'],
            chart_data['total_assets']
        )
        
    except Exception as e:
        error_msg = f"分析过程中出现错误: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, None, None, None, None

def generate_summary_text(analysis_results):
    """生成汇总报告文本"""
    summary = "# CM时期对比分析报告\n\n"
    
    # 基本信息
    summary += "## 分析概览\n\n"
    for period in ['CM0', 'CM1', 'CM2']:
        result = analysis_results[period]
        summary += f"**{period}时期**\n"
        summary += f"- 分析期间: {result['start_date']} 到 {result['end_date']}\n"
        summary += f"- 数据天数: {result['data_count']} 天\n"
        summary += f"- 总有效线索数: {result['total_leads']:,.0f}\n"
        summary += f"- 日均有效线索数: {result['avg_daily_leads']:,.1f}\n"
        summary += f"- 总本品牌人群资产: {result['total_brand_assets']:,.0f}\n"
        summary += f"- 日均本品牌人群资产: {result['avg_daily_brand_assets']:,.1f}\n\n"
    
    # 增长率分析
    summary += "## 增长率分析\n\n"
    if analysis_results['CM0']['data_count'] > 0:
        # CM1 vs CM0
        if analysis_results['CM1']['data_count'] > 0:
            leads_growth_cm1 = ((analysis_results['CM1']['avg_daily_leads'] - 
                               analysis_results['CM0']['avg_daily_leads']) / 
                              analysis_results['CM0']['avg_daily_leads'] * 100)
            assets_growth_cm1 = ((analysis_results['CM1']['avg_daily_brand_assets'] - 
                                 analysis_results['CM0']['avg_daily_brand_assets']) / 
                                analysis_results['CM0']['avg_daily_brand_assets'] * 100)
            summary += f"**CM1 vs CM0:**\n"
            summary += f"- 日均有效线索数增长率: {leads_growth_cm1:+.1f}%\n"
            summary += f"- 日均本品牌人群资产增长率: {assets_growth_cm1:+.1f}%\n\n"
        
        # CM2 vs CM0
        if analysis_results['CM2']['data_count'] > 0:
            leads_growth_cm2 = ((analysis_results['CM2']['avg_daily_leads'] - 
                               analysis_results['CM0']['avg_daily_leads']) / 
                              analysis_results['CM0']['avg_daily_leads'] * 100)
            assets_growth_cm2 = ((analysis_results['CM2']['avg_daily_brand_assets'] - 
                                 analysis_results['CM0']['avg_daily_brand_assets']) / 
                                analysis_results['CM0']['avg_daily_brand_assets'] * 100)
            summary += f"**CM2 vs CM0:**\n"
            summary += f"- 日均有效线索数增长率: {leads_growth_cm2:+.1f}%\n"
            summary += f"- 日均本品牌人群资产增长率: {assets_growth_cm2:+.1f}%\n\n"
    
    # CM2 vs CM1
    if (analysis_results['CM1']['data_count'] > 0 and 
        analysis_results['CM2']['data_count'] > 0):
        leads_growth_cm2_cm1 = ((analysis_results['CM2']['avg_daily_leads'] - 
                               analysis_results['CM1']['avg_daily_leads']) / 
                              analysis_results['CM1']['avg_daily_leads'] * 100)
        assets_growth_cm2_cm1 = ((analysis_results['CM2']['avg_daily_brand_assets'] - 
                                 analysis_results['CM1']['avg_daily_brand_assets']) / 
                                analysis_results['CM1']['avg_daily_brand_assets'] * 100)
        summary += f"**CM2 vs CM1:**\n"
        summary += f"- 日均有效线索数增长率: {leads_growth_cm2_cm1:+.1f}%\n"
        summary += f"- 日均本品牌人群资产增长率: {assets_growth_cm2_cm1:+.1f}%\n\n"
    
    return summary

def create_gradio_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="CM时期对比分析", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# CM时期对比分析系统")
        gr.Markdown("对比CM0、CM1、CM2时期结束后5天（包含end日）的有效线索数和本品牌人群总资产")
        
        # 分析按钮
        analyze_btn = gr.Button("开始分析", variant="primary", size="lg")
        
        # 结果展示区域
        with gr.Row():
            with gr.Column(scale=1):
                summary_output = gr.Markdown("点击'开始分析'按钮开始分析...")
        
        # 图表展示区域
        with gr.Row():
            with gr.Column():
                avg_leads_chart = gr.BarPlot(
                    title="日均有效线索数对比",
                    x="period",
                    y="value",
                    color="period",
                    width=400,
                    height=300
                )
            with gr.Column():
                avg_assets_chart = gr.BarPlot(
                    title="日均本品牌人群资产对比",
                    x="period", 
                    y="value",
                    color="period",
                    width=400,
                    height=300
                )
        
        with gr.Row():
            with gr.Column():
                total_leads_chart = gr.BarPlot(
                    title="总有效线索数对比（5天总和）",
                    x="period",
                    y="value", 
                    color="period",
                    width=400,
                    height=300
                )
            with gr.Column():
                total_assets_chart = gr.BarPlot(
                    title="总本品牌人群资产对比（5天总和）",
                    x="period",
                    y="value",
                    color="period", 
                    width=400,
                    height=300
                )
        
        # 绑定分析函数
        analyze_btn.click(
            fn=run_analysis,
            outputs=[
                summary_output,
                avg_leads_chart,
                avg_assets_chart,
                total_leads_chart,
                total_assets_chart
            ]
        )
    
    return demo

def main():
    """主函数"""
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7864,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()