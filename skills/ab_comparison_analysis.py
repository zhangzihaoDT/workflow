#!/usr/bin/env python3
"""
AB对比分析脚本
用于对比两个样本之间的差异，发现异常
使用Gradio作为前端界面
"""

import pandas as pd
import gradio as gr
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据文件路径
DATA_PATH = "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"

class ABComparisonAnalyzer:
    def __init__(self):
        self.df = None
        self.load_data()
    
    def load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_parquet(DATA_PATH)
            self.df['Intention_Payment_Time'] = pd.to_datetime(self.df['Intention_Payment_Time'])
            if 'intention_refund_time' in self.df.columns:
                self.df['intention_refund_time'] = pd.to_datetime(self.df['intention_refund_time'])
            logger.info(f"数据加载成功，共{len(self.df)}条记录")
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise e
    
    def get_vehicle_types(self) -> List[str]:
        """获取车型列表"""
        if '车型分组' in self.df.columns:
            return sorted(self.df['车型分组'].unique().tolist())
        return []
    
    def get_pre_vehicle_model_types(self) -> List[str]:
        """获取pre_vehicle_model_type列表"""
        if 'pre_vehicle_model_type' in self.df.columns:
            return sorted([str(x) for x in self.df['pre_vehicle_model_type'].dropna().unique().tolist()])
        return []
    
    def get_parent_regions(self) -> List[str]:
        """获取Parent Region Name列表"""
        if 'Parent Region Name' in self.df.columns:
            return sorted(self.df['Parent Region Name'].dropna().unique().tolist())
        return []
    
    def get_date_range(self) -> Tuple[str, str]:
        """获取数据的日期范围"""
        min_date = self.df['Intention_Payment_Time'].min().strftime('%Y-%m-%d')
        max_date = self.df['Intention_Payment_Time'].max().strftime('%Y-%m-%d')
        return min_date, max_date
    
    def get_refund_date_range(self) -> Tuple[str, str]:
        """获取退订时间的日期范围"""
        if 'intention_refund_time' in self.df.columns:
            refund_data = self.df['intention_refund_time'].dropna()
            if len(refund_data) > 0:
                min_date = refund_data.min().strftime('%Y-%m-%d')
                max_date = refund_data.max().strftime('%Y-%m-%d')
                return min_date, max_date
        return '', ''
    
    def filter_sample(self, start_date: str = '', end_date: str = '', vehicle_types: List[str] = None, 
                     include_refund: bool = False, refund_start_date: str = '', refund_end_date: str = '',
                     pre_vehicle_model_types: List[str] = None, parent_regions: List[str] = None,
                     vehicle_groups: List[str] = None, refund_only: bool = False, 
                     locked_only: bool = False) -> pd.DataFrame:
        """筛选样本数据"""
        # 从完整数据开始
        mask = pd.Series([True] * len(self.df), index=self.df.index)
        
        # 1. 小订时间范围筛选
        if start_date and end_date:
            mask = mask & (self.df['Intention_Payment_Time'] >= start_date) & \
                   (self.df['Intention_Payment_Time'] <= end_date)
        
        # 2. 小订退订时间范围筛选
        if refund_start_date and refund_end_date and 'intention_refund_time' in self.df.columns:
            refund_mask = (self.df['intention_refund_time'] >= refund_start_date) & \
                         (self.df['intention_refund_time'] <= refund_end_date)
            mask = mask & refund_mask
        
        # 3. pre_vehicle_model_type筛选
        if pre_vehicle_model_types and 'pre_vehicle_model_type' in self.df.columns:
            mask = mask & (self.df['pre_vehicle_model_type'].astype(str).isin(pre_vehicle_model_types))
        
        # 4. Parent Region Name筛选
        if parent_regions and 'Parent Region Name' in self.df.columns:
            mask = mask & (self.df['Parent Region Name'].isin(parent_regions))
        
        # 5. 车型分组筛选
        if vehicle_groups and '车型分组' in self.df.columns:
            mask = mask & (self.df['车型分组'].isin(vehicle_groups))
        elif vehicle_types and '车型分组' in self.df.columns:  # 保持向后兼容
            mask = mask & (self.df['车型分组'].isin(vehicle_types))
        
        sample_data = self.df[mask].copy()
        
        # 6. 是否退订筛选
        if refund_only and 'intention_refund_time' in self.df.columns:
            sample_data = sample_data[sample_data['intention_refund_time'].notna()]
        elif include_refund and 'intention_refund_time' in self.df.columns:  # 保持向后兼容
            sample_data = sample_data[sample_data['intention_refund_time'].notna()]
        
        # 7. 是否锁单筛选
        if locked_only and 'Lock_Time' in self.df.columns:
            sample_data = sample_data[sample_data['Lock_Time'].notna()]
        
        return sample_data
    
    def analyze_region_distribution(self, sample_a: pd.DataFrame, sample_b: pd.DataFrame, 
                                  parent_regions_filter: List[str] = None) -> List[Dict]:
        """地区分布异常检测"""
        anomalies = []
        
        # 如果有Parent Region筛选，只检测相关的地区
        if parent_regions_filter:
            # 只检测筛选范围内的Province和City
            filtered_sample_a = sample_a[sample_a['Parent Region Name'].isin(parent_regions_filter)]
            filtered_sample_b = sample_b[sample_b['Parent Region Name'].isin(parent_regions_filter)]
            
            # 检查Province和City维度
            region_columns = ['License Province', 'license_city_level', 'License City']
            samples_to_check = [(filtered_sample_a, filtered_sample_b, 'filtered')]
        else:
            # 检查所有地区维度
            region_columns = ['Parent Region Name', 'License Province', 'license_city_level', 'License City']
            samples_to_check = [(sample_a, sample_b, 'all')]
        
        for sample_a_check, sample_b_check, scope in samples_to_check:
            for region_col in region_columns:
                if region_col not in sample_a_check.columns or region_col not in sample_b_check.columns:
                    continue
                
                # 计算分布
                dist_a = sample_a_check[region_col].value_counts(normalize=True)
                dist_b = sample_b_check[region_col].value_counts(normalize=True)
                
                # 计算绝对数量
                count_a = sample_a_check[region_col].value_counts()
                count_b = sample_b_check[region_col].value_counts()
                
                # 获取所有地区
                all_regions = set(dist_a.index) | set(dist_b.index)
                
                # 检查异常（使用"或"逻辑：结构占比变化 OR 对比环比变化）
                for region in all_regions:
                    ratio_a = dist_a.get(region, 0)
                    ratio_b = dist_b.get(region, 0)
                    abs_a = count_a.get(region, 0)
                    abs_b = count_b.get(region, 0)
                    
                    # 条件1：结构占比异常（占比超过1%且变化幅度超过10%）
                    structure_anomaly = False
                    if ratio_b > 0 and ratio_a > 0.01:
                        change_rate = abs(ratio_a - ratio_b) / ratio_b
                        if change_rate > 0.1:
                            structure_anomaly = True
                    
                    # 条件2：环比变化异常（基于绝对值的环比变化）
                    comparison_anomaly = False
                    if abs_b > 0:
                        abs_relative_change = abs((abs_a - abs_b) / abs_b)
                        # 检测环比变化超过50%的情况（包括100%增长）
                        if abs_relative_change > 0.5:
                            comparison_anomaly = True
                    elif abs_a > 0:  # 新出现的情况也算环比异常
                        comparison_anomaly = True
                    
                    # 条件3：新出现的地区
                    new_region_anomaly = ratio_a > 0.01 and ratio_b == 0
                    
                    # "或"逻辑：满足任一条件即为异常
                    if structure_anomaly or comparison_anomaly or new_region_anomaly:
                        if ratio_b > 0:
                             change_rate = abs(ratio_a - ratio_b) / ratio_b
                             # 计算绝对值的环比变化
                             if abs_b > 0:
                                 relative_change = (abs_a - abs_b) / abs_b
                             else:
                                 relative_change = float('inf') if abs_a > 0 else 0
                             
                             anomaly_type = []
                             if structure_anomaly:
                                 anomaly_type.append("结构占比")
                             if comparison_anomaly:
                                 anomaly_type.append("对比环比")
                             
                             anomalies.append({
                                 'type': '地区分布',
                                 'item': f'[{region_col}] {region}',
                                 'anomaly_type': '/'.join(anomaly_type),
                                 'sample_a_abs': abs_a,
                                 'sample_b_abs': abs_b,
                                 'sample_a_ratio': ratio_a,
                                 'sample_b_ratio': ratio_b,
                                 'change': abs(ratio_a - ratio_b),
                                 'relative_change': relative_change
                             })
                        else:  # 新出现的地区
                            anomalies.append({
                                'type': '地区分布',
                                'item': f'[{region_col}] {region}',
                                'anomaly_type': '新增地区',
                                'sample_a_abs': abs_a,
                                'sample_b_abs': abs_b,
                                'sample_a_ratio': ratio_a,
                                'sample_b_ratio': ratio_b,
                                'change': ratio_a,
                                'relative_change': float('inf')
                            })
        
        return anomalies
    
    def analyze_channel_structure(self, sample_a: pd.DataFrame, sample_b: pd.DataFrame) -> List[Dict]:
        """渠道结构异常检测"""
        anomalies = []
        
        # 使用与main.py一致的字段名
        channel_columns = ['first_middle_channel_name', 'Channel', 'Sub_Channel', 'Dealer_Name']
        
        for channel_col in channel_columns:
            if channel_col not in sample_a.columns or channel_col not in sample_b.columns:
                continue
            
            # 计算分布
            dist_a = sample_a[channel_col].value_counts(normalize=True)
            dist_b = sample_b[channel_col].value_counts(normalize=True)
            
            # 计算绝对数量
            count_a = sample_a[channel_col].value_counts()
            count_b = sample_b[channel_col].value_counts()
            
            # 获取所有渠道
            all_channels = set(dist_a.index) | set(dist_b.index)
            
            # 检查异常（使用"或"逻辑：结构占比变化 OR 对比环比变化）
            for channel in all_channels:
                ratio_a = dist_a.get(channel, 0)
                ratio_b = dist_b.get(channel, 0)
                abs_a = count_a.get(channel, 0)
                abs_b = count_b.get(channel, 0)
                
                # 条件1：结构占比异常（占比超过1%且变化幅度超过15%）
                structure_anomaly = False
                if ratio_b > 0 and ratio_a > 0.01:
                    change_rate = abs(ratio_a - ratio_b) / ratio_b
                    if change_rate > 0.15:
                        structure_anomaly = True
                
                # 条件2：环比变化异常（基于绝对值的环比变化）
                comparison_anomaly = False
                if abs_b > 0:
                    abs_relative_change = abs((abs_a - abs_b) / abs_b)
                    # 检测环比变化超过50%的情况（包括100%增长）
                    if abs_relative_change > 0.5:
                        comparison_anomaly = True
                elif abs_a > 0:  # 新出现的情况也算环比异常
                    comparison_anomaly = True
                
                # 条件3：新出现的渠道
                new_channel_anomaly = ratio_a > 0.01 and ratio_b == 0
                
                # "或"逻辑：满足任一条件即为异常
                if structure_anomaly or comparison_anomaly or new_channel_anomaly:
                    if ratio_b > 0:
                         change_rate = abs(ratio_a - ratio_b) / ratio_b
                         # 计算绝对值的环比变化
                         if abs_b > 0:
                             relative_change = (abs_a - abs_b) / abs_b
                         else:
                             relative_change = float('inf') if abs_a > 0 else 0
                         
                         anomaly_type = []
                         if structure_anomaly:
                             anomaly_type.append("结构占比")
                         if comparison_anomaly:
                             anomaly_type.append("对比环比")
                         
                         anomalies.append({
                             'type': '渠道结构',
                             'item': f'[{channel_col}] {channel}',
                             'anomaly_type': '/'.join(anomaly_type),
                             'sample_a_abs': abs_a,
                             'sample_b_abs': abs_b,
                             'sample_a_ratio': ratio_a,
                             'sample_b_ratio': ratio_b,
                             'change': abs(ratio_a - ratio_b),
                             'relative_change': relative_change
                         })
                    else:  # 新出现的渠道
                        anomalies.append({
                            'type': '渠道结构',
                            'item': f'[{channel_col}] {channel}',
                            'anomaly_type': '新增渠道',
                            'sample_a_abs': abs_a,
                            'sample_b_abs': abs_b,
                            'sample_a_ratio': ratio_a,
                            'sample_b_ratio': ratio_b,
                            'change': ratio_a,
                            'relative_change': float('inf')
                        })
        
        return anomalies
    
    def analyze_demographic_structure(self, sample_a: pd.DataFrame, sample_b: pd.DataFrame) -> List[Dict]:
        """人群结构异常检测"""
        anomalies = []
        
        # 性别分布检查
        if 'order_gender' in sample_a.columns and 'order_gender' in sample_b.columns:
            gender_dist_a = sample_a['order_gender'].value_counts(normalize=True)
            gender_dist_b = sample_b['order_gender'].value_counts(normalize=True)
            
            # 计算绝对数量
            gender_count_a = sample_a['order_gender'].value_counts()
            gender_count_b = sample_b['order_gender'].value_counts()
            
            for gender in set(gender_dist_a.index) | set(gender_dist_b.index):
                ratio_a = gender_dist_a.get(gender, 0)
                ratio_b = gender_dist_b.get(gender, 0)
                abs_a = gender_count_a.get(gender, 0)
                abs_b = gender_count_b.get(gender, 0)
                
                # 条件1：结构占比异常（变化幅度超过10%）
                structure_anomaly = False
                if ratio_b > 0:
                    change_rate = abs(ratio_a - ratio_b) / ratio_b
                    if change_rate > 0.1:
                        structure_anomaly = True
                
                # 条件2：环比变化异常（基于绝对值的环比变化）
                comparison_anomaly = False
                if abs_b > 0:
                    abs_relative_change = abs((abs_a - abs_b) / abs_b)
                    # 检测环比变化超过50%的情况（包括100%增长）
                    if abs_relative_change > 0.5:
                        comparison_anomaly = True
                elif abs_a > 0:  # 新出现的情况也算环比异常
                    comparison_anomaly = True
                
                # "或"逻辑：满足任一条件即为异常
                if structure_anomaly or comparison_anomaly:
                     change_rate = abs(ratio_a - ratio_b) / ratio_b
                     # 计算绝对值的环比变化
                     if abs_b > 0:
                         relative_change = (abs_a - abs_b) / abs_b
                     else:
                         relative_change = float('inf') if abs_a > 0 else 0
                     
                     anomaly_type = []
                     if structure_anomaly:
                         anomaly_type.append("结构占比")
                     if comparison_anomaly:
                         anomaly_type.append("对比环比")
                     
                     anomalies.append({
                         'type': '人群结构',
                         'item': f'{gender}性别',
                         'anomaly_type': '/'.join(anomaly_type),
                         'sample_a_abs': abs_a,
                         'sample_b_abs': abs_b,
                         'sample_a_ratio': ratio_a,
                         'sample_b_ratio': ratio_b,
                         'change': abs(ratio_a - ratio_b),
                         'relative_change': relative_change
                     })
        
        # 年龄分布检查 - 使用buyer_age字段创建年龄段
        if 'buyer_age' in sample_a.columns and 'buyer_age' in sample_b.columns:
            # 创建年龄段
            def create_age_groups(df):
                df = df.copy()
                df['age_group'] = pd.cut(df['buyer_age'], 
                                       bins=[0, 25, 35, 45, 55, 100], 
                                       labels=['25岁以下', '25-35岁', '35-45岁', '45-55岁', '55岁以上'],
                                       right=False)
                return df
            
            sample_a_with_age = create_age_groups(sample_a)
            sample_b_with_age = create_age_groups(sample_b)
            
            age_dist_a = sample_a_with_age['age_group'].value_counts(normalize=True)
            age_dist_b = sample_b_with_age['age_group'].value_counts(normalize=True)
            
            # 计算绝对数量
            age_count_a = sample_a_with_age['age_group'].value_counts()
            age_count_b = sample_b_with_age['age_group'].value_counts()
            
            for age_group in set(age_dist_a.index) | set(age_dist_b.index):
                if pd.isna(age_group):  # 跳过NaN值
                    continue
                ratio_a = age_dist_a.get(age_group, 0)
                ratio_b = age_dist_b.get(age_group, 0)
                abs_a = age_count_a.get(age_group, 0)
                abs_b = age_count_b.get(age_group, 0)
                
                # 条件1：结构占比异常（占比超过1%且变化幅度超过15%）
                structure_anomaly = False
                if ratio_b > 0 and ratio_a > 0.01:
                    change_rate = abs(ratio_a - ratio_b) / ratio_b
                    if change_rate > 0.15:
                        structure_anomaly = True
                
                # 条件2：环比变化异常（基于绝对值的环比变化）
                comparison_anomaly = False
                if abs_b > 0:
                    abs_relative_change = abs((abs_a - abs_b) / abs_b)
                    # 检测环比变化超过50%的情况（包括100%增长）
                    if abs_relative_change > 0.5:
                        comparison_anomaly = True
                elif abs_a > 0:  # 新出现的情况也算环比异常
                    comparison_anomaly = True
                
                # "或"逻辑：满足任一条件即为异常
                if structure_anomaly or comparison_anomaly:
                     change_rate = abs(ratio_a - ratio_b) / ratio_b
                     # 计算绝对值的环比变化
                     if abs_b > 0:
                         relative_change = (abs_a - abs_b) / abs_b
                     else:
                         relative_change = float('inf') if abs_a > 0 else 0
                     
                     anomaly_type = []
                     if structure_anomaly:
                         anomaly_type.append("结构占比")
                     if comparison_anomaly:
                         anomaly_type.append("对比环比")
                     
                     anomalies.append({
                         'type': '人群结构',
                         'item': f'{age_group}年龄段',
                         'anomaly_type': '/'.join(anomaly_type),
                         'sample_a_abs': abs_a,
                         'sample_b_abs': abs_b,
                         'sample_a_ratio': ratio_a,
                         'sample_b_ratio': ratio_b,
                         'change': abs(ratio_a - ratio_b),
                         'relative_change': relative_change
                     })
        
        return anomalies
    
    def generate_comparison_report(self, sample_a: pd.DataFrame, sample_b: pd.DataFrame, 
                                 sample_a_desc: str, sample_b_desc: str, 
                                 parent_regions_filter: List[str] = None) -> Tuple[str, pd.DataFrame]:
        """生成对比分析报告"""
        # 执行三种异常检测
        region_anomalies = self.analyze_region_distribution(sample_a, sample_b, parent_regions_filter)
        channel_anomalies = self.analyze_channel_structure(sample_a, sample_b)
        demographic_anomalies = self.analyze_demographic_structure(sample_a, sample_b)
        
        # 合并所有异常数据
        all_anomalies = region_anomalies + channel_anomalies + demographic_anomalies
        
        # 创建异常检测结果表格
        anomaly_data = []
        
        for anomaly in all_anomalies:
            anomaly_data.append({
                '异常类型': anomaly['type'],
                '异常项目': anomaly['item'],
                '异常子类': anomaly['anomaly_type'],
                '样本A绝对值': f"{anomaly['sample_a_abs']:,}",
                '样本B绝对值': f"{anomaly['sample_b_abs']:,}",
                '样本A占比': f"{anomaly['sample_a_ratio']:.2%}",
                '样本B占比': f"{anomaly['sample_b_ratio']:.2%}",
                '占比变化': f"{anomaly['change']:+.2%}",
                '环比变化': f"{anomaly['relative_change']:+.1%}" if anomaly['relative_change'] != float('inf') else "新增",
                '风险等级': '⚠️ 中等'
            })
        
        # 如果没有异常，添加正常状态
        if not anomaly_data:
            anomaly_data.append({
                '异常类型': '整体评估',
                '异常项目': '无异常',
                '异常子类': '正常',
                '样本A绝对值': '-',
                '样本B绝对值': '-',
                '样本A占比': '-',
                '样本B占比': '-',
                '占比变化': '-',
                '环比变化': '-',
                '风险等级': '✅ 正常'
            })
        
        # 创建DataFrame
        anomaly_df = pd.DataFrame(anomaly_data)
        
        # 生成文字报告
        total_anomalies = len(all_anomalies)
        
        report = f"""# AB对比分析报告

## 📊 样本信息
- **样本A**: {sample_a_desc} (共{len(sample_a):,}条记录)
- **样本B**: {sample_b_desc} (共{len(sample_b):,}条记录)
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📈 分析结果统计
- **地区分布异常**: {len(region_anomalies)}个
- **渠道结构异常**: {len(channel_anomalies)}个
- **人群结构异常**: {len(demographic_anomalies)}个
- **总异常数量**: {total_anomalies}个

## 🎯 综合评估
"""
        
        if total_anomalies == 0:
            report += "✅ **整体评估**: 两个样本之间未发现显著异常，结构分布基本一致。\n"
        elif total_anomalies <= 5:
            report += f"⚠️ **整体评估**: 发现{total_anomalies}个异常点，需要关注但整体风险可控。\n"
        else:
            report += f"❌ **整体评估**: 发现{total_anomalies}个异常点，存在较大结构性差异，建议深入分析。\n"
        
        report += "\n## 💡 建议措施\n\n"
        if region_anomalies:
            report += "- **地区分布**: 关注地区分布变化的业务原因，确认是否为正常的市场策略调整\n"
        if channel_anomalies:
            report += "- **渠道结构**: 分析渠道变化对业务的影响，评估渠道效率和质量\n"
        if demographic_anomalies:
            report += "- **人群结构**: 关注目标客群的变化，调整营销策略和产品定位\n"
        
        if total_anomalies == 0:
            report += "- **持续监控**: 建议定期进行AB对比分析，及时发现潜在异常\n"
        
        return report, anomaly_df

# 创建分析器实例
analyzer = ABComparisonAnalyzer()

def run_ab_analysis(start_date_a, end_date_a, refund_start_date_a, refund_end_date_a,
                   pre_vehicle_model_types_a, parent_regions_a, vehicle_types_a, 
                   refund_only_a, locked_only_a,
                   start_date_b, end_date_b, refund_start_date_b, refund_end_date_b,
                   pre_vehicle_model_types_b, parent_regions_b, vehicle_types_b,
                   refund_only_b, locked_only_b):
    """执行AB对比分析"""
    try:
        # 筛选样本A
        sample_a = analyzer.filter_sample(
            start_date=start_date_a, end_date=end_date_a,
            refund_start_date=refund_start_date_a, refund_end_date=refund_end_date_a,
            pre_vehicle_model_types=pre_vehicle_model_types_a if pre_vehicle_model_types_a else None,
            parent_regions=parent_regions_a if parent_regions_a else None,
            vehicle_groups=vehicle_types_a if vehicle_types_a else None,
            refund_only=refund_only_a,
            locked_only=locked_only_a
        )
        sample_a_desc = f"{start_date_a}至{end_date_a}, 车型:{','.join(vehicle_types_a) if vehicle_types_a else '全部'}, {'仅退订' if refund_only_a else ''}{'仅锁单' if locked_only_a else ''}"
        
        # 筛选样本B
        sample_b = analyzer.filter_sample(
            start_date=start_date_b, end_date=end_date_b,
            refund_start_date=refund_start_date_b, refund_end_date=refund_end_date_b,
            pre_vehicle_model_types=pre_vehicle_model_types_b if pre_vehicle_model_types_b else None,
            parent_regions=parent_regions_b if parent_regions_b else None,
            vehicle_groups=vehicle_types_b if vehicle_types_b else None,
            refund_only=refund_only_b,
            locked_only=locked_only_b
        )
        sample_b_desc = f"{start_date_b}至{end_date_b}, 车型:{','.join(vehicle_types_b) if vehicle_types_b else '全部'}, {'仅退订' if refund_only_b else ''}{'仅锁单' if locked_only_b else ''}"
        
        if len(sample_a) == 0:
            empty_df = pd.DataFrame({'错误': ['样本A数据为空，请调整筛选条件']})
            return "❌ 样本A数据为空，请调整筛选条件", empty_df
        
        if len(sample_b) == 0:
            empty_df = pd.DataFrame({'错误': ['样本B数据为空，请调整筛选条件']})
            return "❌ 样本B数据为空，请调整筛选条件", empty_df
        
        # 获取Parent Region筛选条件（取两个样本的交集）
        parent_regions_filter = None
        if parent_regions_a and parent_regions_b:
            # 如果两个样本都有Parent Region筛选，取交集
            parent_regions_filter = list(set(parent_regions_a) & set(parent_regions_b))
        elif parent_regions_a:
            parent_regions_filter = parent_regions_a
        elif parent_regions_b:
            parent_regions_filter = parent_regions_b
        
        # 生成对比报告
        report, anomaly_df = analyzer.generate_comparison_report(sample_a, sample_b, sample_a_desc, sample_b_desc, parent_regions_filter)
        
        return report, anomaly_df
        
    except Exception as e:
        logger.error(f"AB对比分析失败: {str(e)}")
        error_df = pd.DataFrame({'错误': [f"分析失败: {str(e)}"]})
        return f"❌ 分析失败: {str(e)}", error_df

# 获取数据信息
vehicle_types = analyzer.get_vehicle_types()
pre_vehicle_model_types = analyzer.get_pre_vehicle_model_types()
parent_regions = analyzer.get_parent_regions()
min_date, max_date = analyzer.get_date_range()
refund_min_date, refund_max_date = analyzer.get_refund_date_range()

# 创建Gradio界面
with gr.Blocks(title="AB对比分析工具", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 AB对比分析工具")
    gr.Markdown("用于对比两个样本之间的差异，发现地区分布、渠道结构和人群结构异常")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📊 样本A配置")
            
            with gr.Group():
                gr.Markdown("### 小订时间范围")
                with gr.Row():
                    start_date_a = gr.Textbox(label="开始日期", value=min_date, placeholder="YYYY-MM-DD")
                    end_date_a = gr.Textbox(label="结束日期", value=max_date, placeholder="YYYY-MM-DD")
            
            with gr.Group():
                gr.Markdown("### 退订时间范围")
                with gr.Row():
                    refund_start_date_a = gr.Textbox(label="退订开始日期", value="", placeholder="YYYY-MM-DD（可选）")
                    refund_end_date_a = gr.Textbox(label="退订结束日期", value="", placeholder="YYYY-MM-DD（可选）")
            
            pre_vehicle_model_types_a = gr.CheckboxGroup(choices=pre_vehicle_model_types, label="pre_vehicle_model_type", value=[])
            parent_regions_a = gr.Dropdown(choices=parent_regions, label="Parent Region Name", multiselect=True, value=None)
            vehicle_types_a = gr.CheckboxGroup(choices=vehicle_types, label="车型选择", value=[])
            refund_only_a = gr.Checkbox(label="仅退订数据", value=False)
            locked_only_a = gr.Checkbox(label="仅锁单数据", value=False)
        
        with gr.Column(scale=1):
            gr.Markdown("## 📊 样本B配置")
            
            with gr.Group():
                gr.Markdown("### 小订时间范围")
                with gr.Row():
                    start_date_b = gr.Textbox(label="开始日期", value=min_date, placeholder="YYYY-MM-DD")
                    end_date_b = gr.Textbox(label="结束日期", value=max_date, placeholder="YYYY-MM-DD")
            
            with gr.Group():
                gr.Markdown("### 退订时间范围")
                with gr.Row():
                    refund_start_date_b = gr.Textbox(label="退订开始日期", value="", placeholder="YYYY-MM-DD（可选）")
                    refund_end_date_b = gr.Textbox(label="退订结束日期", value="", placeholder="YYYY-MM-DD（可选）")
            
            pre_vehicle_model_types_b = gr.CheckboxGroup(choices=pre_vehicle_model_types, label="pre_vehicle_model_type", value=[])
            parent_regions_b = gr.Dropdown(choices=parent_regions, label="Parent Region Name", multiselect=True, value=None)
            vehicle_types_b = gr.CheckboxGroup(choices=vehicle_types, label="车型选择", value=[])
            refund_only_b = gr.Checkbox(label="仅退订数据", value=False)
            locked_only_b = gr.Checkbox(label="仅锁单数据", value=False)
    
    with gr.Row():
        analyze_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column(scale=1):
            output = gr.Markdown(label="分析结果")
        with gr.Column(scale=2):
            anomaly_table = gr.DataFrame(
                label="异常数据详情",
                interactive=False,
                wrap=True
            )
    
    # 绑定分析函数
    analyze_btn.click(
        fn=run_ab_analysis,
        inputs=[
            start_date_a, end_date_a, refund_start_date_a, refund_end_date_a,
            pre_vehicle_model_types_a, parent_regions_a, vehicle_types_a,
            refund_only_a, locked_only_a,
            start_date_b, end_date_b, refund_start_date_b, refund_end_date_b,
            pre_vehicle_model_types_b, parent_regions_b, vehicle_types_b,
            refund_only_b, locked_only_b
        ],
        outputs=[output, anomaly_table]
    )
    
    # 添加使用说明
    with gr.Accordion("📖 使用说明", open=False):
        gr.Markdown("""
        ### 功能说明
        1. **地区分布异常检测**: 对比两个样本在不同地区的订单分布差异
        2. **渠道结构异常检测**: 分析渠道销量占比的变化情况
        3. **人群结构异常检测**: 检测性别比例和年龄段结构的差异
        
        ### 使用步骤
        1. 配置样本A的筛选条件（时间范围、车型、是否包含退订）
        2. 配置样本B的筛选条件
        3. 点击"开始分析"按钮
        4. 查看分析结果和建议措施
        
        ### 异常检测阈值（"或"逻辑）
        **地区分布异常**：
        - 结构占比异常：占比>1%且变化幅度>10% OR
        - 环比变化异常：绝对值环比变化>50%（包括100%增长）

        **渠道结构异常**：
        - 结构占比异常：占比>1%且变化幅度>15% OR
        - 环比变化异常：绝对值环比变化>50%（包括100%增长）

        **人群结构异常**：
        - 性别分布：变化幅度>10% OR 绝对值环比变化>50%
        - 年龄分布：占比>1%且变化幅度>15% OR 绝对值环比变化>50%
        """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)