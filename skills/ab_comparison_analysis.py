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
# 兼容直接运行与包运行两种方式的导入
try:
    from id_card_validator import validate_id_card
except Exception:
    try:
        from skills.id_card_validator import validate_id_card
    except Exception:
        from .id_card_validator import validate_id_card

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
        """获取产品分类列表：基于Product Name分为增程和纯电"""
        return ["增程", "纯电"]
    
    def get_battery_types(self) -> List[str]:
        """获取电池类型分类列表：基于business_definition.json中的battery_types定义"""
        try:
            with open("/Users/zihao_/Documents/github/W35_workflow/business_definition.json", "r", encoding="utf-8") as f:
                business_def = json.load(f)
                
            if "battery_types" in business_def:
                # 返回电池类型分类（A_LFP, B_NCM, C_EXCLUDE）
                return list(business_def["battery_types"].keys())
            return []
        except Exception as e:
            logger.error(f"获取电池类型分类失败: {str(e)}")
            return []
            
    def get_battery_type_mapping(self) -> Dict[str, str]:
        """获取车型到电池类型的映射"""
        try:
            with open("/Users/zihao_/Documents/github/W35_workflow/business_definition.json", "r", encoding="utf-8") as f:
                business_def = json.load(f)
            
            if "battery_types" not in business_def:
                return {}
                
            # 创建车型到电池类型的映射
            model_to_battery_type = {}
            for battery_type, models in business_def["battery_types"].items():
                for model in models:
                    model_to_battery_type[model] = battery_type
                    
            return model_to_battery_type
        except Exception as e:
            logger.error(f"获取电池类型映射失败: {str(e)}")
            return {}
    
    def get_parent_regions(self) -> List[str]:
        """获取Parent Region Name列表"""
        if 'Parent Region Name' in self.df.columns:
            return sorted(self.df['Parent Region Name'].dropna().unique().tolist())
        return []
    
    def get_date_range(self) -> Tuple[str, str]:
        """获取订单创建时间的日期范围"""
        if 'Order_Create_Time' in self.df.columns:
            create_data = self.df['Order_Create_Time'].dropna()
            if not create_data.empty:
                min_date = create_data.min().strftime('%Y-%m-%d')
                max_date = create_data.max().strftime('%Y-%m-%d')
                return min_date, max_date
        # 如果Order_Create_Time不存在或为空，使用Intention_Payment_Time作为备选
        min_date = self.df['Intention_Payment_Time'].min().strftime('%Y-%m-%d')
        max_date = self.df['Intention_Payment_Time'].max().strftime('%Y-%m-%d')
        return min_date, max_date
    
    def get_refund_date_range(self) -> Tuple[str, str]:
        """获取退订时间范围"""
        if 'intention_refund_time' in self.df.columns:
            refund_data = self.df['intention_refund_time'].dropna()
            if not refund_data.empty:
                min_date = refund_data.min().strftime('%Y-%m-%d')
                max_date = refund_data.max().strftime('%Y-%m-%d')
                return min_date, max_date
        return '', ''
    
    def get_order_create_date_range(self) -> Tuple[str, str]:
        """获取订单创建时间范围"""
        if 'Order_Create_Time' in self.df.columns:
            create_data = self.df['Order_Create_Time'].dropna()
            if not create_data.empty:
                min_date = create_data.min().strftime('%Y-%m-%d')
                max_date = create_data.max().strftime('%Y-%m-%d')
                return min_date, max_date
        return '', ''
    
    def get_lock_date_range(self) -> Tuple[str, str]:
        """获取锁单时间范围，如果没有锁单数据则返回默认范围"""
        if 'Lock_Time' in self.df.columns:
            lock_data = self.df['Lock_Time'].dropna()
            if not lock_data.empty:
                min_date = lock_data.min().strftime('%Y-%m-%d')
                max_date = lock_data.max().strftime('%Y-%m-%d')
                return min_date, max_date
        
        # 如果没有锁单数据，返回基于订单创建时间的默认范围
        order_min, order_max = self.get_date_range()
        return order_min, order_max
    
    def filter_sample(self, start_date: str = '', end_date: str = '', vehicle_types: List[str] = None, 
                     include_refund: bool = False, refund_start_date: str = '', refund_end_date: str = '',
                     pre_vehicle_model_types: List[str] = None, parent_regions: List[str] = None,
                     vehicle_groups: List[str] = None, refund_only: bool = False, 
                     locked_only: bool = False, order_create_start_date: str = '', order_create_end_date: str = '',
                     lock_start_date: str = '', lock_end_date: str = '',
                     exclude_refund: bool = False, exclude_locked: bool = False,
                     include_invalid_id: bool = True,
                     battery_types: List[str] = None, repeat_buyer_only: bool = False, exclude_repeat_buyer: bool = False) -> pd.DataFrame:
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
        
        # 3. 订单创建时间范围筛选
        if order_create_start_date and order_create_end_date and 'Order_Create_Time' in self.df.columns:
            create_mask = (self.df['Order_Create_Time'] >= order_create_start_date) & \
                         (self.df['Order_Create_Time'] <= order_create_end_date)
            mask = mask & create_mask
        
        # 4. 锁单时间范围筛选
        if lock_start_date and lock_end_date and 'Lock_Time' in self.df.columns:
            lock_mask = (self.df['Lock_Time'] >= lock_start_date) & \
                       (self.df['Lock_Time'] <= lock_end_date)
            mask = mask & lock_mask
        
        # 5. 产品分类筛选（基于Product Name）
        if pre_vehicle_model_types and 'Product Name' in self.df.columns:
            product_mask = pd.Series([False] * len(self.df), index=self.df.index)
            
            for category in pre_vehicle_model_types:
                if category == "增程":
                    # 产品名称中包含"新一代"和数字52或66的为增程
                    category_mask = (
                        self.df['Product Name'].str.contains('新一代', na=False) & 
                        (self.df['Product Name'].str.contains('52', na=False) | 
                         self.df['Product Name'].str.contains('66', na=False))
                    )
                elif category == "纯电":
                    # 其他产品为纯电
                    category_mask = ~(
                        self.df['Product Name'].str.contains('新一代', na=False) & 
                        (self.df['Product Name'].str.contains('52', na=False) | 
                         self.df['Product Name'].str.contains('66', na=False))
                    )
                else:
                    category_mask = pd.Series([False] * len(self.df), index=self.df.index)
                
                product_mask = product_mask | category_mask
            
            mask = mask & product_mask
        
        # 6. Parent Region Name筛选
        # 5.1 身份证号异常检测（仅在提供 Buyer Identity No 字段时生效）
        if (not include_invalid_id) and ('Buyer Identity No' in self.df.columns):
            id_series = self.df['Buyer Identity No']
            # 检测异常身份证号：空值、长度不足18位、校验失败
            def is_valid_id_card(id_val):
                if pd.isna(id_val):
                    return False  # 空值视为异常
                id_str = str(id_val).strip()
                if id_str == '' or len(id_str) != 18:
                    return False  # 空字符串或长度不足18位视为异常
                return validate_id_card(id_str)  # 校验失败视为异常
            
            # 保留正常身份证号，剔除异常身份证号
            validity_mask = id_series.apply(is_valid_id_card)
            mask = mask & validity_mask
        
        if parent_regions and 'Parent Region Name' in self.df.columns:
            mask = mask & (self.df['Parent Region Name'].isin(parent_regions))
        
        # 7. 车型分组筛选
        if vehicle_groups and '车型分组' in self.df.columns:
            mask = mask & (self.df['车型分组'].isin(vehicle_groups))
        elif vehicle_types and '车型分组' in self.df.columns:  # 保持向后兼容
            mask = mask & (self.df['车型分组'].isin(vehicle_types))
            
        # 8. 电池类型筛选
        if battery_types and 'Product Name' in self.df.columns:
            battery_type_mapping = self.get_battery_type_mapping()
            if battery_type_mapping:
                # 创建一个临时列来标记电池类型
                temp_df = self.df.copy()
                temp_df['battery_type'] = temp_df['Product Name'].map(battery_type_mapping)
                # 筛选指定电池类型的数据
                battery_mask = temp_df['battery_type'].isin(battery_types)
                mask = mask & battery_mask
        
        sample_data = self.df[mask].copy()
        
        # 8. 是否退订筛选
        if refund_only and 'intention_refund_time' in self.df.columns:
            sample_data = sample_data[sample_data['intention_refund_time'].notna()]
        elif include_refund and 'intention_refund_time' in self.df.columns:  # 保持向后兼容
            sample_data = sample_data[sample_data['intention_refund_time'].notna()]
        
        # 9. 是否锁单筛选
        if locked_only and 'Lock_Time' in self.df.columns:
            sample_data = sample_data[sample_data['Lock_Time'].notna()]
        
        # 10. 排除退订数据
        if exclude_refund and 'intention_refund_time' in self.df.columns:
            sample_data = sample_data[sample_data['intention_refund_time'].isna()]
        
        # 11. 排除锁单数据
        if exclude_locked and 'Lock_Time' in self.df.columns:
            sample_data = sample_data[sample_data['Lock_Time'].isna()]
        
        # 12. 复购用户筛选
        if (repeat_buyer_only or exclude_repeat_buyer) and 'Buyer Identity No' in self.df.columns and 'Invoice_Upload_Time' in self.df.columns:
            # 检查互斥性：如果同时设置两个选项，返回空结果
            if repeat_buyer_only and exclude_repeat_buyer:
                # 同时设置两个选项时返回空DataFrame
                sample_data = sample_data.iloc[0:0]  # 返回空DataFrame但保持列结构
            else:
                # 复购用户的判断标准：
                # 1. 一个买家有多个订单（基于身份证号）
                # 2. 且这批订单中含有"Invoice_Upload_Time"
                # 3. 并且，该"Invoice_Upload_Time"还应该<用户控件选择的"锁单开始日期"
                
                # 获取锁单开始日期，如果没有提供则使用当前筛选的开始日期
                reference_date = lock_start_date if lock_start_date else start_date
                
                if reference_date:
                    # 找出复购用户的身份证号
                    repeat_buyer_ids = set()
                    
                    # 按身份证号分组，找出有多个订单的买家
                    buyer_groups = self.df.groupby('Buyer Identity No')
                    for buyer_id, group in buyer_groups:
                        if len(group) > 1:  # 有多个订单
                            # 检查是否有Invoice_Upload_Time且早于参考日期
                            invoice_times = group['Invoice_Upload_Time'].dropna()
                            if len(invoice_times) > 0:
                                # 检查是否有Invoice_Upload_Time早于参考日期
                                early_invoices = invoice_times[invoice_times < reference_date]
                                if len(early_invoices) > 0:
                                    repeat_buyer_ids.add(buyer_id)
                    
                    # 根据选择的模式进行筛选
                    if repeat_buyer_only and repeat_buyer_ids:
                        # 仅保留复购用户
                        sample_data = sample_data[sample_data['Buyer Identity No'].isin(repeat_buyer_ids)]
                    elif exclude_repeat_buyer and repeat_buyer_ids:
                        # 排除复购用户
                        sample_data = sample_data[~sample_data['Buyer Identity No'].isin(repeat_buyer_ids)]
        
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
                                 'change': ratio_a - ratio_b,
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
                             'change': ratio_a - ratio_b,
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
                         'change': ratio_a - ratio_b,
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
                         'change': ratio_a - ratio_b,
                         'relative_change': relative_change
                     })
        
        return anomalies
    
    def analyze_sales_agent_comparison(self, sample_a: pd.DataFrame, sample_b: pd.DataFrame) -> List[Dict]:
        """销售代理分析对比"""
        try:
            # 读取销售代理数据
            sales_info_path = "/Users/zihao_/Documents/coding/dataset/formatted/sales_info_data.json"
            try:
                with open(sales_info_path, 'r', encoding='utf-8') as f:
                    import json
                    sales_info_json = json.load(f)
                    sales_info_data = sales_info_json.get('data', [])
            except Exception as e:
                logger.warning(f"无法读取销售代理数据文件: {str(e)}")
                return []
            
            # 构建销售代理查找集合
            sales_agents_lookup = set()
            if isinstance(sales_info_data, list):
                for item in sales_info_data:
                    if isinstance(item, dict):
                        member_name = str(item.get('Member Name', '')).strip() if item.get('Member Name') else ''
                        member_code = str(item.get('Member Code', '')).strip() if item.get('Member Code') else ''
                        id_card = str(item.get('Id Card', '')).strip() if item.get('Id Card') else ''
                        
                        if member_name and member_code and id_card:
                            sales_agents_lookup.add((member_name, member_code, id_card))
            
            def analyze_sample_sales_agent(sample_df):
                if len(sample_df) == 0:
                    return {'total_orders': 0, 'total_unique_buyers': 0, 'agent_orders': 0, 'agent_ratio': 0.0, 'repeat_buyer_orders': 0, 'repeat_buyer_ratio': 0.0, 'unique_repeat_buyers': 0, 'repeat_buyer_orders_combo': 0, 'repeat_buyer_ratio_combo': 0.0, 'unique_repeat_buyers_combo': 0}
                
                # 预处理字段
                sample_df = sample_df.copy()
                sample_df['clean_store_agent_name'] = sample_df['Store Agent Name'].fillna('').astype(str).str.strip()
                sample_df['clean_store_agent_id'] = sample_df['Store Agent Id'].fillna('').astype(str).str.strip()
                sample_df['clean_buyer_identity'] = sample_df['Buyer Identity No'].fillna('').astype(str).str.strip()
                
                # 创建组合字段用于匹配
                agent_combos = list(zip(
                    sample_df['clean_store_agent_name'],
                    sample_df['clean_store_agent_id'], 
                    sample_df['clean_buyer_identity']
                ))
                
                # 计算匹配的订单数
                matched_combos = set(agent_combos) & sales_agents_lookup
                agent_orders = sum(1 for combo in agent_combos if combo in matched_combos)
                
                total_orders = len(sample_df)
                agent_ratio = agent_orders / total_orders if total_orders > 0 else 0.0
                
                # 重复买家分析 - 口径1：仅基于身份证号
                buyer_identity_counts = sample_df['Buyer Identity No'].value_counts()
                repeat_buyers = buyer_identity_counts[buyer_identity_counts >= 2]
                repeat_buyer_orders = repeat_buyers.sum()
                repeat_buyer_ratio = repeat_buyer_orders / total_orders if total_orders > 0 else 0.0
                unique_repeat_buyers = len(repeat_buyers)
                
                # 计算总买家数量（基于身份证号）
                total_unique_buyers = sample_df['Buyer Identity No'].nunique()
                
                # 重复买家分析 - 口径2：身份证号+手机号双重匹配
                # 创建身份证号+手机号的组合键
                sample_df_clean = sample_df.dropna(subset=['Buyer Identity No', 'Buyer Cell Phone'])
                buyer_combo_key = sample_df_clean['Buyer Identity No'].astype(str) + '_' + sample_df_clean['Buyer Cell Phone'].astype(str)
                buyer_combo_counts = buyer_combo_key.value_counts()
                repeat_buyers_combo = buyer_combo_counts[buyer_combo_counts >= 2]
                repeat_buyer_orders_combo = repeat_buyers_combo.sum()
                repeat_buyer_ratio_combo = repeat_buyer_orders_combo / total_orders if total_orders > 0 else 0.0
                unique_repeat_buyers_combo = len(repeat_buyers_combo)
                
                return {
                    'total_orders': total_orders,
                    'total_unique_buyers': total_unique_buyers,
                    'agent_orders': agent_orders,
                    'agent_ratio': agent_ratio,
                    'repeat_buyer_orders': repeat_buyer_orders,
                    'repeat_buyer_ratio': repeat_buyer_ratio,
                    'unique_repeat_buyers': unique_repeat_buyers,
                    'repeat_buyer_orders_combo': repeat_buyer_orders_combo,
                    'repeat_buyer_ratio_combo': repeat_buyer_ratio_combo,
                    'unique_repeat_buyers_combo': unique_repeat_buyers_combo
                }
            
            # 分析两个样本
            result_a = analyze_sample_sales_agent(sample_a)
            result_b = analyze_sample_sales_agent(sample_b)
            
            return [{
                'type': '销售代理分析',
                'sample_a': result_a,
                'sample_b': result_b
            }]
            
        except Exception as e:
            logger.error(f"销售代理分析失败: {str(e)}")
            return []
    
    def analyze_time_interval_comparison(self, sample_a: pd.DataFrame, sample_b: pd.DataFrame) -> List[Dict]:
        """时间间隔分析对比"""
        try:
            def analyze_sample_time_intervals(sample_df):
                if len(sample_df) == 0:
                    return {'payment_to_refund': {}, 'payment_to_assign': {}, 'payment_to_lock': {}}
                
                sample_df = sample_df.copy()
                
                # 确保时间列为datetime类型
                time_columns = ['Intention_Payment_Time', 'intention_refund_time', 'first_assign_time', 'Lock_Time']
                for col in time_columns:
                    if col in sample_df.columns:
                        sample_df[col] = pd.to_datetime(sample_df[col], errors='coerce')
                
                # 计算支付到退款的时间间隔
                payment_to_refund_stats = {}
                if 'intention_refund_time' in sample_df.columns:
                    valid_refund_mask = sample_df['intention_refund_time'].notna() & sample_df['Intention_Payment_Time'].notna()
                    if valid_refund_mask.any():
                        intervals = (sample_df.loc[valid_refund_mask, 'intention_refund_time'] - 
                                   sample_df.loc[valid_refund_mask, 'Intention_Payment_Time']).dt.days
                        if len(intervals) > 0:
                            payment_to_refund_stats = {
                                'count': len(intervals),
                                'mean': float(intervals.mean()),
                                'median': float(intervals.median()),
                                'std': float(intervals.std()) if len(intervals) > 1 else 0.0
                            }
                
                # 计算支付到分配的时间间隔
                payment_to_assign_stats = {}
                if 'first_assign_time' in sample_df.columns:
                    valid_assign_mask = sample_df['first_assign_time'].notna() & sample_df['Intention_Payment_Time'].notna()
                    if valid_assign_mask.any():
                        intervals = (sample_df.loc[valid_assign_mask, 'first_assign_time'] - 
                                   sample_df.loc[valid_assign_mask, 'Intention_Payment_Time']).dt.days
                        if len(intervals) > 0:
                            payment_to_assign_stats = {
                                'count': len(intervals),
                                'mean': float(intervals.mean()),
                                'median': float(intervals.median()),
                                'std': float(intervals.std()) if len(intervals) > 1 else 0.0
                            }
                
                # 计算支付到锁单（lock time）的时间间隔
                payment_to_lock_stats = {}
                if 'Lock_Time' in sample_df.columns:
                    valid_lock_mask = sample_df['Lock_Time'].notna() & sample_df['Intention_Payment_Time'].notna()
                    if valid_lock_mask.any():
                        intervals = (sample_df.loc[valid_lock_mask, 'Lock_Time'] - 
                                   sample_df.loc[valid_lock_mask, 'Intention_Payment_Time']).dt.days
                        if len(intervals) > 0:
                            payment_to_lock_stats = {
                                'count': len(intervals),
                                'mean': float(intervals.mean()),
                                'median': float(intervals.median()),
                                'std': float(intervals.std()) if len(intervals) > 1 else 0.0
                            }
                
                return {
                    'payment_to_refund': payment_to_refund_stats,
                    'payment_to_assign': payment_to_assign_stats,
                    'payment_to_lock': payment_to_lock_stats
                }
            
            # 分析两个样本
            result_a = analyze_sample_time_intervals(sample_a)
            result_b = analyze_sample_time_intervals(sample_b)
            
            return [{
                'type': '时间间隔分析',
                'sample_a': result_a,
                'sample_b': result_b
            }]
            
        except Exception as e:
            logger.error(f"时间间隔分析失败: {str(e)}")
            return []
    
    def generate_comparison_report(self, sample_a: pd.DataFrame, sample_b: pd.DataFrame, 
                                 sample_a_desc: str, sample_b_desc: str, 
                                 parent_regions_filter: List[str] = None,
                                 sample_a_label: str = "样本A",
                                 sample_b_label: str = "样本B") -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """生成对比分析报告"""
        # 执行三种异常检测
        region_anomalies = self.analyze_region_distribution(sample_a, sample_b, parent_regions_filter)
        channel_anomalies = self.analyze_channel_structure(sample_a, sample_b)
        demographic_anomalies = self.analyze_demographic_structure(sample_a, sample_b)
        
        # 执行销售代理分析和时间间隔分析
        sales_agent_results = self.analyze_sales_agent_comparison(sample_a, sample_b)
        time_interval_results = self.analyze_time_interval_comparison(sample_a, sample_b)

        # 动态样本名称用于列名和展示（已从调用方传入），若为空则回退到默认
        sample_a_label = sample_a_label or "样本A"
        sample_b_label = sample_b_label or "样本B"
        
        # 合并所有异常数据
        all_anomalies = region_anomalies + channel_anomalies + demographic_anomalies
        
        # 创建异常检测结果表格
        anomaly_data = []
        
        for anomaly in all_anomalies:
            # 为占比变化添加颜色标识
            change_value = anomaly['change']
            if change_value > 0:
                change_display = f"<span style='color: red;'>+{change_value:.2%}</span>"
            elif change_value < 0:
                change_display = f"<span style='color: green;'>{change_value:.2%}</span>"
            else:
                change_display = f"{change_value:.2%}"
            
            # 为环比变化添加颜色标识
            relative_change_value = anomaly['relative_change']
            if relative_change_value == float('inf'):
                relative_change_display = "新增"
            elif relative_change_value > 0:
                relative_change_display = f"<span style='color: red;'>+{relative_change_value:.1%}</span>"
            elif relative_change_value < 0:
                relative_change_display = f"<span style='color: green;'>{relative_change_value:.1%}</span>"
            else:
                relative_change_display = f"{relative_change_value:.1%}"
            
            anomaly_data.append({
                '异常类型': anomaly['type'],
                '异常项目': anomaly['item'],
                '异常子类': anomaly['anomaly_type'],
                f"{sample_a_label}绝对值": f"{anomaly['sample_a_abs']:,}",
                f"{sample_b_label}绝对值": f"{anomaly['sample_b_abs']:,}",
                f"{sample_a_label}占比": f"{anomaly['sample_a_ratio']:.2%}",
                f"{sample_b_label}占比": f"{anomaly['sample_b_ratio']:.2%}",
                '占比变化': change_display,
                '环比变化': relative_change_display,
                '风险等级': '⚠️ 中等'
            })
        
        # 如果没有异常，添加正常状态
        if not anomaly_data:
            anomaly_data.append({
                '异常类型': '整体评估',
                '异常项目': '无异常',
                '异常子类': '正常',
                f"{sample_a_label}绝对值": '-',
                f"{sample_b_label}绝对值": '-',
                f"{sample_a_label}占比": '-',
                f"{sample_b_label}占比": '-',
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
- **{sample_a_desc}** (共{len(sample_a):,}条记录)
- **{sample_b_desc}** (共{len(sample_b):,}条记录)
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
        
        # 生成销售代理分析对比表格
        sales_agent_data = []
        if sales_agent_results:
            result = sales_agent_results[0]
            sample_a_result = result['sample_a']
            sample_b_result = result['sample_b']
            
            sales_agent_data.append({
                '指标': '总订单数',
                sample_a_label: f"{sample_a_result['total_orders']:,}",
                sample_b_label: f"{sample_b_result['total_orders']:,}",
                '差异': f"{sample_a_result['total_orders'] - sample_b_result['total_orders']:+,}"
            })
            
            sales_agent_data.append({
                '指标': '买家数量',
                sample_a_label: f"{sample_a_result['total_unique_buyers']:,}",
                sample_b_label: f"{sample_b_result['total_unique_buyers']:,}",
                '差异': f"{sample_a_result['total_unique_buyers'] - sample_b_result['total_unique_buyers']:+,}"
            })
            
            sales_agent_data.append({
                '指标': '销售代理订单数',
                sample_a_label: f"{sample_a_result['agent_orders']:,}",
                sample_b_label: f"{sample_b_result['agent_orders']:,}",
                '差异': f"{sample_a_result['agent_orders'] - sample_b_result['agent_orders']:+,}"
            })
            
            sales_agent_data.append({
                '指标': '销售代理订单比例',
                sample_a_label: f"{sample_a_result['agent_ratio']:.2%}",
                sample_b_label: f"{sample_b_result['agent_ratio']:.2%}",
                '差异': f"{sample_a_result['agent_ratio'] - sample_b_result['agent_ratio']:+.2%}"
            })
            
            sales_agent_data.append({
                '指标': '重复买家订单数',
                sample_a_label: f"{sample_a_result['repeat_buyer_orders']:,}",
                sample_b_label: f"{sample_b_result['repeat_buyer_orders']:,}",
                '差异': f"{sample_a_result['repeat_buyer_orders'] - sample_b_result['repeat_buyer_orders']:+,}"
            })
            
            sales_agent_data.append({
                '指标': '重复买家订单比例',
                sample_a_label: f"{sample_a_result['repeat_buyer_ratio']:.2%}",
                sample_b_label: f"{sample_b_result['repeat_buyer_ratio']:.2%}",
                '差异': f"{sample_a_result['repeat_buyer_ratio'] - sample_b_result['repeat_buyer_ratio']:+.2%}"
            })
            
            sales_agent_data.append({
                '指标': '重复买家数量',
                sample_a_label: f"{sample_a_result['unique_repeat_buyers']:,}",
                sample_b_label: f"{sample_b_result['unique_repeat_buyers']:,}",
                '差异': f"{sample_a_result['unique_repeat_buyers'] - sample_b_result['unique_repeat_buyers']:+,}"
            })
            
            # 新增：身份证号+手机号双重匹配的重复买家指标
            sales_agent_data.append({
                '指标': '重复买家订单数(身份证+手机)',
                sample_a_label: f"{sample_a_result['repeat_buyer_orders_combo']:,}",
                sample_b_label: f"{sample_b_result['repeat_buyer_orders_combo']:,}",
                '差异': f"{sample_a_result['repeat_buyer_orders_combo'] - sample_b_result['repeat_buyer_orders_combo']:+,}"
            })
            
            sales_agent_data.append({
                '指标': '重复买家订单比例(身份证+手机)',
                sample_a_label: f"{sample_a_result['repeat_buyer_ratio_combo']:.2%}",
                sample_b_label: f"{sample_b_result['repeat_buyer_ratio_combo']:.2%}",
                '差异': f"{sample_a_result['repeat_buyer_ratio_combo'] - sample_b_result['repeat_buyer_ratio_combo']:+.2%}"
            })
            
            sales_agent_data.append({
                '指标': '重复买家数量(身份证+手机)',
                sample_a_label: f"{sample_a_result['unique_repeat_buyers_combo']:,}",
                sample_b_label: f"{sample_b_result['unique_repeat_buyers_combo']:,}",
                '差异': f"{sample_a_result['unique_repeat_buyers_combo'] - sample_b_result['unique_repeat_buyers_combo']:+,}"
            })
        else:
            sales_agent_data.append({
                '指标': '数据获取失败',
                sample_a_label: '-',
                sample_b_label: '-',
                '差异': '-'
            })
        
        sales_agent_df = pd.DataFrame(sales_agent_data)
        
        # 生成时间间隔分析对比表格
        time_interval_data = []
        if time_interval_results:
            result = time_interval_results[0]
            sample_a_result = result['sample_a']
            sample_b_result = result['sample_b']
            
            # 支付到退款时间间隔
            refund_a = sample_a_result.get('payment_to_refund', {})
            refund_b = sample_b_result.get('payment_to_refund', {})
            
            if refund_a and refund_b:
                time_interval_data.append({
                    '时间间隔类型': '支付到退款-样本数',
                    sample_a_label: f"{refund_a.get('count', 0):,}",
                    sample_b_label: f"{refund_b.get('count', 0):,}",
                    '差异': f"{refund_a.get('count', 0) - refund_b.get('count', 0):+,}"
                })
                
                time_interval_data.append({
                    '时间间隔类型': '支付到退款-平均天数',
                    sample_a_label: f"{refund_a.get('mean', 0):.1f}",
                    sample_b_label: f"{refund_b.get('mean', 0):.1f}",
                    '差异': f"{refund_a.get('mean', 0) - refund_b.get('mean', 0):+.1f}"
                })
                
                time_interval_data.append({
                    '时间间隔类型': '支付到退款-中位数天数',
                    sample_a_label: f"{refund_a.get('median', 0):.1f}",
                    sample_b_label: f"{refund_b.get('median', 0):.1f}",
                    '差异': f"{refund_a.get('median', 0) - refund_b.get('median', 0):+.1f}"
                })
            
            # 支付到分配时间间隔
            assign_a = sample_a_result.get('payment_to_assign', {})
            assign_b = sample_b_result.get('payment_to_assign', {})
            
            if assign_a and assign_b:
                time_interval_data.append({
                    '时间间隔类型': '支付到分配-样本数',
                    sample_a_label: f"{assign_a.get('count', 0):,}",
                    sample_b_label: f"{assign_b.get('count', 0):,}",
                    '差异': f"{assign_a.get('count', 0) - assign_b.get('count', 0):+,}"
                })
                
                time_interval_data.append({
                    '时间间隔类型': '支付到分配-平均天数',
                    sample_a_label: f"{assign_a.get('mean', 0):.1f}",
                    sample_b_label: f"{assign_b.get('mean', 0):.1f}",
                    '差异': f"{assign_a.get('mean', 0) - assign_b.get('mean', 0):+.1f}"
                })
                
                time_interval_data.append({
                    '时间间隔类型': '支付到分配-中位数天数',
                    sample_a_label: f"{assign_a.get('median', 0):.1f}",
                    sample_b_label: f"{assign_b.get('median', 0):.1f}",
                    '差异': f"{assign_a.get('median', 0) - assign_b.get('median', 0):+.1f}"
                })
            
            # 支付到锁单时间间隔
            lock_a = sample_a_result.get('payment_to_lock', {})
            lock_b = sample_b_result.get('payment_to_lock', {})
            
            if lock_a and lock_b:
                time_interval_data.append({
                    '时间间隔类型': '支付到锁单-样本数',
                    sample_a_label: f"{lock_a.get('count', 0):,}",
                    sample_b_label: f"{lock_b.get('count', 0):,}",
                    '差异': f"{lock_a.get('count', 0) - lock_b.get('count', 0):+,}"
                })
                
                time_interval_data.append({
                    '时间间隔类型': '支付到锁单-平均天数',
                    sample_a_label: f"{lock_a.get('mean', 0):.1f}",
                    sample_b_label: f"{lock_b.get('mean', 0):.1f}",
                    '差异': f"{lock_a.get('mean', 0) - lock_b.get('mean', 0):+.1f}"
                })
                
                time_interval_data.append({
                    '时间间隔类型': '支付到锁单-中位数天数',
                    sample_a_label: f"{lock_a.get('median', 0):.1f}",
                    sample_b_label: f"{lock_b.get('median', 0):.1f}",
                    '差异': f"{lock_a.get('median', 0) - lock_b.get('median', 0):+.1f}"
                })
        
        if not time_interval_data:
            time_interval_data.append({
                '时间间隔类型': '无有效数据',
                sample_a_label: '-',
                sample_b_label: '-',
                '差异': '-'
            })
        
        time_interval_df = pd.DataFrame(time_interval_data)
        
        return report, anomaly_df, sales_agent_df, time_interval_df

# 创建分析器实例
analyzer = ABComparisonAnalyzer()

def run_analysis(start_date_a, end_date_a, refund_start_date_a, refund_end_date_a, 
                       order_create_start_date_a, order_create_end_date_a, lock_start_date_a, lock_end_date_a,
                       pre_vehicle_model_types_a, parent_regions_a, vehicle_types_a, include_invalid_id_a, refund_only_a, locked_only_a,
                       exclude_refund_a, exclude_locked_a, battery_types_a, repeat_buyer_only_a, exclude_repeat_buyer_a,
                       start_date_b, end_date_b, refund_start_date_b, refund_end_date_b,
                       order_create_start_date_b, order_create_end_date_b, lock_start_date_b, lock_end_date_b,
                       pre_vehicle_model_types_b, parent_regions_b, vehicle_types_b, include_invalid_id_b, refund_only_b, locked_only_b,
                       exclude_refund_b, exclude_locked_b, battery_types_b, repeat_buyer_only_b, exclude_repeat_buyer_b):
    """执行AB对比分析"""
    try:
        # 筛选样本A
        sample_a = analyzer.filter_sample(
            start_date=start_date_a, end_date=end_date_a,
            refund_start_date=refund_start_date_a, refund_end_date=refund_end_date_a,
            order_create_start_date=order_create_start_date_a, order_create_end_date=order_create_end_date_a,
            lock_start_date=lock_start_date_a, lock_end_date=lock_end_date_a,
            pre_vehicle_model_types=pre_vehicle_model_types_a if pre_vehicle_model_types_a else None,
            parent_regions=parent_regions_a if parent_regions_a else None,
            vehicle_groups=vehicle_types_a if vehicle_types_a else None,
            refund_only=refund_only_a,
            locked_only=locked_only_a,
            exclude_refund=exclude_refund_a,
            exclude_locked=exclude_locked_a,
            include_invalid_id=include_invalid_id_a,
            battery_types=battery_types_a if battery_types_a else None,
            repeat_buyer_only=repeat_buyer_only_a,
            exclude_repeat_buyer=exclude_repeat_buyer_a
        )
        # 动态构建样本A名称（不含时间周期，默认或全部不加入）
        def add_segment(seg_list, title, vals):
            if vals and not ("全部" in vals or "All" in vals):
                seg_list.append(f"{title}:{','.join(vals)}")

        segments_a = []
        add_segment(segments_a, "车型", vehicle_types_a)
        add_segment(segments_a, "产品", pre_vehicle_model_types_a)
        add_segment(segments_a, "电池", battery_types_a)
        add_segment(segments_a, "区域", parent_regions_a)
        if refund_only_a:
            segments_a.append("仅退订")
        if locked_only_a:
            segments_a.append("仅锁单")
        sample_a_label = " | ".join(segments_a)
        sample_a_desc = sample_a_label if sample_a_label else "样本A"
        
        # 筛选样本B
        sample_b = analyzer.filter_sample(
            start_date=start_date_b, end_date=end_date_b,
            refund_start_date=refund_start_date_b, refund_end_date=refund_end_date_b,
            order_create_start_date=order_create_start_date_b, order_create_end_date=order_create_end_date_b,
            lock_start_date=lock_start_date_b, lock_end_date=lock_end_date_b,
            pre_vehicle_model_types=pre_vehicle_model_types_b if pre_vehicle_model_types_b else None,
            parent_regions=parent_regions_b if parent_regions_b else None,
            vehicle_groups=vehicle_types_b if vehicle_types_b else None,
            refund_only=refund_only_b,
            locked_only=locked_only_b,
            exclude_refund=exclude_refund_b,
            exclude_locked=exclude_locked_b,
            include_invalid_id=include_invalid_id_b,
            battery_types=battery_types_b if battery_types_b else None,
            repeat_buyer_only=repeat_buyer_only_b,
            exclude_repeat_buyer=exclude_repeat_buyer_b
        )
        # 动态构建样本B名称（不含时间周期，默认或全部不加入）
        segments_b = []
        add_segment(segments_b, "车型", vehicle_types_b)
        add_segment(segments_b, "产品", pre_vehicle_model_types_b)
        add_segment(segments_b, "电池", battery_types_b)
        add_segment(segments_b, "区域", parent_regions_b)
        if refund_only_b:
            segments_b.append("仅退订")
        if locked_only_b:
            segments_b.append("仅锁单")
        sample_b_label = " | ".join(segments_b)
        sample_b_desc = sample_b_label if sample_b_label else "样本B"
        
        if len(sample_a) == 0:
            name_a = sample_a_label or "样本A"
            empty_df = pd.DataFrame({'错误': [f"{name_a} 数据为空，请调整筛选条件"]})
            return f"❌ {name_a} 数据为空，请调整筛选条件", empty_df, empty_df, empty_df
        
        if len(sample_b) == 0:
            name_b = sample_b_label or "样本B"
            empty_df = pd.DataFrame({'错误': [f"{name_b} 数据为空，请调整筛选条件"]})
            return f"❌ {name_b} 数据为空，请调整筛选条件", empty_df, empty_df, empty_df
        
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
        report, anomaly_df, sales_agent_df, time_interval_df = analyzer.generate_comparison_report(
            sample_a, sample_b, sample_a_desc, sample_b_desc, parent_regions_filter,
            sample_a_label=sample_a_label, sample_b_label=sample_b_label
        )
        
        return report, anomaly_df, sales_agent_df, time_interval_df
        
    except Exception as e:
        logger.error(f"AB对比分析失败: {str(e)}")
        error_df = pd.DataFrame({'错误': [f"分析失败: {str(e)}"]})
        return f"❌ 分析失败: {str(e)}", error_df, error_df, error_df

# 获取数据信息
vehicle_types = analyzer.get_vehicle_types()
pre_vehicle_model_types = analyzer.get_pre_vehicle_model_types()
parent_regions = analyzer.get_parent_regions()
min_date, max_date = analyzer.get_date_range()
refund_min_date, refund_max_date = analyzer.get_refund_date_range()
order_create_min_date, order_create_max_date = analyzer.get_order_create_date_range()
lock_min_date, lock_max_date = analyzer.get_lock_date_range()

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
            
            with gr.Group():
                gr.Markdown("### 订单创建时间范围")
                with gr.Row():
                    order_create_start_date_a = gr.Textbox(label="创建开始日期", value="", placeholder="YYYY-MM-DD（可选）")
                    order_create_end_date_a = gr.Textbox(label="创建结束日期", value="", placeholder="YYYY-MM-DD（可选）")
            
            with gr.Group():
                gr.Markdown("### 锁单时间范围")
                with gr.Row():
                    lock_start_date_a = gr.Textbox(label="锁单开始日期", value=lock_min_date, placeholder="YYYY-MM-DD（可选）")
                    lock_end_date_a = gr.Textbox(label="锁单结束日期", value=lock_max_date, placeholder="YYYY-MM-DD（可选）")
            
            pre_vehicle_model_types_a = gr.CheckboxGroup(choices=pre_vehicle_model_types, label="产品分类（增程/纯电）", value=[])
            
            # 添加电池类型选择组件
            battery_types = analyzer.get_battery_types()
            battery_types_a = gr.CheckboxGroup(choices=battery_types, label="电池类型分类", value=[])
            parent_regions_a = gr.Dropdown(choices=parent_regions, label="Parent Region Name", multiselect=True, value=None)
            vehicle_types_a = gr.CheckboxGroup(choices=vehicle_types, label="车型选择", value=[])
            include_invalid_id_a = gr.Checkbox(label="包含异常身份证号", value=True)
            repeat_buyer_only_a = gr.Checkbox(label="仅复购用户", value=False)
            exclude_repeat_buyer_a = gr.Checkbox(label="排除复购用户", value=False)
            refund_only_a = gr.Checkbox(label="仅退订数据", value=False)
            locked_only_a = gr.Checkbox(label="仅锁单数据", value=False)
            exclude_refund_a = gr.Checkbox(label="排除退订数据", value=False)
            exclude_locked_a = gr.Checkbox(label="排除锁单数据", value=False)
        
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
            
            with gr.Group():
                gr.Markdown("### 订单创建时间范围")
                with gr.Row():
                    order_create_start_date_b = gr.Textbox(label="创建开始日期", value="", placeholder="YYYY-MM-DD（可选）")
                    order_create_end_date_b = gr.Textbox(label="创建结束日期", value="", placeholder="YYYY-MM-DD（可选）")
            
            with gr.Group():
                gr.Markdown("### 锁单时间范围")
                with gr.Row():
                    lock_start_date_b = gr.Textbox(label="锁单开始日期", value=lock_min_date, placeholder="YYYY-MM-DD（可选）")
                    lock_end_date_b = gr.Textbox(label="锁单结束日期", value=lock_max_date, placeholder="YYYY-MM-DD（可选）")
            
            pre_vehicle_model_types_b = gr.CheckboxGroup(choices=pre_vehicle_model_types, label="产品分类（增程/纯电）", value=[])
            
            # 添加电池类型选择组件
            battery_types_b = gr.CheckboxGroup(choices=battery_types, label="电池类型分类", value=[])
            parent_regions_b = gr.Dropdown(choices=parent_regions, label="Parent Region Name", multiselect=True, value=None)
            vehicle_types_b = gr.CheckboxGroup(choices=vehicle_types, label="车型选择", value=[])
            include_invalid_id_b = gr.Checkbox(label="包含异常身份证号", value=True)
            repeat_buyer_only_b = gr.Checkbox(label="仅复购用户", value=False)
            exclude_repeat_buyer_b = gr.Checkbox(label="排除复购用户", value=False)
            refund_only_b = gr.Checkbox(label="仅退订数据", value=False)
            locked_only_b = gr.Checkbox(label="仅锁单数据", value=False)
            exclude_refund_b = gr.Checkbox(label="排除退订数据", value=False)
            exclude_locked_b = gr.Checkbox(label="排除锁单数据", value=False)
    
    with gr.Row():
        analyze_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column(scale=1):
            output = gr.Markdown(label="分析结果")
        with gr.Column(scale=4):
            anomaly_table = gr.DataFrame(
                label="异常数据详情",
                interactive=False,
                wrap=True,
                datatype=["str", "str", "str", "str", "str", "str", "str", "html", "html", "str"]
            )
    
    with gr.Row():
        with gr.Column(scale=1):
            sales_agent_table = gr.DataFrame(
                label="销售代理分析对比",
                interactive=False,
                wrap=True,
                datatype=["str", "str", "str", "html"]
            )
        with gr.Column(scale=1):
            time_interval_table = gr.DataFrame(
                label="时间间隔分析对比",
                interactive=False,
                wrap=True
            )
    
    # 绑定分析函数
    analyze_btn.click(
        fn=run_analysis,
        inputs=[
            start_date_a, end_date_a, refund_start_date_a, refund_end_date_a,
            order_create_start_date_a, order_create_end_date_a, lock_start_date_a, lock_end_date_a,
            pre_vehicle_model_types_a, parent_regions_a, vehicle_types_a, include_invalid_id_a,
            refund_only_a, locked_only_a, exclude_refund_a, exclude_locked_a, battery_types_a, repeat_buyer_only_a, exclude_repeat_buyer_a,
            start_date_b, end_date_b, refund_start_date_b, refund_end_date_b,
            order_create_start_date_b, order_create_end_date_b, lock_start_date_b, lock_end_date_b,
            pre_vehicle_model_types_b, parent_regions_b, vehicle_types_b, include_invalid_id_b,
            refund_only_b, locked_only_b, exclude_refund_b, exclude_locked_b, battery_types_b, repeat_buyer_only_b, exclude_repeat_buyer_b
        ],
        outputs=[output, anomaly_table, sales_agent_table, time_interval_table]
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
    demo.launch(server_name="0.0.0.0", server_port=7866, share=False)