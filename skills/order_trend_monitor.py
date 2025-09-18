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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据路径
DATA_PATH = "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"
BUSINESS_DEF_PATH = "/Users/zihao_/Documents/github/W35_workflow/business_definition.json"

def get_dynamic_vehicle_data(vehicle_name: str) -> pd.DataFrame:
    """动态获取车型的订单数据"""
    try:
        # 加载数据和业务定义
        df = pd.read_parquet(DATA_PATH)
        with open(BUSINESS_DEF_PATH, 'r', encoding='utf-8') as f:
            business_data = json.load(f)
        
        # 适配新的JSON结构
        if 'time_periods' in business_data:
            business_def = business_data['time_periods']
        else:
            business_def = business_data
        
        if vehicle_name not in business_def:
            logger.warning(f"车型 {vehicle_name} 不在业务定义中")
            return pd.DataFrame()
        vehicle_data = df[df['车型分组'] == vehicle_name].copy()
        if len(vehicle_data) == 0:
            logger.warning(f"未找到车型 {vehicle_name} 的数据")
            return pd.DataFrame()
        
        # 计算该车型的最大天数
        start_date = datetime.strptime(business_def[vehicle_name]['start'], '%Y-%m-%d')
        end_date = datetime.strptime(business_def[vehicle_name]['end'], '%Y-%m-%d')
        max_days = (end_date - start_date).days + 1
        
        # 按日期分组统计订单数
        daily_orders = vehicle_data.groupby(vehicle_data['Intention_Payment_Time'].dt.date).size().reset_index()
        daily_orders.columns = ['date', 'daily_orders']
        daily_orders['date'] = pd.to_datetime(daily_orders['date'])
        
        # 计算从预售开始的天数
        def calculate_days_from_start(date):
            if isinstance(date, str):
                date = datetime.strptime(date, '%Y-%m-%d')
            elif hasattr(date, 'to_pydatetime'):
                date = date.to_pydatetime()
            return (date - start_date).days + 1
        
        daily_orders['day'] = daily_orders['date'].apply(calculate_days_from_start)
        
        # 过滤有效天数
        daily_orders = daily_orders[
            (daily_orders['day'] >= 1) & 
            (daily_orders['day'] <= max_days)
        ]
        
        # 计算累计订单数
        daily_orders = daily_orders.sort_values('day')
        daily_orders['cumulative_orders'] = daily_orders['daily_orders'].cumsum()
        
        # 返回所需格式的DataFrame
        result_df = daily_orders[['day', 'daily_orders', 'cumulative_orders']].copy()
        
        logger.info(f"成功获取 {vehicle_name} 的 {len(result_df)} 天数据")
        return result_df
        
    except Exception as e:
        logger.error(f"获取 {vehicle_name} 数据时出错: {str(e)}")
        return pd.DataFrame()

# 动态获取CM1和CM2数据
cm1_df = get_dynamic_vehicle_data('CM1')
cm2_df = get_dynamic_vehicle_data('CM2')

# 如果动态获取失败，使用备用的静态数据
if cm1_df.empty:
    logger.warning("CM1动态数据获取失败，使用静态备用数据")
    cm1_data = [
        (1, 2165, 2165), (2, 1089, 3254), (3, 936, 4190), (4, 573, 4763),
        (5, 502, 5265), (6, 463, 5728), (7, 433, 6161), (8, 481, 6642),
        (9, 865, 7507), (10, 955, 8462), (11, 446, 8908), (12, 384, 9292),
        (13, 370, 9662), (14, 372, 10034), (15, 365, 10399), (16, 531, 10930),
        (17, 721, 11651), (18, 621, 12272), (19, 730, 13002), (20, 473, 13475),
        (21, 528, 14003), (22, 600, 14603), (23, 1164, 15767), (24, 1331, 17098),
        (25, 1169, 18267), (26, 1503, 19770), (27, 3000, 22770), (28, 4167, 26937)
    ]
    cm1_df = pd.DataFrame(cm1_data, columns=['day', 'daily_orders', 'cumulative_orders'])

if cm2_df.empty:
    logger.warning("CM2动态数据获取失败，使用静态备用数据")
    cm2_data = [
        (1, 5351, 5351), (2, 3126, 8477), (3, 2207, 10684), (4, 1079, 11763),
        (5, 845, 12608), (6, 880, 13488), (7, 873, 14361), (8, 695, 15056),
        (9, 1300, 16356), (10, 1278, 17634), (11, 684, 18318), (12, 700, 19018),
        (13, 701, 19719), (14, 601, 20320), (15, 1029, 21349), (16, 1656, 23005),
        (17, 1773, 24778), (18, 864, 25642), (19, 886, 26528), (20, 730, 27258),
        (21, 911, 28169), (22, 1157, 29326), (23, 2124, 31459), (24, 2448, 33907)
    ]
    cm2_df = pd.DataFrame(cm2_data, columns=['day', 'daily_orders', 'cumulative_orders'])

logger.info(f"CM1数据: {len(cm1_df)}天, 最新累计订单: {cm1_df['cumulative_orders'].iloc[-1] if not cm1_df.empty else 0}")
logger.info(f"CM2数据: {len(cm2_df)}天, 最新累计订单: {cm2_df['cumulative_orders'].iloc[-1] if not cm2_df.empty else 0}")

class OrderPredictor:
    def __init__(self):
        self.cm1_model = None
        self.cm2_model = None
        self.cm1_growth_phases = None
        self.cm2_growth_phases = None
        self.cm1_df = None
        self.cm2_df = None
        self.refresh_data()
        self.train_models()
    
    def refresh_data(self):
        """刷新CM1和CM2的最新数据"""
        global cm1_df, cm2_df
        
        # 重新获取动态数据
        new_cm1_df = get_dynamic_vehicle_data('CM1')
        new_cm2_df = get_dynamic_vehicle_data('CM2')
        
        # 如果获取成功，更新数据
        if not new_cm1_df.empty:
            cm1_df = new_cm1_df
            logger.info(f"CM1数据已更新: {len(cm1_df)}天, 最新累计订单: {cm1_df['cumulative_orders'].iloc[-1]}")
        
        if not new_cm2_df.empty:
            cm2_df = new_cm2_df
            logger.info(f"CM2数据已更新: {len(cm2_df)}天, 最新累计订单: {cm2_df['cumulative_orders'].iloc[-1]}")
        
        # 更新实例变量
        self.cm1_df = cm1_df.copy()
        self.cm2_df = cm2_df.copy()
    
    def analyze_growth_phases(self, df, model_name):
        """分析增长阶段特征"""
        daily_orders = df['daily_orders'].values
        cumulative = df['cumulative_orders'].values
        days = df['day'].values
        
        # 计算增长率
        growth_rates = []
        for i in range(1, len(cumulative)):
            rate = (cumulative[i] - cumulative[i-1]) / cumulative[i-1] * 100
            growth_rates.append(rate)
        
        # 识别阶段特征
        phases = {
            'initial_surge': [],  # 开局冲高阶段
            'plateau': [],        # 平台期
            'final_surge': []     # 后段拉升
        }
        
        # 基于CM1实际数据分析的精确分位值识别阶段
        max_day = max(days) if len(days) > 0 else 28  # 获取实际最大天数，默认28天
        
        if model_name == 'CM1':
            # CM1实际特征: 0-0.107初期爆发，0.179-0.750平台期，0.500后重新加速，0.964末期爆发
            for i, day in enumerate(days):
                time_ratio = day / max_day
                if time_ratio <= 0.107:  # 初期爆发阶段：前3天高峰期
                    phases['initial_surge'].append(i)
                elif 0.179 <= time_ratio <= 0.750:  # 平台期：第5-21天相对稳定期
                    phases['plateau'].append(i)
                else:  # 后期加速：0.500后开始重新加速，0.964处大幅跃升
                    phases['final_surge'].append(i)
        else:  # CM2
            # CM2: 参考CM1精确特征，适当调整阶段划分
            for i, day in enumerate(days):
                time_ratio = day / max_day
                if time_ratio <= 0.107:  # 初期爆发：保持与CM1一致
                    phases['initial_surge'].append(i)
                elif 0.179 <= time_ratio <= 0.750:  # 平台期：参考CM1的0.750平台期结束点
                    phases['plateau'].append(i)
                else:  # 后期拉升：0.750后进入拉升期
                    phases['final_surge'].append(i)
        
        return phases
    
    def sigmoid_model(self, x, a, b, c, d):
        """S型曲线模型"""
        return a / (1 + np.exp(-b * (x - c))) + d
    
    def fit_cm1_sigmoid(self, X_cm1, y_cm1):
        """基于CM1实际数据特征的S型曲线拟合"""
        def sigmoid_growth(x, L, k, x0, b):
            return L / (1 + np.exp(-k * (x - x0))) + b
        
        try:
            # 基于CM1实际数据特征的参数估计
            max_day = max(X_cm1) if len(X_cm1) > 0 else 28
            
            # 根据CM1实际增长特征调整参数
            L_init = max(y_cm1) * 1.15  # 最大值的1.15倍，考虑CM1在末期的大幅跃升
            k_init = 0.25  # 适中的增长率，反映CM1的渐进式增长特征
            
            # 拐点设置在0.750分位值附近，对应CM1平台期结束、后期加速开始
            x0_init = max_day * 0.750  # 约第21天，CM1开始明显加速的位置
            b_init = min(y_cm1) * 0.8  # 基础值稍低，突出初期爆发特征
            
            # 设置参数边界，确保拟合结果符合CM1实际特征
            bounds = (
                [max(y_cm1) * 0.9, 0.1, max_day * 0.5, 0],  # 下界
                [max(y_cm1) * 1.5, 0.5, max_day * 0.9, min(y_cm1)]  # 上界
            )
            
            popt, _ = curve_fit(sigmoid_growth, X_cm1, y_cm1, 
                              p0=[L_init, k_init, x0_init, b_init],
                              bounds=bounds,
                              maxfev=8000)
            
            self.cm1_sigmoid_params = popt
            
            # 生成拟合曲线数据，延伸到35天以展示完整的S型特征
            x_fit = np.linspace(1, 35, 100)
            y_fit = sigmoid_growth(x_fit, *popt)
            
            return x_fit, y_fit
            
        except Exception as e:
            print(f"S型曲线拟合失败: {e}")
            # 返回基于CM1特征的分段线性拟合作为备选
            x_fit = np.linspace(1, 35, 100)
            
            # 分段插值，保持CM1的三阶段特征
            max_day = max(X_cm1) if len(X_cm1) > 0 else 28
            initial_end = int(max_day * 0.107)
            plateau_end = int(max_day * 0.750)
            
            y_fit = np.zeros_like(x_fit)
            for i, x in enumerate(x_fit):
                if x <= initial_end:
                    # 初期爆发阶段
                    y_fit[i] = np.interp(x, X_cm1[X_cm1 <= initial_end], y_cm1[X_cm1 <= initial_end])
                elif x <= plateau_end:
                    # 平台期
                    plateau_mask = (X_cm1 >= initial_end) & (X_cm1 <= plateau_end)
                    if np.sum(plateau_mask) > 0:
                        y_fit[i] = np.interp(x, X_cm1[plateau_mask], y_cm1[plateau_mask])
                    else:
                        y_fit[i] = np.interp(x, X_cm1, y_cm1)
                else:
                    # 后期加速阶段
                    final_mask = X_cm1 >= plateau_end
                    if np.sum(final_mask) > 0:
                        y_fit[i] = np.interp(x, X_cm1[final_mask], y_cm1[final_mask])
                    else:
                        y_fit[i] = np.interp(x, X_cm1, y_cm1)
            
            return x_fit, y_fit
    
    def train_models(self):
        """训练改进的预测模型"""
        
        # 确保数据已加载
        if self.cm1_df.empty or self.cm2_df.empty:
            logger.warning("数据为空，尝试重新加载")
            self.refresh_data()
        
        # 分析增长阶段
        self.cm1_growth_phases = self.analyze_growth_phases(self.cm1_df, 'CM1')
        self.cm2_growth_phases = self.analyze_growth_phases(self.cm2_df, 'CM2')
        
        # 为CM1训练S型曲线模型
        X_cm1 = self.cm1_df['day'].values
        y_cm1 = self.cm1_df['cumulative_orders'].values
        
        try:
            # 使用改进的S型曲线拟合方法
            x_fit, y_fit = self.fit_cm1_sigmoid(X_cm1, y_cm1)
        except:
            # 如果S型拟合失败，使用多项式回归作为备选
            self.cm1_model = Pipeline([
                ('poly', PolynomialFeatures(degree=4)),
                ('linear', LinearRegression())
            ])
            self.cm1_model.fit(X_cm1.reshape(-1, 1), y_cm1)
            self.cm1_sigmoid_params = None
        
        # 为CM2训练分段模型
        X_cm2 = self.cm2_df['day'].values
        y_cm2 = self.cm2_df['cumulative_orders'].values
        
        # 基于CM1实际数据特征的精确分段建模
        self.cm2_phase_models = {}
        max_day_cm2 = max(X_cm2) if len(X_cm2) > 0 else 28
        
        # 初期爆发阶段 (0-0.107分位值，约前3天)
        initial_threshold = int(max_day_cm2 * 0.107)
        initial_mask = X_cm2 <= initial_threshold
        initial_days = X_cm2[initial_mask]
        initial_orders = y_cm2[initial_mask]
        if len(initial_days) > 1:
            self.cm2_phase_models['initial'] = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ])
            self.cm2_phase_models['initial'].fit(initial_days.reshape(-1, 1), initial_orders)
        
        # 平台期 (0.179-0.714分位值)
        plateau_start = int(max_day_cm2 * 0.179)
        plateau_end = int(max_day_cm2 * 0.714)
        plateau_mask = (X_cm2 >= plateau_start) & (X_cm2 <= plateau_end)
        plateau_days = X_cm2[plateau_mask]
        plateau_orders = y_cm2[plateau_mask]
        if len(plateau_days) > 2:
            self.cm2_phase_models['plateau'] = Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ])
            self.cm2_phase_models['plateau'].fit(plateau_days.reshape(-1, 1), plateau_orders)
        
        # 后期拉升阶段 (0.714分位值以后) - 参考CM1在0.500分位值后的重新加速特征
        final_threshold = int(max_day_cm2 * 0.714)
        final_mask = X_cm2 >= final_threshold
        final_days = X_cm2[final_mask]
        final_orders = y_cm2[final_mask]
        if len(final_days) > 1:
            # 使用更高次多项式捕捉加速增长特征
            self.cm2_phase_models['final'] = Pipeline([
                ('poly', PolynomialFeatures(degree=3)),
                ('linear', LinearRegression())
            ])
            self.cm2_phase_models['final'].fit(final_days.reshape(-1, 1), final_orders)
        
        # 存储CM1的后期加速特征用于CM2预测参考 (基于0.500分位值后的数据)
        max_day_cm1 = max(X_cm1) if len(X_cm1) > 0 else 28
        cm1_acceleration_threshold = int(max_day_cm1 * 0.500)  # CM1在0.500分位值后开始重新加速
        cm1_final_mask = X_cm1 >= cm1_acceleration_threshold
        self.cm1_final_growth_pattern = None
        if np.sum(cm1_final_mask) > 2:
            cm1_final_days = X_cm1[cm1_final_mask]
            cm1_final_orders = y_cm1[cm1_final_mask]
            # 计算CM1后期的增长加速度
            cm1_daily_growth = np.diff(cm1_final_orders)
            self.cm1_final_growth_pattern = {
                'avg_daily_growth': np.mean(cm1_daily_growth),
                'growth_acceleration': np.mean(np.diff(cm1_daily_growth)),
                'max_daily_growth': np.max(cm1_daily_growth),
                'acceleration_start_ratio': 0.500,  # CM1重新加速的分位值
                'surge_start_ratio': 0.964  # CM1大幅跃升的分位值
            }
        
        # 整体S型曲线拟合作为补充
        try:
            popt_cm2, _ = curve_fit(self.sigmoid_model, X_cm2, y_cm2, 
                                   p0=[50000, 0.15, 12, 0], maxfev=5000)
            self.cm2_sigmoid_params = popt_cm2
        except:
            self.cm2_sigmoid_params = None
    
    def predict_cm2(self, target_days, current_day=None):
        """预测CM2到指定天数的累计订单数 - 根据目标天数动态调整预测策略"""
        if target_days <= len(self.cm2_df):
            # 如果目标天数在已有数据范围内，直接返回实际数据
            return self.cm2_df[self.cm2_df['day'] <= target_days]['cumulative_orders'].iloc[-1]
        
        # 如果没有指定当前天数，默认为目标天数
        if current_day is None:
            current_day = target_days
            
        # CM2节奏更快：24天达到33907单，相当于CM1的28天26937单
        # CM2的时间压缩比例：24/28 = 0.857，即CM2用85.7%的时间完成更大体量
        cm2_acceleration_ratio = 0.857  # CM2节奏加速比例
        
        # 关键改进：根据目标天数动态调整分位值阈值
        # 目标天数越长，各阶段的转换点应该相应延后
        target_ratio = target_days / 28.0  # 相对于CM1标准周期的比例
        
        # 基于CM1实际数据优化的分位值阈值（更准确反映非线性增长特征）
        initial_threshold = int(target_days * 0.536)    # 初期平稳期结束（第15天/28天=0.536）
        plateau_start = int(target_days * 0.143)        # 平台期开始（保持原值）
        plateau_end = int(target_days * 0.750)          # 中期波动期结束（第21天/28天=0.750）
        acceleration_start = int(target_days * 0.786)   # 加速准备期开始（第22天/28天=0.786）
        surge_start = int(target_days * 0.929)          # 爆发增长期开始（第26天/28天=0.929）
        
        if current_day <= initial_threshold and 'initial' in self.cm2_phase_models:
            # 初期阶段 - 使用动态阈值
            predicted_cumulative = self.cm2_phase_models['initial'].predict([[current_day]])[0]
        elif current_day <= plateau_end and 'plateau' in self.cm2_phase_models:
            # 平台期阶段 - 使用动态阈值
            predicted_cumulative = self.cm2_phase_models['plateau'].predict([[current_day]])[0]
        elif current_day >= plateau_end:
            # 后期拉升阶段 - 基于目标天数动态调整增长模式
            if 'final' in self.cm2_phase_models and current_day <= acceleration_start:
                # 使用分段模型进行近期预测
                predicted_cumulative = self.cm2_phase_models['final'].predict([[current_day]])[0]
            else:
                # 长期预测：根据目标天数调整增长策略
                last_actual = cm2_df['cumulative_orders'].iloc[-1]  # 33907
                last_day = cm2_df['day'].iloc[-1]  # 24
                
                if current_day > acceleration_start:
                    # 根据目标天数调整加速策略
                    days_beyond = current_day - last_day
                    
                    # CM2的基础日增长（基于实际数据：第19-24天平均增长）
                    cm2_recent_daily_growth = np.mean([886, 730, 911, 1157, 2124, 2448])  # 约1376单/天
                    
                    # 关键改进：基于目标天数的动态加速因子
                    # 目标天数越长，中间阶段的增长应该更平缓，为后期留出更大空间
                    target_position_ratio = current_day / target_days  # 当前天数在目标中的位置
                    
                    acceleration_factor = 1.0
                    
                    if target_position_ratio >= 0.929:  # 接近目标的最后阶段（爆发期）
                        # 大幅跃升期 - 根据目标天数调整爆发强度
                        excess_ratio = target_position_ratio - 0.929
                        # 目标天数越长，后期爆发越强
                        target_boost = 1.0 + (target_days - 28) * 0.1  # 每多一天增加10%爆发力
                        acceleration_factor = 2.0 + excess_ratio * 12.0 * target_boost
                    elif target_position_ratio >= 0.786:  # 加速准备期
                        # 根据目标天数调整渐进加速强度
                        progress_ratio = (target_position_ratio - 0.786) / (0.929 - 0.786)
                        target_moderation = max(0.5, 1.0 - (target_days - 28) * 0.02)  # 目标越长，中期越平缓
                        acceleration_factor = 1.0 + progress_ratio * 1.0 * target_moderation
                    
                    # 考虑CM2规模特征和目标天数的影响
                    scale_factor = 1.2 * (1.0 + (target_days - 28) * 0.05)  # 目标越长，规模越大
                    predicted_cumulative = last_actual + (cm2_recent_daily_growth * days_beyond * acceleration_factor * scale_factor)
                elif self.cm2_sigmoid_params is not None:
                    # 使用S型曲线进行长期预测，根据目标天数调整参数
                    # 调整S型曲线的参数以适应不同的目标天数
                    adjusted_params = list(self.cm2_sigmoid_params)
                    if len(adjusted_params) >= 4:
                        # 调整最大值参数以适应目标天数
                        adjusted_params[0] *= (1.0 + (target_days - 28) * 0.1)
                    predicted_cumulative = self.sigmoid_model(current_day, *adjusted_params)
                else:
                    # 备选：基于目标天数调整增长趋势
                    recent_growth = []
                    for i in range(max(0, len(cm2_df)-3), len(cm2_df)-1):
                        growth = (cm2_df['cumulative_orders'].iloc[i+1] - cm2_df['cumulative_orders'].iloc[i])
                        recent_growth.append(growth)
                    
                    avg_daily_growth = np.mean(recent_growth) if recent_growth else 800
                    # 根据目标天数调整增长率
                    target_adjustment = 1.0 + (target_days - 28) * 0.03
                    growth_multiplier = 1.0 + (current_day - last_day) * 0.05 * target_adjustment
                    predicted_cumulative = last_actual + (current_day - last_day) * avg_daily_growth * growth_multiplier
        else:
            # 使用S型曲线进行预测，根据目标天数调整
            if self.cm2_sigmoid_params is not None:
                adjusted_params = list(self.cm2_sigmoid_params)
                if len(adjusted_params) >= 4:
                    # 根据目标天数调整S型曲线参数
                    adjusted_params[0] *= (1.0 + (target_days - 28) * 0.1)
                predicted_cumulative = self.sigmoid_model(current_day, *adjusted_params)
            else:
                # 最后的备选方案 - 根据目标天数调整基础增长
                last_actual = cm2_df['cumulative_orders'].iloc[-1]
                last_day = cm2_df['day'].iloc[-1]
                base_growth = 500 * (1.0 + (target_days - 28) * 0.05)
                predicted_cumulative = last_actual + (current_day - last_day) * base_growth
        
        # 确保预测值不小于最后一天的实际值
        last_actual = cm2_df['cumulative_orders'].iloc[-1]
        return max(predicted_cumulative, last_actual)
    
    def generate_prediction_curve(self, target_days):
        """生成完整的预测曲线数据 - 与predict_cm2使用相同的优化逻辑"""
        # 实际数据部分
        actual_days = cm2_df['day'].values
        actual_cumulative = cm2_df['cumulative_orders'].values
        
        # 预测数据部分
        if target_days > len(self.cm2_df):
            prediction_days = np.arange(len(self.cm2_df) + 1, target_days + 1)
            prediction_cumulative = []
            
            for day in prediction_days:
                # 关键改进：传递目标天数，让每个中间天数都根据最终目标调整预测值
                pred = self.predict_cm2(target_days, current_day=day)
                prediction_cumulative.append(pred)
            
            prediction_cumulative = np.array(prediction_cumulative)
            
            # 确保预测曲线的连续性
            last_actual = actual_cumulative[-1]
            prediction_cumulative = np.maximum(prediction_cumulative, last_actual)
        else:
            prediction_days = np.array([])
            prediction_cumulative = np.array([])
        
        return actual_days, actual_cumulative, prediction_days, prediction_cumulative

# 初始化预测器
predictor = OrderPredictor()

def create_prediction_plot(target_days):
    """创建预测图表 - 2x1布局：柱状图+折线图"""
    # 创建子图布局 2x1
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('每日订单数对比', '累计订单数预测'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
        vertical_spacing=0.15
    )
    
    # === 左侧：柱状图显示每日订单数 ===
    
    # 计算CM1每日订单数
    cm1_daily = [cm1_df['cumulative_orders'].iloc[0]]  # 第1天
    for i in range(1, len(cm1_df)):
        daily_orders = cm1_df['cumulative_orders'].iloc[i] - cm1_df['cumulative_orders'].iloc[i-1]
        cm1_daily.append(daily_orders)
    
    # 计算CM2每日订单数（实际）
    cm2_daily_actual = [cm2_df['cumulative_orders'].iloc[0]]  # 第1天
    for i in range(1, len(cm2_df)):
        daily_orders = cm2_df['cumulative_orders'].iloc[i] - cm2_df['cumulative_orders'].iloc[i-1]
        cm2_daily_actual.append(daily_orders)
    
    # 计算CM2每日订单数（预测部分）- 关键修复：传递target_days参数
    cm2_daily_predicted = []
    if target_days > len(predictor.cm2_df):
        last_cumulative = predictor.cm2_df['cumulative_orders'].iloc[-1]
        for day in range(len(predictor.cm2_df) + 1, target_days + 1):
            current_cumulative = predictor.predict_cm2(target_days, current_day=day)
            if day == len(predictor.cm2_df) + 1:
                daily_orders = current_cumulative - last_cumulative
            else:
                prev_cumulative = predictor.predict_cm2(target_days, current_day=day - 1)
                daily_orders = current_cumulative - prev_cumulative
            cm2_daily_predicted.append(daily_orders)
    
    # 添加CM1每日订单柱状图
    fig.add_trace(go.Bar(
        x=cm1_df['day'],
        y=cm1_daily,
        name='CM1 每日订单',
        marker_color='lightblue',
        opacity=0.8
    ), row=1, col=1)
    
    # 添加CM2实际每日订单柱状图
    fig.add_trace(go.Bar(
        x=cm2_df['day'],
        y=cm2_daily_actual,
        name='CM2 实际每日订单',
        marker_color='lightcoral',
        opacity=0.8
    ), row=1, col=1)
    
    # 添加CM2预测每日订单柱状图（透明度0.5）
    if cm2_daily_predicted:
        prediction_days = list(range(len(predictor.cm2_df) + 1, target_days + 1))
        fig.add_trace(go.Bar(
            x=prediction_days,
            y=cm2_daily_predicted,
            name='CM2 预测每日订单',
            marker_color='orange',
            opacity=0.5  # 透明度降低0.5
        ), row=1, col=1)
    
    # === 右侧：折线图显示累计订单数 ===
    
    # 添加CM1累计订单折线
    fig.add_trace(go.Scatter(
        x=cm1_df['day'],
        y=cm1_df['cumulative_orders'],
        mode='lines+markers',
        name='CM1 累计订单',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ), row=2, col=1)
    
    # 获取CM2的实际和预测数据
    actual_days, actual_cumulative, prediction_days, prediction_cumulative = predictor.generate_prediction_curve(target_days)
    
    # 添加CM2实际累计订单折线
    fig.add_trace(go.Scatter(
        x=actual_days,
        y=actual_cumulative,
        mode='lines+markers',
        name='CM2 实际累计订单',
        line=dict(color='red', width=2),
        marker=dict(size=4)
    ), row=2, col=1)
    
    # 添加CM2预测累计订单折线
    if len(prediction_days) > 0:
        # 连接点：从最后一个实际数据点到第一个预测点
        connection_x = [actual_days[-1], prediction_days[0]]
        connection_y = [actual_cumulative[-1], prediction_cumulative[0]]
        
        fig.add_trace(go.Scatter(
            x=connection_x,
            y=connection_y,
            mode='lines',
            name='',
            line=dict(color='orange', width=2, dash='dash'),
            showlegend=False
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=prediction_days,
            y=prediction_cumulative,
            mode='lines+markers',
            name='CM2 预测累计订单',
            line=dict(color='orange', width=2, dash='dash'),
            marker=dict(size=4)
        ), row=2, col=1)
    
    # 设置图表布局
    fig.update_layout(
        title=f'CM1 vs CM2 订单分析与预测 (CM2 目标天数: {target_days})',
        showlegend=True,
        height=800,
        hovermode='x unified',
        template='plotly_white'
    )
    
    # 设置子图坐标轴标题
    fig.update_xaxes(title_text="天数", row=1, col=1)
    fig.update_yaxes(title_text="每日订单数", row=1, col=1)
    fig.update_xaxes(title_text="天数", row=2, col=1)
    fig.update_yaxes(title_text="累计订单数", row=2, col=1)
    
    # 添加网格
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def predict_orders(target_days):
    """预测函数，返回图表和预测结果"""
    try:
        target_days = int(target_days)
        if target_days < 1:
            return None, "请输入大于0的天数"
        
        # 创建图表
        fig = create_prediction_plot(target_days)
        
        # 计算预测结果
        predicted_cumulative = predictor.predict_cm2(target_days)
        
        # 生成结果文本
        if target_days <= len(predictor.cm2_df):
            result_text = f"""📊 **预测结果**
            
**目标天数**: {target_days} 天
**累计订单数**: {predicted_cumulative:,.0f} 单 (实际数据)

💡 **说明**: 该天数在现有数据范围内，显示的是实际累计订单数。
            """
        else:
            # 计算相比最后一天的增长
            last_actual = predictor.cm2_df['cumulative_orders'].iloc[-1]
            growth = predicted_cumulative - last_actual
            growth_rate = (growth / last_actual) * 100
            
            result_text = f"""📊 **预测结果**
            
**目标天数**: {target_days} 天
**预测累计订单数**: {predicted_cumulative:,.0f} 单
**相比第{len(predictor.cm2_df)}天增长**: {growth:,.0f} 单 (+{growth_rate:.1f}%)

💡 **说明**: 预测基于CM1全量数据和CM2已有{len(predictor.cm2_df)}天数据的趋势分析。
            """
        
        return fig, result_text
        
    except ValueError:
        return None, "请输入有效的数字"
    except Exception as e:
        return None, f"预测过程中出现错误: {str(e)}"

class OrderTrendMonitor:
    def __init__(self):
        self.df = None
        self.business_def = None
        self.vehicle_prices = {}
        self.load_data()
        self.load_business_definition()
    
    def load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_parquet(DATA_PATH)
            # 确保日期列为datetime类型
            if 'Intention_Payment_Time' in self.df.columns:
                self.df['Intention_Payment_Time'] = pd.to_datetime(self.df['Intention_Payment_Time'])
            if 'intention_refund_time' in self.df.columns:
                self.df['intention_refund_time'] = pd.to_datetime(self.df['intention_refund_time'])
            if 'Lock_Time' in self.df.columns:
                self.df['Lock_Time'] = pd.to_datetime(self.df['Lock_Time'])
            logger.info(f"数据加载成功，共{len(self.df)}条记录")
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise
    
    def load_business_definition(self):
        """加载业务定义"""
        try:
            with open(BUSINESS_DEF_PATH, 'r', encoding='utf-8') as f:
                business_data = json.load(f)
            
            # 适配新的JSON结构
            if 'time_periods' in business_data:
                self.business_def = business_data['time_periods']
                self.vehicle_prices = business_data.get('vehicle_prices', {})
            else:
                # 兼容旧格式
                self.business_def = business_data
                self.vehicle_prices = {}
            
            logger.info("业务定义加载成功")
        except Exception as e:
            logger.error(f"业务定义加载失败: {str(e)}")
            raise
    
    def get_vehicle_price(self, product_name: str) -> str:
        """根据Product Name获取对应的价格"""
        try:
            if product_name in self.vehicle_prices:
                price = self.vehicle_prices[product_name]
                if price and price != 0:
                    return f"{price:,.0f}"
                else:
                    return "暂无价格"
            else:
                return "暂无价格"
        except Exception as e:
            logger.error(f"获取价格时出错: {e}")
            return "暂无价格"
    
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
    
    def calculate_days_from_end(self, vehicle_group: str, date: datetime) -> int:
        """计算从预售结束的天数（用于锁单第N日计算）"""
        if vehicle_group not in self.business_def:
            return -1
        
        end_date = datetime.strptime(self.business_def[vehicle_group]['end'], '%Y-%m-%d')
        return (date - end_date).days  # 预售结束当天为第0日开始计算
    
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
            daily_orders['车型分组'] = vehicle
            
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
        
        for i, vehicle in enumerate(data['车型分组'].unique()):
            vehicle_data = data[data['车型分组'] == vehicle]
            
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
        
        for i, vehicle in enumerate(data['车型分组'].unique()):
            vehicle_data = data[data['车型分组'] == vehicle]
            
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
        
        for i, vehicle in enumerate(data['车型分组'].unique()):
            vehicle_data = data[data['车型分组'] == vehicle]
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
    
    def prepare_conversion_rate_data(self, selected_vehicles: List[str], days_after_launch: int = 1) -> pd.DataFrame:
        """准备小订转化率数据用于对比折线图"""
        try:
            conversion_data = []
            
            for vehicle in selected_vehicles:
                if vehicle not in self.business_def:
                    continue
                    
                vehicle_data = self.df[self.df['车型分组'] == vehicle].copy()
                if vehicle_data.empty:
                    continue
                
                # 获取预售周期
                start_date = datetime.strptime(self.business_def[vehicle]['start'], '%Y-%m-%d')
                end_date = datetime.strptime(self.business_def[vehicle]['end'], '%Y-%m-%d')
                max_days = (end_date - start_date).days + 1
                
                # 按日期分组统计订单数
                daily_orders = vehicle_data.groupby(vehicle_data['Intention_Payment_Time'].dt.date).size().reset_index()
                daily_orders.columns = ['date', 'daily_orders']
                daily_orders['date'] = pd.to_datetime(daily_orders['date'])
                
                # 计算从预售开始的天数
                daily_orders['days_from_start'] = daily_orders['date'].apply(
                    lambda x: self.calculate_days_from_start(vehicle, x.to_pydatetime())
                )
                
                # 过滤有效天数
                daily_orders = daily_orders[
                    (daily_orders['days_from_start'] >= 1) & 
                    (daily_orders['days_from_start'] <= max_days)
                ]
                
                # 计算每日的小订转化率
                lock_cutoff_date = end_date + timedelta(days=days_after_launch)
                
                for _, row in daily_orders.iterrows():
                    day_num = row['days_from_start']
                    target_date = start_date + timedelta(days=int(day_num) - 1)
                    
                    # 获取当日的小订订单
                    daily_orders_data = self.df[
                        (self.df['车型分组'] == vehicle) & 
                        (self.df['Intention_Payment_Time'].dt.date == target_date.date())
                    ]
                    
                    # 计算小订留存锁单数
                    lock_orders = daily_orders_data[
                        (daily_orders_data['Lock_Time'].notna()) & 
                        (daily_orders_data['Intention_Payment_Time'].notna()) & 
                        (daily_orders_data['Lock_Time'] < pd.Timestamp(lock_cutoff_date))
                    ]
                    
                    lock_count = len(lock_orders)
                    total_count = len(daily_orders_data)
                    conversion_rate = (lock_count / total_count * 100) if total_count > 0 else 0
                    
                    conversion_data.append({
                        '车型分组': vehicle,
                        'days_from_start': day_num,
                        'conversion_rate': conversion_rate,
                        'lock_count': lock_count,
                        'total_count': total_count
                    })
            
            return pd.DataFrame(conversion_data)
            
        except Exception as e:
            logger.error(f"准备转化率数据时出错: {str(e)}")
            return pd.DataFrame()

    def create_conversion_rate_chart(self, selected_vehicles: List[str], days_after_launch: int = 1) -> go.Figure:
        """创建车型小订转化率对比折线图"""
        try:
            # 获取转化率数据
            conversion_data = self.prepare_conversion_rate_data(selected_vehicles, days_after_launch)
            
            if conversion_data.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="暂无转化率数据",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                fig.update_layout(
                    title="车型小订转化率对比",
                    xaxis_title="预售天数",
                    yaxis_title="转化率 (%)",
                    height=400
                )
                return fig
            
            fig = go.Figure()
            
            # 为每个车型添加折线
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            
            for i, vehicle in enumerate(selected_vehicles):
                vehicle_data = conversion_data[conversion_data['车型分组'] == vehicle]
                
                if not vehicle_data.empty:
                    # 按天数排序
                    vehicle_data = vehicle_data.sort_values('days_from_start')
                    
                    # 创建悬停文本
                    hover_text = [
                        f"车型: {vehicle}<br>" +
                        f"预售第{int(row['days_from_start'])}天<br>" +
                        f"转化率: {row['conversion_rate']:.2f}%<br>" +
                        f"锁单数: {int(row['lock_count'])}<br>" +
                        f"小订数: {int(row['total_count'])}"
                        for _, row in vehicle_data.iterrows()
                    ]
                    
                    fig.add_trace(go.Scatter(
                        x=vehicle_data['days_from_start'],
                        y=vehicle_data['conversion_rate'],
                        mode='lines+markers',
                        name=vehicle,
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=6),
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=hover_text
                    ))
            
            # 更新布局
            fig.update_layout(
                title={
                    'text': f"车型小订转化率对比 (预售结束后{days_after_launch}天内锁单)",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title="预售天数",
                yaxis_title="转化率 (%)",
                height=400,
                hovermode='closest',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(t=80, b=50, l=50, r=50),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            # 设置网格
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                range=[0, 50]
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"创建转化率图表时出错: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"图表生成失败: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=14, color='red')
            )
            fig.update_layout(
                title="车型小订转化率对比",
                height=400
            )
            return fig

    def create_daily_change_table(self, data: pd.DataFrame, days_after_launch: int = 1) -> pd.DataFrame:
        """创建订单的日变化表格 - 车型对比格式，严格按顺序排列并高亮较大值"""
        if data.empty:
            return pd.DataFrame({'提示': ['暂无数据']})
        
        # 获取所有车型和天数
        vehicles = sorted(data['车型分组'].unique())
        if len(vehicles) == 0:
            return pd.DataFrame({'提示': ['暂无数据']})
        
        # 获取所有天数的并集
        all_days = set()
        for vehicle in vehicles:
            vehicle_data = data[data['车型分组'] == vehicle]
            all_days.update(vehicle_data['days_from_start'].tolist())
        all_days = sorted(list(all_days))
        
        # 准备表格数据
        table_data = []
        
        for day in all_days:
            # 收集当前行的所有数据
            day_data = {}
            for vehicle in vehicles:
                vehicle_data = data[
                    (data['车型分组'] == vehicle) & 
                    (data['days_from_start'] == day)
                ]
                
                if not vehicle_data.empty:
                    row = vehicle_data.iloc[0]
                    
                    # 计算小订留存锁单数和转化率
                    if vehicle in self.business_def:
                        start_date = datetime.strptime(self.business_def[vehicle]['start'], '%Y-%m-%d')
                        end_date = datetime.strptime(self.business_def[vehicle]['end'], '%Y-%m-%d')
                        target_date = start_date + timedelta(days=int(day) - 1)
                        
                        # 获取当日的小订订单
                        daily_orders_data = self.df[
                            (self.df['车型分组'] == vehicle) & 
                            (self.df['Intention_Payment_Time'].dt.date == target_date.date())
                        ]
                        
                        # 计算小订留存锁单数：同时含有Lock_Time、Intention_Payment_Time，且Lock_Time < 发布会结束日期+N日
                        lock_cutoff_date = end_date + timedelta(days=days_after_launch)
                        lock_orders = daily_orders_data[
                            (daily_orders_data['Lock_Time'].notna()) & 
                            (daily_orders_data['Intention_Payment_Time'].notna()) & 
                            (daily_orders_data['Lock_Time'] < pd.Timestamp(lock_cutoff_date))
                        ]
                        
                        lock_count = len(lock_orders)
                        total_count = len(daily_orders_data)
                        conversion_rate = (lock_count / total_count * 100) if total_count > 0 else 0
                    else:
                        lock_count = 0
                        conversion_rate = 0
                    
                    day_data[vehicle] = {
                        'daily_orders': int(row['daily_orders']),
                        'cumulative_orders': int(row['cumulative_orders']),
                        'lock_count': lock_count,
                        'conversion_rate': conversion_rate
                    }
                else:
                    day_data[vehicle] = {
                        'daily_orders': None,
                        'cumulative_orders': None,
                        'lock_count': None,
                        'conversion_rate': None
                    }
            
            # 按照严格顺序构建行数据
            row_data = {'第N日': int(day)}
            
            # 1. 小订数对比
            daily_values = [day_data[v]['daily_orders'] for v in vehicles if day_data[v]['daily_orders'] is not None]
            max_daily = max(daily_values) if daily_values else None
            for vehicle in vehicles:
                value = day_data[vehicle]['daily_orders']
                if value is not None:
                    if max_daily and value == max_daily and len([v for v in daily_values if v == max_daily]) == 1:
                        row_data[f'{vehicle}小订数'] = f"<span style='color: red;'>{value}</span>"
                    else:
                        row_data[f'{vehicle}小订数'] = str(value)
                else:
                    row_data[f'{vehicle}小订数'] = '-'
            
            # 2. 累计小订数对比
            cumulative_values = [day_data[v]['cumulative_orders'] for v in vehicles if day_data[v]['cumulative_orders'] is not None]
            max_cumulative = max(cumulative_values) if cumulative_values else None
            for vehicle in vehicles:
                value = day_data[vehicle]['cumulative_orders']
                if value is not None:
                    if max_cumulative and value == max_cumulative and len([v for v in cumulative_values if v == max_cumulative]) == 1:
                        row_data[f'{vehicle}累计小订数'] = f"<span style='color: red;'>{value}</span>"
                    else:
                        row_data[f'{vehicle}累计小订数'] = str(value)
                else:
                    row_data[f'{vehicle}累计小订数'] = '-'
            
            # 3. 发布会后N日锁单数对比
            lock_values = [day_data[v]['lock_count'] for v in vehicles if day_data[v]['lock_count'] is not None]
            max_lock = max(lock_values) if lock_values else None
            for vehicle in vehicles:
                value = day_data[vehicle]['lock_count']
                if value is not None:
                    if max_lock and value == max_lock and len([v for v in lock_values if v == max_lock]) == 1:
                        row_data[f'{vehicle}发布会后{days_after_launch}日锁单数'] = f"<span style='color: red;'>{value}</span>"
                    else:
                        row_data[f'{vehicle}发布会后{days_after_launch}日锁单数'] = str(value)
                else:
                    row_data[f'{vehicle}发布会后{days_after_launch}日锁单数'] = '-'
            
            # 4. 小订转化率对比
            conversion_values = [day_data[v]['conversion_rate'] for v in vehicles if day_data[v]['conversion_rate'] is not None]
            max_conversion = max(conversion_values) if conversion_values else None
            for vehicle in vehicles:
                value = day_data[vehicle]['conversion_rate']
                if value is not None:
                    if max_conversion and value == max_conversion and len([v for v in conversion_values if v == max_conversion]) == 1:
                        row_data[f'{vehicle}小订转化率(%)'] = f"<span style='color: red;'>{value:.1f}%</span>"
                    else:
                        row_data[f'{vehicle}小订转化率(%)'] = f"{value:.1f}%"
                else:
                    row_data[f'{vehicle}小订转化率(%)'] = '-'
            
            table_data.append(row_data)
        
        return pd.DataFrame(table_data)
    
    def prepare_summary_statistics_data(self, selected_vehicles: List[str], days_after_launch: int = 1) -> pd.DataFrame:
        """创建汇总统计表格数据"""
        if not selected_vehicles:
            return pd.DataFrame({'提示': ['请选择车型']})
        
        summary_data = []
        
        for vehicle in selected_vehicles:
            if vehicle not in self.business_def:
                continue
                
            # 获取车型的预售时间定义
            start_date = datetime.strptime(self.business_def[vehicle]['start'], '%Y-%m-%d')
            end_date = datetime.strptime(self.business_def[vehicle]['end'], '%Y-%m-%d')
            
            # 计算累计预售天数（从开始到结束）
            total_presale_days = (end_date - start_date).days + 1
            
            # 获取该车型的所有小订数据
            vehicle_orders = self.df[
                (self.df['车型分组'] == vehicle) & 
                (self.df['Intention_Payment_Time'].notna())
            ]
            
            # 计算累计到预售结束为止的累计预售小订数
            presale_orders = vehicle_orders[
                vehicle_orders['Intention_Payment_Time'].dt.date <= end_date.date()
            ]
            total_presale_orders = len(presale_orders)
            
            # 计算发布会后N日内的所有锁单数据
            lock_cutoff_date = end_date + timedelta(days=days_after_launch)
            vehicle_all_data = self.df[self.df['车型分组'] == vehicle].copy()
            lock_data_in_period = vehicle_all_data[
                (vehicle_all_data['Lock_Time'].notna()) &
                (pd.to_datetime(vehicle_all_data['Lock_Time']).dt.date >= end_date.date()) &
                (pd.to_datetime(vehicle_all_data['Lock_Time']).dt.date <= lock_cutoff_date.date())
            ]
            
            # 获取该车型的预售结束时间
            vehicle_end_date = datetime.strptime(self.business_def[vehicle]['end'], '%Y-%m-%d')
            
            # 小订留存锁单数：Lock_Time和Intention_Payment_Time都非空，且Intention_Payment_Time < vehicle_end_date
            retained_locks = len(lock_data_in_period[
                (lock_data_in_period['Intention_Payment_Time'].notna()) &
                (pd.to_datetime(lock_data_in_period['Intention_Payment_Time']).dt.date <= vehicle_end_date.date())
            ])
            
            # 发布会后小订锁单数：Lock_Time和Intention_Payment_Time都非空，且Intention_Payment_Time >= vehicle_end_date
            post_launch_locks = len(lock_data_in_period[
                (lock_data_in_period['Intention_Payment_Time'].notna()) &
                (pd.to_datetime(lock_data_in_period['Intention_Payment_Time']).dt.date > vehicle_end_date.date())
            ])
            
            # 直接锁单数：含有Lock_Time但没有Intention_Payment_Time的订单数
            direct_locks = len(lock_data_in_period[
                lock_data_in_period['Intention_Payment_Time'].isna()
            ])
            
            # 发布会后N日累计锁单数应该等于三个锁单数的总和
            total_lock_orders = retained_locks + post_launch_locks + direct_locks
            
            # 计算小订转化率（小订留存锁单数 / 累计预售小订数）
            conversion_rate = (retained_locks / total_presale_orders * 100) if total_presale_orders > 0 else 0
            
            summary_data.append({
                '车型': vehicle,
                '累计预售天数': total_presale_days,
                '累计预售小订数': total_presale_orders,
                f'发布会后{days_after_launch}日累计锁单数': total_lock_orders,
                '小订留存锁单数': retained_locks,
                '发布会后小订锁单数': post_launch_locks,
                '直接锁单数': direct_locks,
                '小订转化率(%)': round(conversion_rate, 2)
            })
        
        return pd.DataFrame(summary_data)
    
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
                lambda x: self.calculate_days_from_start(vehicle, x.to_pydatetime() if hasattr(x, 'to_pydatetime') else x)
            )
            
            # 过滤有效天数（>=1 且 <= max_days）
            daily_refunds = daily_refunds[
                (pd.to_numeric(daily_refunds['days_from_start'], errors='coerce') >= 1) & 
                (pd.to_numeric(daily_refunds['days_from_start'], errors='coerce') <= max_days)
            ]
            
            # 计算累计退订数
            daily_refunds = daily_refunds.sort_values('days_from_start')
            daily_refunds['cumulative_refunds'] = daily_refunds['daily_refunds'].cumsum()
            
            # 添加车型标识
            daily_refunds['车型分组'] = vehicle
            
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
            vehicle_orders = order_data[order_data['车型分组'] == vehicle].copy()
            vehicle_refunds = refund_data[refund_data['车型分组'] == vehicle].copy() if not refund_data.empty else pd.DataFrame()
            
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
            merged['车型分组'] = vehicle
            
            rate_stats.append(merged)
        
        if rate_stats:
            return pd.concat(rate_stats, ignore_index=True)
        return pd.DataFrame()
    
    def prepare_daily_refund_rate_data(self, selected_vehicles: List[str]) -> pd.DataFrame:
        """准备每日小订在当前观察时间范围的累计退订率数据 - 统一按CM2观察时间点计算"""
        if not selected_vehicles:
            return pd.DataFrame()
        
        # 首先获取CM2的当前观察时间点（参考main.py的逻辑）
        cm2_current_day = None
        if 'CM2' in selected_vehicles and 'CM2' in self.business_def:
            cm2_data = self.df[self.df['车型分组'] == 'CM2']
            if not cm2_data.empty:
                cm2_start = datetime.strptime(self.business_def['CM2']['start'], '%Y-%m-%d')
                cm2_data_copy = cm2_data.copy()
                cm2_data_copy['date'] = pd.to_datetime(cm2_data_copy['Intention_Payment_Time']).dt.date
                # 过滤掉NaN值，确保数据类型一致
                valid_dates = cm2_data_copy['date'].dropna()
                cm2_latest_date = valid_dates.max() if not valid_dates.empty else None
                if cm2_latest_date is not None:
                    cm2_current_day = (pd.to_datetime(cm2_latest_date) - pd.to_datetime(cm2_start)).days
                else:
                    cm2_current_day = None
        
        # 如果没有CM2数据，则动态计算观察时间点
        if cm2_current_day is None:
            vehicle_max_days = {}
            for vehicle in selected_vehicles:
                if vehicle not in self.business_def:
                    continue
                vehicle_data = self.df[self.df['车型分组'] == vehicle]
                if not vehicle_data.empty:
                    vehicle_start = datetime.strptime(self.business_def[vehicle]['start'], '%Y-%m-%d')
                    vehicle_end = datetime.strptime(self.business_def[vehicle]['end'], '%Y-%m-%d')
                    business_max_days = (vehicle_end - vehicle_start).days + 1
                    max_date = vehicle_data['Intention_Payment_Time'].max()
                    actual_max_day = (max_date - vehicle_start).days + 1
                    vehicle_max_days[vehicle] = min(actual_max_day, business_max_days)
            
            if not vehicle_max_days:
                return pd.DataFrame()
            observation_day = min(vehicle_max_days.values())
        else:
            observation_day = cm2_current_day
        
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
            
            # 只包含预售开始后且在观察时间范围内的数据
            daily_orders = daily_orders[
                (daily_orders['days_from_start'] >= 0) & 
                (daily_orders['days_from_start'] <= observation_day)
            ]
            
            # 计算每日订单的退订率（统一按观察截止时间点计算）
            for _, row in daily_orders.iterrows():
                day_num = row['days_from_start']
                order_date = row['order_date'].date()
                
                # 获取当日下单的订单（参考main.py的逻辑）
                daily_order_data = vehicle_data[vehicle_data['order_date'] == order_date]
                
                # 计算当日订单中有多少在观察截止时间点前退订了
                daily_refunds = 0
                if len(daily_order_data) > 0:
                    # 获取当日下单的订单ID
                    daily_order_ids = set(daily_order_data.index)
                    # 只考虑在观察截止时间点前的退订
                    refunded_before_cutoff = vehicle_data[
                        (vehicle_data['intention_refund_time'].notna()) &
                        (pd.to_datetime(vehicle_data['intention_refund_time']) <= observation_cutoff_date)
                    ]
                    refunded_order_ids = set(refunded_before_cutoff.index)
                    # 计算当日订单中有多少在观察截止时间点前退订了
                    daily_refunds = len(daily_order_ids.intersection(refunded_order_ids))
                
                # 计算退订率
                refund_rate = (daily_refunds / row['daily_orders'] * 100) if row['daily_orders'] > 0 else 0
                
                daily_rate_stats.append({
                    '车型分组': vehicle,
                    'days_from_start': day_num,
                    'daily_orders': row['daily_orders'],
                    'daily_refunds': daily_refunds,
                    'daily_refund_rate': refund_rate,
                    'observation_day': observation_day  # 记录观察时间点
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
        
        for i, vehicle in enumerate(data['车型分组'].unique()):
            vehicle_data = data[data['车型分组'] == vehicle]
            
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
        
        for i, vehicle in enumerate(data['车型分组'].unique()):
            vehicle_data = data[data['车型分组'] == vehicle]
            
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
        
        for i, vehicle in enumerate(data['车型分组'].unique()):
            vehicle_data = data[data['车型分组'] == vehicle]
            
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
        
        for vehicle in data['车型分组'].unique():
            vehicle_data = data[data['车型分组'] == vehicle].sort_values('days_from_start')
            
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
    
    def prepare_regional_summary_data(self, selected_vehicles: List[str]) -> pd.DataFrame:
        """准备分区域累计订单、退订数和退订率汇总数据 - 按观察日期计算"""
        if not selected_vehicles:
            return pd.DataFrame()
        
        if self.df is None:
            return pd.DataFrame()
        
        # 获取观察时间点（参考prepare_daily_refund_rate_data的逻辑）
        cm2_current_day = None
        if 'CM2' in selected_vehicles and 'CM2' in self.business_def:
            cm2_data = self.df[self.df['车型分组'] == 'CM2']
            if not cm2_data.empty:
                cm2_start = datetime.strptime(self.business_def['CM2']['start'], '%Y-%m-%d')
                cm2_data_copy = cm2_data.copy()
                cm2_data_copy['date'] = pd.to_datetime(cm2_data_copy['Intention_Payment_Time']).dt.date
                # 过滤掉NaN值，确保数据类型一致
                valid_dates = cm2_data_copy['date'].dropna()
                cm2_latest_date = valid_dates.max() if not valid_dates.empty else None
                if cm2_latest_date is not None:
                    cm2_current_day = (pd.to_datetime(cm2_latest_date) - pd.to_datetime(cm2_start)).days
                else:
                    cm2_current_day = None
        
        # 如果没有CM2数据，则动态计算观察时间点
        if cm2_current_day is None:
            vehicle_max_days = {}
            for vehicle in selected_vehicles:
                if vehicle not in self.business_def:
                    continue
                vehicle_data = self.df[self.df['车型分组'] == vehicle]
                if not vehicle_data.empty:
                    vehicle_start = datetime.strptime(self.business_def[vehicle]['start'], '%Y-%m-%d')
                    vehicle_end = datetime.strptime(self.business_def[vehicle]['end'], '%Y-%m-%d')
                    business_max_days = (vehicle_end - vehicle_start).days + 1
                    max_date = vehicle_data['Intention_Payment_Time'].max()
                    actual_max_day = (max_date - vehicle_start).days + 1
                    vehicle_max_days[vehicle] = min(actual_max_day, business_max_days)
            
            if not vehicle_max_days:
                return pd.DataFrame()
            observation_day = min(vehicle_max_days.values())
        else:
            observation_day = cm2_current_day
        
        regional_stats = []
        
        for vehicle in selected_vehicles:
            if vehicle not in self.business_def:
                continue
                
            vehicle_start = datetime.strptime(self.business_def[vehicle]['start'], '%Y-%m-%d')
            observation_cutoff_date = vehicle_start + timedelta(days=observation_day)
            
            # 获取车型数据，限制在观察时间点内
            vehicle_df = self.df[
                (self.df['车型分组'] == vehicle) & 
                (self.df['Intention_Payment_Time'] <= observation_cutoff_date)
            ].copy()
            
            if vehicle_df.empty:
                continue
            
            # 检查区域字段
            region_columns = ['Parent Region Name', 'License Province', 'License City']
            available_region_col = None
            for col in region_columns:
                if col in vehicle_df.columns:
                    available_region_col = col
                    break
            
            if available_region_col:
                # 按区域统计订单数
                region_orders = vehicle_df.groupby(available_region_col).size().reset_index(name='orders')
                
                # 按区域统计退订数（只统计在观察时间点内的退订）
                if 'intention_refund_time' in vehicle_df.columns:
                    refund_df = vehicle_df[
                        (vehicle_df['intention_refund_time'].notna()) &
                        (pd.to_datetime(vehicle_df['intention_refund_time']) <= observation_cutoff_date)
                    ]
                    region_refunds = refund_df.groupby(available_region_col).size().reset_index(name='refunds')
                else:
                    region_refunds = pd.DataFrame(columns=[available_region_col, 'refunds'])
                
                # 合并订单和退订数据
                region_summary = region_orders.merge(region_refunds, on=available_region_col, how='left')
                region_summary['refunds'] = region_summary['refunds'].fillna(0)
                region_summary['refund_rate'] = (region_summary['refunds'] / region_summary['orders'] * 100).round(2)
                region_summary['车型分组'] = vehicle
                region_summary['region_type'] = available_region_col
                region_summary['region_name'] = region_summary[available_region_col]
                
                regional_stats.append(region_summary)
        
        if regional_stats:
            return pd.concat(regional_stats, ignore_index=True)
        return pd.DataFrame()
    
    def create_regional_summary_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建车型对比的分区域累计订单、退订数和退订率汇总表格"""
        if data.empty:
            return pd.DataFrame({'提示': ['暂无区域数据']})
        
        # 获取所有车型和区域
        vehicles = sorted(data['车型分组'].unique())
        regions = data[['region_type', 'region_name']].drop_duplicates().sort_values(['region_type', 'region_name'])
        
        # 创建对比表格数据
        table_data = []
        
        for _, region_row in regions.iterrows():
            region_type = region_row['region_type']
            region_name = region_row['region_name']
            
            # 基础行数据
            row_data = {
                '区域类型': region_type,
                '区域名称': region_name
            }
            
            # 为每个车型添加订单数/退订数和退订率列
            for vehicle in vehicles:
                vehicle_data = data[
                    (data['车型分组'] == vehicle) & 
                    (data['region_type'] == region_type) & 
                    (data['region_name'] == region_name)
                ]
                
                if not vehicle_data.empty:
                    row = vehicle_data.iloc[0]
                    refund_rate = row['refund_rate']
                    
                    # 添加订单数/退订数列
                    row_data[f'{vehicle} 小订数/退订数'] = f"{int(row['orders']):,}/{int(row['refunds']):,}"
                    
                    # 添加退订率列（带emoji）
                    if refund_rate < 5:
                        trend_emoji = "🟢"
                    elif refund_rate < 10:
                        trend_emoji = "🟡"
                    else:
                        trend_emoji = "🔴"
                    
                    row_data[f'{vehicle} 退订率'] = f"{refund_rate:.1f}% {trend_emoji}"
                else:
                    row_data[f'{vehicle} 小订数/退订数'] = "-"
                    row_data[f'{vehicle} 退订率'] = "-"
            
            table_data.append(row_data)
        
        # 创建DataFrame并排序
        table_df = pd.DataFrame(table_data)
        
        return table_df
    
    def prepare_city_summary_data(self, selected_vehicles: List[str]) -> pd.DataFrame:
        """准备分城市累计订单、退订数和退订率数据 - 按观察日期计算"""
        if not selected_vehicles:
            return pd.DataFrame()
        
        # 获取观察时间点（参考prepare_daily_refund_rate_data的逻辑）
        cm2_current_day = None
        if 'CM2' in selected_vehicles and 'CM2' in self.business_def:
            cm2_data = self.df[self.df['车型分组'] == 'CM2']
            if not cm2_data.empty:
                cm2_start = datetime.strptime(self.business_def['CM2']['start'], '%Y-%m-%d')
                cm2_data_copy = cm2_data.copy()
                cm2_data_copy['date'] = pd.to_datetime(cm2_data_copy['Intention_Payment_Time']).dt.date
                # 过滤掉NaN值，确保数据类型一致
                valid_dates = cm2_data_copy['date'].dropna()
                cm2_latest_date = valid_dates.max() if not valid_dates.empty else None
                if cm2_latest_date is not None:
                    cm2_current_day = (pd.to_datetime(cm2_latest_date) - pd.to_datetime(cm2_start)).days
                else:
                    cm2_current_day = None
        
        # 如果没有CM2数据，则动态计算观察时间点
        if cm2_current_day is None:
            vehicle_max_days = {}
            for vehicle in selected_vehicles:
                if vehicle not in self.business_def:
                    continue
                vehicle_data = self.df[self.df['车型分组'] == vehicle]
                if not vehicle_data.empty:
                    vehicle_start = datetime.strptime(self.business_def[vehicle]['start'], '%Y-%m-%d')
                    vehicle_end = datetime.strptime(self.business_def[vehicle]['end'], '%Y-%m-%d')
                    business_max_days = (vehicle_end - vehicle_start).days + 1
                    max_date = vehicle_data['Intention_Payment_Time'].max()
                    actual_max_day = (max_date - vehicle_start).days + 1
                    vehicle_max_days[vehicle] = min(actual_max_day, business_max_days)
            
            if not vehicle_max_days:
                return pd.DataFrame()
            observation_day = min(vehicle_max_days.values())
        else:
            observation_day = cm2_current_day
        
        city_stats = []
        
        for vehicle in selected_vehicles:
            if vehicle not in self.business_def:
                continue
                
            vehicle_start = datetime.strptime(self.business_def[vehicle]['start'], '%Y-%m-%d')
            observation_cutoff_date = vehicle_start + timedelta(days=observation_day)
            
            # 获取车型数据，限制在观察时间点内
            vehicle_df = self.df[
                (self.df['车型分组'] == vehicle) & 
                (self.df['Intention_Payment_Time'] <= observation_cutoff_date)
            ].copy()
            
            if vehicle_df.empty:
                continue
            
            # 按城市统计订单数
            city_orders = vehicle_df.groupby('License City').size().reset_index(name='orders')
            
            # 按城市统计退订数（只统计在观察时间点内的退订）
            if 'intention_refund_time' in vehicle_df.columns:
                refund_df = vehicle_df[
                    (vehicle_df['intention_refund_time'].notna()) &
                    (pd.to_datetime(vehicle_df['intention_refund_time']) <= observation_cutoff_date)
                ]
                city_refunds = refund_df.groupby('License City').size().reset_index(name='refunds')
            else:
                city_refunds = pd.DataFrame(columns=['License City', 'refunds'])
            
            # 合并订单和退订数据
            city_summary = city_orders.merge(city_refunds, on='License City', how='left')
            city_summary['refunds'] = city_summary['refunds'].fillna(0)
            city_summary['refund_rate'] = (city_summary['refunds'] / city_summary['orders'] * 100).round(2)
            city_summary['车型分组'] = vehicle
            city_summary.rename(columns={'License City': 'license_city'}, inplace=True)
            
            city_stats.append(city_summary)
        
        if city_stats:
            return pd.concat(city_stats, ignore_index=True)
        return pd.DataFrame()
    
    def create_city_summary_table(self, data: pd.DataFrame, order_threshold: List[float] = [100, 2000]) -> pd.DataFrame:
        """创建车型对比的分城市累计订单、退订数和退订率汇总表格"""
        if data.empty:
            return pd.DataFrame({'提示': ['暂无城市数据']})
        
        # 获取所有车型和城市
        vehicles = sorted(data['车型分组'].unique())
        cities = sorted(data['license_city'].unique())
        
        # 创建对比表格数据
        table_data = []
        
        # 解析范围参数
        min_threshold, max_threshold = order_threshold[0], order_threshold[1]
        
        for city in cities:
            # 计算该城市所有车型的总小订数，用于筛选
            city_total_orders = data[data['license_city'] == city]['orders'].sum()
            
            # 如果总小订数不在范围内，跳过该城市
            if city_total_orders < min_threshold or city_total_orders > max_threshold:
                continue
            
            # 基础行数据
            row_data = {
                '城市': city
            }
            
            # 先收集所有车型的数据
            vehicle_stats = {}
            for vehicle in vehicles:
                vehicle_data = data[
                    (data['车型分组'] == vehicle) & 
                    (data['license_city'] == city)
                ]
                
                if not vehicle_data.empty:
                    row = vehicle_data.iloc[0]
                    orders = int(row['orders'])
                    refunds = int(row['refunds'])
                    refund_rate = row['refund_rate']
                    
                    # 退订率emoji
                    if refund_rate < 5:
                        trend_emoji = "🟢"
                    elif refund_rate < 10:
                        trend_emoji = "🟡"
                    else:
                        trend_emoji = "🔴"
                    
                    vehicle_stats[vehicle] = {
                        'orders': f"{orders:,}",
                        'refunds': f"{refunds:,}",
                        'refund_rate': f"{refund_rate:.1f}% {trend_emoji}"
                    }
                else:
                    vehicle_stats[vehicle] = {
                        'orders': "-",
                        'refunds': "-",
                        'refund_rate': "-"
                    }
            
            # 按指标类型分组添加列：先小订数，再退订数，最后退订率
            # 添加小订数列
            for vehicle in vehicles:
                row_data[f'{vehicle} 小订数'] = vehicle_stats[vehicle]['orders']
            
            # 添加退订数列
            for vehicle in vehicles:
                row_data[f'{vehicle} 退订数'] = vehicle_stats[vehicle]['refunds']
            
            # 添加退订率列
            for vehicle in vehicles:
                row_data[f'{vehicle} 退订率'] = vehicle_stats[vehicle]['refund_rate']
            
            table_data.append(row_data)
        
        # 创建DataFrame
        if not table_data:
            return pd.DataFrame({'提示': [f'没有小订数大于{order_threshold}的城市']})
        
        table_df = pd.DataFrame(table_data)
        
        # 按城市名称排序
        table_df = table_df.sort_values('城市')
        
        return table_df
    
    def prepare_lock_data(self, selected_vehicles: List[str], n_days: int = 30) -> pd.DataFrame:
        """准备锁单数据
        
        第N日计算逻辑：
        - 基于各车型预售结束日期 + N天来计算目标日期范围
        - 统计每日的锁单数量（有Lock_Time的订单）
        - 第N日是指预售结束后的第N日
        """
        try:
            # 获取各车型的预售结束日期
            end_dates = {}
            for vehicle in selected_vehicles:
                if vehicle in self.business_def:
                    end_date = datetime.strptime(self.business_def[vehicle]['end'], '%Y-%m-%d')
                    end_dates[vehicle] = end_date
            
            # 准备锁单数据
            lock_data = []
            for vehicle in selected_vehicles:
                if vehicle not in end_dates:
                    continue
                    
                vehicle_data = self.df[self.df['车型分组'] == vehicle].copy()
                vehicle_end_date = end_dates[vehicle]
                
                # 获取有Lock_Time的数据
                lock_orders = vehicle_data[vehicle_data['Lock_Time'].notna()].copy()
                
                if lock_orders.empty:
                    continue
                
                # 确保Lock_Time为datetime类型
                lock_orders['Lock_Time'] = pd.to_datetime(lock_orders['Lock_Time'])
                
                # 计算每日锁单数据，从预售结束当天开始到第N日
                start_date = vehicle_end_date  # 预售结束当天（第0日）
                end_date = vehicle_end_date + timedelta(days=n_days)  # 预售结束后第N日
                
                # 按日期分组统计锁单数
                daily_locks = []
                current_date = start_date
                cumulative_locks = 0
                
                while current_date <= end_date:
                    # 统计当日锁单数
                    daily_lock_count = len(lock_orders[lock_orders['Lock_Time'].dt.date == current_date.date()])
                    cumulative_locks += daily_lock_count
                    
                    # 计算从预售结束的天数（第N日）
                    days_from_end = self.calculate_days_from_end(vehicle, current_date)
                    
                    # 只保留有实际锁单发生的日期点
                    if daily_lock_count > 0:
                        daily_locks.append({
                            'vehicle': vehicle,
                            'date': current_date,
                            'days_from_end': days_from_end,
                            'daily_locks': daily_lock_count,
                            'cumulative_locks': cumulative_locks
                        })
                    
                    current_date += timedelta(days=1)
                
                lock_data.extend(daily_locks)
            
            return pd.DataFrame(lock_data)
                
        except Exception as e:
            logger.error(f"准备锁单数据时出错: {e}")
            return pd.DataFrame()
    
    def prepare_lock_conversion_data(self, selected_vehicles: List[str], n_days: int = 30) -> pd.DataFrame:
        """准备小订转化率数据"""
        try:
            conversion_data = []
            
            for vehicle in selected_vehicles:
                if vehicle not in self.business_def:
                    continue
                    
                vehicle_data = self.df[self.df['车型分组'] == vehicle].copy()
                
                # 获取预售周期内的累计小订数（分母）
                start_date = datetime.strptime(self.business_def[vehicle]['start'], '%Y-%m-%d')
                end_date = datetime.strptime(self.business_def[vehicle]['end'], '%Y-%m-%d')
                
                # 预售期内的小订数
                presale_orders = vehicle_data[
                    (pd.to_datetime(vehicle_data['Intention_Payment_Time']) >= start_date) &
                    (pd.to_datetime(vehicle_data['Intention_Payment_Time']) <= end_date)
                ]
                total_presale_orders = len(presale_orders)
                
                # 计算锁单数（分子）：同时满足Lock_Time、Intention_Payment_Time、Intention_Payment_Time小于最大日期
                lock_orders = vehicle_data[
                    (vehicle_data['Lock_Time'].notna()) &
                    (vehicle_data['Intention_Payment_Time'].notna()) &
                    (pd.to_datetime(vehicle_data['Intention_Payment_Time']) <= end_date)
                ]
                
                # 按Lock_Time分组计算累计转化率
                if not lock_orders.empty:
                    # 确保Lock_Time是datetime类型
                    lock_orders = lock_orders.copy()
                    lock_orders['Lock_Time'] = pd.to_datetime(lock_orders['Lock_Time'], errors='coerce')
                    # 过滤掉无效日期
                    lock_orders = lock_orders[lock_orders['Lock_Time'].notna()]
                    if not lock_orders.empty:
                        lock_orders['Lock_Date'] = lock_orders['Lock_Time'].dt.date
                        daily_locks = lock_orders.groupby('Lock_Date').size().reset_index(name='daily_locks')
                        daily_locks['cumulative_locks'] = daily_locks['daily_locks'].cumsum()
                        
                        # 计算转化率
                        if total_presale_orders > 0:
                            daily_locks['conversion_rate'] = (daily_locks['cumulative_locks'] / total_presale_orders) * 100
                        else:
                            daily_locks['conversion_rate'] = 0
                        
                        daily_locks['vehicle'] = vehicle
                        
                        # 计算从预售结束的天数
                        daily_locks['days_from_end'] = daily_locks['Lock_Date'].apply(lambda x: self.calculate_days_from_end(vehicle, datetime.combine(x, datetime.min.time())))
                        
                        # 过滤掉负数天数（锁单时间在预售结束前的数据）和超过N天的数据
                        daily_locks = daily_locks[
                            (daily_locks['days_from_end'] >= 0) & 
                            (daily_locks['days_from_end'] <= n_days)
                        ]
                        
                        if not daily_locks.empty:
                            conversion_data.append(daily_locks)
            
            if conversion_data:
                result_df = pd.concat(conversion_data, ignore_index=True)
                return result_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"准备转化率数据时出错: {e}")
            return pd.DataFrame()
    
    def create_cumulative_lock_chart(self, data: pd.DataFrame) -> go.Figure:
        """创建累计锁单数对比图"""
        fig = go.Figure()
        
        if data.empty:
            fig.add_annotation(
                text="暂无锁单数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, vehicle in enumerate(data['vehicle'].unique()):
            vehicle_data = data[data['vehicle'] == vehicle].sort_values('days_from_end')
            
            # 过滤掉累计锁单数为0的数据点，避免绘制无意义的折线
            vehicle_data = vehicle_data[vehicle_data['cumulative_locks'] > 0]
            
            if not vehicle_data.empty:
                # 进一步优化：只显示有实际锁单发生的关键节点
                # 找到第一个有锁单的日期作为起点
                first_lock_day = vehicle_data['days_from_end'].min()
                vehicle_data = vehicle_data[vehicle_data['days_from_end'] >= first_lock_day]
                
                fig.add_trace(go.Scatter(
                    x=vehicle_data['days_from_end'],
                    y=vehicle_data['cumulative_locks'],
                    mode='lines+markers',
                    name=f'{vehicle}',
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=6),
                    hovertemplate=f'<b>{vehicle}</b><br>' +
                                 '第%{x}日（预售结束当天为第0日）<br>' +
                                 '累计锁单数: %{y}<br>' +
                                 '<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(
                text='累计锁单数对比图',
                x=0.5,
                font=dict(size=16, color='#2c3e50')
            ),
            xaxis=dict(
                title='第N日',
                gridcolor='lightgray',
                showgrid=True
            ),
            yaxis=dict(
                title='累计锁单数',
                gridcolor='lightgray',
                showgrid=True
            ),
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="#2c3e50")
        )
        
        return fig
    
    def create_lock_conversion_chart(self, data: pd.DataFrame) -> go.Figure:
        """创建累计小订转化率对比图"""
        fig = go.Figure()
        
        if data.empty:
            fig.add_annotation(
                text="暂无转化率数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, vehicle in enumerate(data['vehicle'].unique()):
            vehicle_data = data[data['vehicle'] == vehicle].sort_values('days_from_end')
            
            fig.add_trace(go.Scatter(
                x=vehicle_data['days_from_end'],
                y=vehicle_data['conversion_rate'],
                mode='lines+markers',
                name=f'{vehicle}',
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=6),
                hovertemplate=f'<b>{vehicle}</b><br>' +
                             '第%{x}日（预售结束当天为第0日）<br>' +
                             '转化率: %{y:.2f}%<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text='累计小订转化率对比图',
                x=0.5,
                font=dict(size=16, color='#2c3e50')
            ),
            xaxis=dict(
                title='第N日',
                gridcolor='lightgray',
                showgrid=True
            ),
            yaxis=dict(
                title='转化率 (%)',
                gridcolor='lightgray',
                showgrid=True
            ),
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="#2c3e50")
        )
        
        return fig
    
    def create_daily_lock_chart(self, data: pd.DataFrame) -> go.Figure:
        """创建每日锁单数对比图"""
        fig = go.Figure()
        
        if data.empty:
            fig.add_annotation(
                text="暂无每日锁单数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, vehicle in enumerate(data['vehicle'].unique()):
            vehicle_data = data[data['vehicle'] == vehicle].sort_values('days_from_end')
            
            fig.add_trace(go.Bar(
                x=vehicle_data['days_from_end'],
                y=vehicle_data['daily_locks'],
                name=f'{vehicle}',
                marker_color=colors[i % len(colors)],
                opacity=0.8,
                hovertemplate=f'<b>{vehicle}</b><br>' +
                             '第%{x}日（预售结束当天为第0日）<br>' +
                             '当日锁单数: %{y}<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text='每日锁单数对比图',
                x=0.5,
                font=dict(size=16, color='#2c3e50')
            ),
            xaxis=dict(
                title='第N日',
                gridcolor='lightgray',
                showgrid=True
            ),
            yaxis=dict(
                title='每日锁单数',
                gridcolor='lightgray',
                showgrid=True
            ),
            barmode='group',
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="#2c3e50")
        )
        
        return fig
    
    def prepare_daily_lock_change_data(self, selected_vehicles: List[str], n_days: int = 30) -> pd.DataFrame:
        """准备每日锁单数环比变化数据"""
        try:
            # 先获取基础锁单数据
            lock_data = self.prepare_lock_data(selected_vehicles, n_days)
            
            if lock_data.empty:
                return pd.DataFrame()
            
            change_data = []
            
            for vehicle in lock_data['vehicle'].unique():
                vehicle_data = lock_data[lock_data['vehicle'] == vehicle].sort_values('days_from_end')
                
                # 计算环比变化率
                vehicle_data = vehicle_data.copy()
                vehicle_data['prev_daily_locks'] = vehicle_data['daily_locks'].shift(1)
                vehicle_data['change_rate'] = ((vehicle_data['daily_locks'] - vehicle_data['prev_daily_locks']) / 
                                             vehicle_data['prev_daily_locks'] * 100).fillna(0)
                
                # 处理无穷大值
                vehicle_data['change_rate'] = vehicle_data['change_rate'].replace([np.inf, -np.inf], 0)
                
                change_data.append(vehicle_data)
            
            if change_data:
                result_df = pd.concat(change_data, ignore_index=True)
                return result_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"准备每日锁单数环比变化数据时出错: {e}")
            return pd.DataFrame()
    
    def create_daily_lock_change_chart(self, data: pd.DataFrame) -> go.Figure:
        """创建每日锁单数环比变化图"""
        fig = go.Figure()
        
        if data.empty:
            fig.add_annotation(
                text="暂无环比变化数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, vehicle in enumerate(data['vehicle'].unique()):
            vehicle_data = data[data['vehicle'] == vehicle].sort_values('days_from_end')
            
            # 过滤掉第一天（没有环比数据）
            vehicle_data = vehicle_data[vehicle_data['days_from_end'] > 0]
            
            # 过滤掉每日锁单数为0的数据点，避免绘制无意义的折线
            vehicle_data = vehicle_data[vehicle_data['daily_locks'] > 0]
            
            if not vehicle_data.empty:
                fig.add_trace(go.Scatter(
                    x=vehicle_data['days_from_end'],
                    y=vehicle_data['change_rate'],
                    mode='lines+markers',
                    name=f'{vehicle}',
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=6),
                    hovertemplate=f'<b>{vehicle}</b><br>' +
                                 '第%{x}日（预售结束当天为第0日）<br>' +
                                 '环比变化: %{y:.1f}%<br>' +
                                 '<extra></extra>'
                ))
        
        # 添加零线
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
        
        fig.update_layout(
            title=dict(
                text='每日锁单数环比变化图',
                x=0.5,
                font=dict(size=16, color='#2c3e50')
            ),
            xaxis=dict(
                title='第N日',
                gridcolor='lightgray',
                showgrid=True
            ),
            yaxis=dict(
                title='环比变化率 (%)',
                gridcolor='lightgray',
                showgrid=True
            ),
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="#2c3e50")
        )
        
        return fig
    
    def prepare_product_name_lock_data(self, selected_vehicles: List[str], start_date: str = '', end_date: str = '', 
                                     product_types: List[str] = None, lock_start_date: str = '', lock_end_date: str = '', lock_n_days: int = 30,
                                     weekend_lock_filter: str = "全部") -> pd.DataFrame:
        """准备Product Name锁单统计数据（支持多种筛选条件）"""
        try:
            if self.df.empty:
                return pd.DataFrame()
            
            # 基础筛选：车型分组
            filtered_data = self.df[self.df['车型分组'].isin(selected_vehicles)].copy()
            
            if filtered_data.empty:
                return pd.DataFrame()
            
            # 小订时间范围筛选（基于Intention_Payment_Time）
            if start_date and 'Intention_Payment_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Intention_Payment_Time'] >= start_date]
            if end_date and 'Intention_Payment_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Intention_Payment_Time'] <= end_date]
            
            # 锁单时间范围筛选（基于Lock_Time）
            if lock_start_date and 'Lock_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Lock_Time'] >= lock_start_date]
            if lock_end_date and 'Lock_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Lock_Time'] <= lock_end_date]
            
            # 基于锁单后N天的筛选（基于business_definition.json最大值+N天）
            if lock_n_days and 'Lock_Time' in filtered_data.columns and hasattr(self, 'business_def'):
                for vehicle in selected_vehicles:
                    if vehicle in self.business_def:
                        # 计算该车型的预售结束日期 + N天
                        end_date_str = self.business_def[vehicle]['end']
                        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                        max_lock_date = end_date + timedelta(days=lock_n_days)
                        
                        # 筛选该车型在指定时间范围内的锁单数据
                        vehicle_mask = (filtered_data['车型分组'] == vehicle) & (filtered_data['Lock_Time'] <= max_lock_date.strftime('%Y-%m-%d'))
                        if vehicle == selected_vehicles[0]:  # 第一个车型，直接赋值
                            time_filtered_data = filtered_data[vehicle_mask]
                        else:  # 后续车型，合并数据
                            time_filtered_data = pd.concat([time_filtered_data, filtered_data[vehicle_mask]], ignore_index=True)
                
                if 'time_filtered_data' in locals():
                    filtered_data = time_filtered_data
            
            # 产品分类筛选
            if product_types and 'Product Name' in filtered_data.columns:
                product_mask = pd.Series([False] * len(filtered_data), index=filtered_data.index)
                
                for category in product_types:
                    if category == "增程":
                        # 产品名称中包含"新一代"和数字52或66的为增程
                        category_mask = (
                            filtered_data['Product Name'].str.contains('新一代', na=False) & 
                            (filtered_data['Product Name'].str.contains('52', na=False) | 
                             filtered_data['Product Name'].str.contains('66', na=False))
                        )
                    elif category == "纯电":
                        # 其他产品为纯电
                        category_mask = ~(
                            filtered_data['Product Name'].str.contains('新一代', na=False) & 
                            (filtered_data['Product Name'].str.contains('52', na=False) | 
                             filtered_data['Product Name'].str.contains('66', na=False))
                        )
                    else:
                        continue
                    
                    product_mask = product_mask | category_mask
                
                filtered_data = filtered_data[product_mask]
            
            # 周末锁单筛选
            if weekend_lock_filter != "全部" and 'Lock_Time' in filtered_data.columns:
                # 将Lock_Time转换为datetime格式
                filtered_data['Lock_Time_dt'] = pd.to_datetime(filtered_data['Lock_Time'], errors='coerce')
                
                if weekend_lock_filter == "仅周末锁单":
                    # 筛选周末锁单（周六=5, 周日=6）
                    weekend_mask = filtered_data['Lock_Time_dt'].dt.weekday.isin([5, 6])
                    filtered_data = filtered_data[weekend_mask]
                elif weekend_lock_filter == "仅工作日锁单":
                    # 筛选工作日锁单（周一到周五=0-4）
                    weekday_mask = filtered_data['Lock_Time_dt'].dt.weekday.isin([0, 1, 2, 3, 4])
                    filtered_data = filtered_data[weekday_mask]
                
                # 删除临时列
                filtered_data = filtered_data.drop('Lock_Time_dt', axis=1)
            
            # 最终筛选：只保留有锁单数据的记录用于统计
            if 'Lock_Time' in filtered_data.columns:
                lock_data = filtered_data[filtered_data['Lock_Time'].notna()].copy()
            else:
                return pd.DataFrame()
            
            if lock_data.empty:
                return pd.DataFrame()
            
            # 按车型分组和Product Name统计锁单数
            result_data = []
            
            for vehicle in selected_vehicles:
                vehicle_data = lock_data[lock_data['车型分组'] == vehicle]
                if vehicle_data.empty:
                    continue
                
                # 统计该车型的总锁单数
                total_locks = len(vehicle_data)
                
                # 按Product Name分组统计
                product_stats = vehicle_data.groupby('Product Name').size().reset_index(name='锁单数')
                product_stats['车型'] = vehicle
                product_stats['锁单占比(%)'] = (product_stats['锁单数'] / total_locks * 100).round(2)
                
                result_data.append(product_stats)
            
            if result_data:
                final_df = pd.concat(result_data, ignore_index=True)
                # 按车型和锁单数排序
                final_df = final_df.sort_values(['车型', '锁单数'], ascending=[True, False])
                return final_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"准备Product Name锁单数据时出错: {e}")
            return pd.DataFrame()
    
    def prepare_channel_lock_data(self, selected_vehicles: List[str], start_date: str = '', end_date: str = '', 
                                product_types: List[str] = None, lock_start_date: str = '', lock_end_date: str = '', lock_n_days: int = 30,
                                weekend_lock_filter: str = "全部") -> pd.DataFrame:
        """准备中间渠道锁单统计数据（支持多种筛选条件）"""
        try:
            if self.df.empty:
                return pd.DataFrame()
            
            # 基础筛选：车型分组
            filtered_data = self.df[self.df['车型分组'].isin(selected_vehicles)].copy()
            
            if filtered_data.empty:
                return pd.DataFrame()
            
            # 小订时间范围筛选（基于Intention_Payment_Time）
            if start_date and 'Intention_Payment_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Intention_Payment_Time'] >= start_date]
            if end_date and 'Intention_Payment_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Intention_Payment_Time'] <= end_date]
            
            # 锁单时间范围筛选（基于Lock_Time）
            if lock_start_date and 'Lock_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Lock_Time'] >= lock_start_date]
            if lock_end_date and 'Lock_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Lock_Time'] <= lock_end_date]
            
            # 基于锁单后N天的筛选（基于business_definition.json最大值+N天）
            if lock_n_days and 'Lock_Time' in filtered_data.columns and hasattr(self, 'business_def'):
                for vehicle in selected_vehicles:
                    if vehicle in self.business_def:
                        # 计算该车型的预售结束日期 + N天
                        end_date_str = self.business_def[vehicle]['end']
                        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                        max_lock_date = end_date + timedelta(days=lock_n_days)
                        
                        # 筛选该车型在指定时间范围内的锁单数据
                        vehicle_mask = (filtered_data['车型分组'] == vehicle) & (filtered_data['Lock_Time'] <= max_lock_date.strftime('%Y-%m-%d'))
                        if vehicle == selected_vehicles[0]:  # 第一个车型，直接赋值
                            time_filtered_data = filtered_data[vehicle_mask]
                        else:  # 后续车型，合并数据
                            time_filtered_data = pd.concat([time_filtered_data, filtered_data[vehicle_mask]], ignore_index=True)
                
                if 'time_filtered_data' in locals():
                    filtered_data = time_filtered_data
            
            # 产品分类筛选
            if product_types and 'Product Name' in filtered_data.columns:
                product_mask = pd.Series([False] * len(filtered_data), index=filtered_data.index)
                
                for category in product_types:
                    if category == "增程":
                        # 产品名称中包含"新一代"和数字52或66的为增程
                        category_mask = (
                            filtered_data['Product Name'].str.contains('新一代', na=False) & 
                            (filtered_data['Product Name'].str.contains('52', na=False) | 
                             filtered_data['Product Name'].str.contains('66', na=False))
                        )
                    elif category == "纯电":
                        # 其他产品为纯电
                        category_mask = ~(
                            filtered_data['Product Name'].str.contains('新一代', na=False) & 
                            (filtered_data['Product Name'].str.contains('52', na=False) | 
                             filtered_data['Product Name'].str.contains('66', na=False))
                        )
                    else:
                        continue
                    
                    product_mask = product_mask | category_mask
                
                filtered_data = filtered_data[product_mask]
            
            # 周末锁单筛选
            if weekend_lock_filter != "全部" and 'Lock_Time' in filtered_data.columns:
                # 将Lock_Time转换为datetime格式
                filtered_data['Lock_Time_dt'] = pd.to_datetime(filtered_data['Lock_Time'], errors='coerce')
                
                if weekend_lock_filter == "仅周末锁单":
                    # 筛选周末锁单（周六=5, 周日=6）
                    weekend_mask = filtered_data['Lock_Time_dt'].dt.weekday.isin([5, 6])
                    filtered_data = filtered_data[weekend_mask]
                elif weekend_lock_filter == "仅工作日锁单":
                    # 筛选工作日锁单（周一到周五=0-4）
                    weekday_mask = filtered_data['Lock_Time_dt'].dt.weekday.isin([0, 1, 2, 3, 4])
                    filtered_data = filtered_data[weekday_mask]
                
                # 删除临时列
                filtered_data = filtered_data.drop('Lock_Time_dt', axis=1)
            
            # 最终筛选：只保留有锁单数据的记录用于统计
            if 'Lock_Time' in filtered_data.columns:
                lock_data = filtered_data[filtered_data['Lock_Time'].notna()].copy()
            else:
                return pd.DataFrame()
            
            if lock_data.empty:
                return pd.DataFrame()
            
            # 按渠道分组统计各车型的锁单数，调整为车型对比格式
            if 'first_middle_channel_name' not in lock_data.columns:
                return pd.DataFrame()
            
            # 获取所有渠道
            all_channels = lock_data['first_middle_channel_name'].dropna().unique()
            
            # 构建车型对比表格数据
            result_data = []
            
            for channel in all_channels:
                channel_data = lock_data[lock_data['first_middle_channel_name'] == channel]
                
                row_data = {'渠道名称': channel if pd.notna(channel) else '未知渠道'}
                
                # 为每个车型统计锁单数和占比
                for vehicle in selected_vehicles:
                    vehicle_channel_data = channel_data[channel_data['车型分组'] == vehicle]
                    vehicle_total_data = lock_data[lock_data['车型分组'] == vehicle]
                    
                    lock_count = len(vehicle_channel_data)
                    total_count = len(vehicle_total_data)
                    lock_ratio = (lock_count / total_count * 100) if total_count > 0 else 0
                    
                    row_data[f'{vehicle}锁单数'] = lock_count
                    row_data[f'{vehicle}锁单占比'] = round(lock_ratio, 2)
                
                result_data.append(row_data)
            
            if result_data:
                final_df = pd.DataFrame(result_data)
                # 按第一个车型的锁单数排序
                if selected_vehicles:
                    sort_column = f'{selected_vehicles[0]}锁单数'
                    final_df = final_df.sort_values(sort_column, ascending=False)
                return final_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"准备中间渠道锁单数据时出错: {e}")
            return pd.DataFrame()

    def create_product_name_lock_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建Product Name锁单统计表格"""
        if data.empty:
            return pd.DataFrame({'提示': ['暂无锁单数据']})
        
        try:
            # 重新组织表格格式
            table_data = []
            
            for _, row in data.iterrows():
                product_name = row['Product Name']
                price = self.get_vehicle_price(product_name)
                
                table_row = {
                    '车型': row['车型'],
                    'Product Name': product_name,
                    '价格': price,
                    '锁单数': f"{int(row['锁单数']):,}",
                    '锁单占比': f"{row['锁单占比(%)']}%"
                }
                table_data.append(table_row)
            
            return pd.DataFrame(table_data)
            
        except Exception as e:
            logger.error(f"创建Product Name锁单表格时出错: {e}")
            return pd.DataFrame({'错误': [f'表格生成失败: {str(e)}']})

    def create_channel_lock_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建中间渠道锁单统计表格（车型对比格式）"""
        if data.empty:
            return pd.DataFrame({'提示': ['暂无锁单数据']})
        
        try:
            # 重新组织表格格式，实现车型对比和高亮功能
            table_data = []
            
            for _, row in data.iterrows():
                channel_name = row['渠道名称']
                
                # 构建基础行数据
                table_row = {'渠道名称': channel_name}
                
                # 获取所有车型相关的列
                vehicle_columns = [col for col in data.columns if col.endswith('锁单数') or col.endswith('锁单占比')]
                
                # 分别处理锁单数和锁单占比的高亮
                lock_count_columns = [col for col in vehicle_columns if col.endswith('锁单数')]
                lock_ratio_columns = [col for col in vehicle_columns if col.endswith('锁单占比')]
                
                # 找出锁单数最大值
                if lock_count_columns:
                    lock_count_values = [row[col] for col in lock_count_columns]
                    max_lock_count = max(lock_count_values) if lock_count_values else 0
                
                # 找出锁单占比最大值
                if lock_ratio_columns:
                    lock_ratio_values = [row[col] for col in lock_ratio_columns]
                    max_lock_ratio = max(lock_ratio_values) if lock_ratio_values else 0
                
                # 添加车型数据，对最大值进行高亮
                for col in vehicle_columns:
                    value = row[col]
                    if col.endswith('锁单数'):
                        formatted_value = f"{int(value):,}"
                        # 如果是最大值且大于0，添加红色高亮
                        if value == max_lock_count and value > 0 and len(lock_count_columns) > 1:
                            formatted_value = f'<span style="color: red; font-weight: bold;">{formatted_value}</span>'
                        table_row[col] = formatted_value
                    elif col.endswith('锁单占比'):
                        formatted_value = f"{value}%"
                        # 如果是最大值且大于0，添加红色高亮
                        if value == max_lock_ratio and value > 0 and len(lock_ratio_columns) > 1:
                            formatted_value = f'<span style="color: red; font-weight: bold;">{formatted_value}</span>'
                        table_row[col] = formatted_value
                
                table_data.append(table_row)
            
            return pd.DataFrame(table_data)
            
        except Exception as e:
            logger.error(f"创建中间渠道锁单表格时出错: {e}")
            return pd.DataFrame({'错误': [f'表格生成失败: {str(e)}']})

    def prepare_age_lock_data(self, selected_vehicles: List[str], start_date: str = '', end_date: str = '', 
                            product_types: List[str] = None, lock_start_date: str = '', lock_end_date: str = '', lock_n_days: int = 30, include_unknown: bool = True,
                            weekend_lock_filter: str = "全部") -> pd.DataFrame:
        """准备买家年龄锁单统计数据（支持多种筛选条件）"""
        try:
            if self.df.empty:
                return pd.DataFrame()
            
            # 基础筛选：车型分组
            filtered_data = self.df[self.df['车型分组'].isin(selected_vehicles)].copy()
            
            if filtered_data.empty:
                return pd.DataFrame()
            
            # 小订时间范围筛选（基于Intention_Payment_Time）
            if start_date and 'Intention_Payment_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Intention_Payment_Time'] >= start_date]
            if end_date and 'Intention_Payment_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Intention_Payment_Time'] <= end_date]
            
            # 锁单时间范围筛选（基于Lock_Time）
            if lock_start_date and 'Lock_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Lock_Time'] >= lock_start_date]
            if lock_end_date and 'Lock_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Lock_Time'] <= lock_end_date]
            
            # 基于锁单后N天的筛选（基于business_definition.json最大值+N天）
            if lock_n_days and 'Lock_Time' in filtered_data.columns and hasattr(self, 'business_def'):
                for vehicle in selected_vehicles:
                    if vehicle in self.business_def:
                        # 计算该车型的预售结束日期 + N天
                        end_date_str = self.business_def[vehicle]['end']
                        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                        max_lock_date = end_date + timedelta(days=lock_n_days)
                        
                        # 筛选该车型在指定时间范围内的锁单数据
                        vehicle_mask = (filtered_data['车型分组'] == vehicle) & (filtered_data['Lock_Time'] <= max_lock_date.strftime('%Y-%m-%d'))
                        if vehicle == selected_vehicles[0]:  # 第一个车型，直接赋值
                            time_filtered_data = filtered_data[vehicle_mask]
                        else:  # 后续车型，合并数据
                            time_filtered_data = pd.concat([time_filtered_data, filtered_data[vehicle_mask]], ignore_index=True)
                
                if 'time_filtered_data' in locals():
                    filtered_data = time_filtered_data
            
            # 产品分类筛选
            if product_types and 'Product Name' in filtered_data.columns:
                product_mask = pd.Series([False] * len(filtered_data), index=filtered_data.index)
                
                for category in product_types:
                    if category == "增程":
                        # 产品名称中包含"新一代"和数字52或66的为增程
                        category_mask = (
                            filtered_data['Product Name'].str.contains('新一代', na=False) & 
                            (filtered_data['Product Name'].str.contains('52', na=False) | 
                             filtered_data['Product Name'].str.contains('66', na=False))
                        )
                    elif category == "纯电":
                        # 其他产品为纯电
                        category_mask = ~(
                            filtered_data['Product Name'].str.contains('新一代', na=False) & 
                            (filtered_data['Product Name'].str.contains('52', na=False) | 
                             filtered_data['Product Name'].str.contains('66', na=False))
                        )
                    else:
                        continue
                    
                    product_mask = product_mask | category_mask
                
                filtered_data = filtered_data[product_mask]
            
            # 周末锁单筛选
            if weekend_lock_filter != "全部" and 'Lock_Time' in filtered_data.columns:
                # 将Lock_Time转换为datetime格式
                filtered_data['Lock_Time_dt'] = pd.to_datetime(filtered_data['Lock_Time'], errors='coerce')
                
                if weekend_lock_filter == "仅周末锁单":
                    # 筛选周末锁单（周六=5, 周日=6）
                    weekend_mask = filtered_data['Lock_Time_dt'].dt.weekday.isin([5, 6])
                    filtered_data = filtered_data[weekend_mask]
                elif weekend_lock_filter == "仅工作日锁单":
                    # 筛选工作日锁单（周一到周五=0-4）
                    weekday_mask = filtered_data['Lock_Time_dt'].dt.weekday.isin([0, 1, 2, 3, 4])
                    filtered_data = filtered_data[weekday_mask]
                
                # 删除临时列
                filtered_data = filtered_data.drop('Lock_Time_dt', axis=1)
            
            # 最终筛选：只保留有锁单数据的记录用于统计
            if 'Lock_Time' in filtered_data.columns:
                lock_data = filtered_data[filtered_data['Lock_Time'].notna()].copy()
            else:
                return pd.DataFrame()
            
            if lock_data.empty:
                return pd.DataFrame()
            
            # 按年龄段分组统计各车型的锁单数，调整为车型对比格式
            if 'buyer_age' not in lock_data.columns:
                return pd.DataFrame()
            
            # 创建年龄段分类函数
            def categorize_age(age):
                if pd.isna(age):
                    return '未知年龄'
                age = int(age)
                if age < 25:
                    return '25岁以下'
                elif age < 30:
                    return '25-29岁'
                elif age < 35:
                    return '30-34岁'
                elif age < 40:
                    return '35-39岁'
                elif age < 45:
                    return '40-44岁'
                elif age < 50:
                    return '45-49岁'
                elif age < 55:
                    return '50-54岁'
                else:
                    return '55岁以上'
            
            # 为所有数据添加年龄段
            lock_data['年龄段'] = lock_data['buyer_age'].apply(categorize_age)
            
            # 根据include_unknown参数过滤未知年龄数据
            if not include_unknown:
                lock_data = lock_data[lock_data['年龄段'] != '未知年龄']
            
            # 获取所有年龄段
            all_age_groups = lock_data['年龄段'].dropna().unique()
            
            # 构建车型对比表格数据
            result_data = []
            
            for age_group in all_age_groups:
                age_data = lock_data[lock_data['年龄段'] == age_group]
                
                row_data = {'年龄段': age_group}
                
                # 为每个车型统计锁单数和占比
                for vehicle in selected_vehicles:
                    vehicle_age_data = age_data[age_data['车型分组'] == vehicle]
                    vehicle_total_data = lock_data[lock_data['车型分组'] == vehicle]
                    
                    lock_count = len(vehicle_age_data)
                    total_count = len(vehicle_total_data)
                    lock_ratio = (lock_count / total_count * 100) if total_count > 0 else 0
                    
                    row_data[f'{vehicle}锁单数'] = lock_count
                    row_data[f'{vehicle}锁单占比'] = round(lock_ratio, 2)
                
                result_data.append(row_data)
            
            if result_data:
                final_df = pd.DataFrame(result_data)
                # 按第一个车型的锁单数排序
                if selected_vehicles:
                    sort_column = f'{selected_vehicles[0]}锁单数'
                    final_df = final_df.sort_values(sort_column, ascending=False)
                return final_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"准备买家年龄锁单数据时出错: {e}")
            return pd.DataFrame()

    def create_age_lock_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建买家年龄锁单统计表格（车型对比格式）"""
        if data.empty:
            return pd.DataFrame({'提示': ['暂无锁单数据']})
        
        try:
            # 重新组织表格格式，实现车型对比和高亮功能
            table_data = []
            
            for _, row in data.iterrows():
                age_group = row['年龄段']
                
                # 构建基础行数据
                table_row = {'年龄段': age_group}
                
                # 获取所有车型相关的列
                vehicle_columns = [col for col in data.columns if col.endswith('锁单数') or col.endswith('锁单占比')]
                
                # 分别处理锁单数和锁单占比的高亮
                lock_count_columns = [col for col in vehicle_columns if col.endswith('锁单数')]
                lock_ratio_columns = [col for col in vehicle_columns if col.endswith('锁单占比')]
                
                # 找出锁单数最大值
                if lock_count_columns:
                    lock_count_values = [row[col] for col in lock_count_columns]
                    max_lock_count = max(lock_count_values) if lock_count_values else 0
                
                # 找出锁单占比最大值
                if lock_ratio_columns:
                    lock_ratio_values = [row[col] for col in lock_ratio_columns]
                    max_lock_ratio = max(lock_ratio_values) if lock_ratio_values else 0
                
                # 添加车型数据，对最大值进行高亮
                for col in vehicle_columns:
                    value = row[col]
                    if col.endswith('锁单数'):
                        formatted_value = f"{int(value):,}"
                        # 如果是最大值且大于0，添加红色高亮
                        if value == max_lock_count and value > 0 and len(lock_count_columns) > 1:
                            formatted_value = f'<span style="color: red; font-weight: bold;">{formatted_value}</span>'
                        table_row[col] = formatted_value
                    elif col.endswith('锁单占比'):
                        formatted_value = f"{value}%"
                        # 如果是最大值且大于0，添加红色高亮
                        if value == max_lock_ratio and value > 0 and len(lock_ratio_columns) > 1:
                            formatted_value = f'<span style="color: red; font-weight: bold;">{formatted_value}</span>'
                        table_row[col] = formatted_value
                
                table_data.append(table_row)
            
            return pd.DataFrame(table_data)
            
        except Exception as e:
            logger.error(f"创建买家年龄锁单表格时出错: {e}")
            return pd.DataFrame({'错误': [f'表格生成失败: {str(e)}']})

    def prepare_gender_lock_data(self, selected_vehicles: List[str], start_date: str = '', end_date: str = '', 
                               product_types: List[str] = None, lock_start_date: str = '', lock_end_date: str = '', lock_n_days: int = 30, include_unknown: bool = True, weekend_lock_filter: str = "全部") -> pd.DataFrame:
        """准备订单性别锁单统计数据（支持多种筛选条件）"""
        try:
            if self.df.empty:
                return pd.DataFrame()
            
            # 基础筛选：车型分组
            filtered_data = self.df[self.df['车型分组'].isin(selected_vehicles)].copy()
            
            if filtered_data.empty:
                return pd.DataFrame()
            
            # 小订时间范围筛选（基于Intention_Payment_Time）
            if start_date and 'Intention_Payment_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Intention_Payment_Time'] >= start_date]
            if end_date and 'Intention_Payment_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Intention_Payment_Time'] <= end_date]
            
            # 锁单时间范围筛选（基于Lock_Time）
            if lock_start_date and 'Lock_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Lock_Time'] >= lock_start_date]
            if lock_end_date and 'Lock_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Lock_Time'] <= lock_end_date]
            
            # 基于锁单后N天的筛选（基于business_definition.json最大值+N天）
            if lock_n_days and 'Lock_Time' in filtered_data.columns and hasattr(self, 'business_def'):
                for vehicle in selected_vehicles:
                    if vehicle in self.business_def:
                        # 计算该车型的预售结束日期 + N天
                        end_date_str = self.business_def[vehicle]['end']
                        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                        max_lock_date = end_date + timedelta(days=lock_n_days)
                        
                        # 筛选该车型在指定时间范围内的锁单数据
                        vehicle_mask = (filtered_data['车型分组'] == vehicle) & (filtered_data['Lock_Time'] <= max_lock_date.strftime('%Y-%m-%d'))
                        if vehicle == selected_vehicles[0]:  # 第一个车型，直接赋值
                            time_filtered_data = filtered_data[vehicle_mask]
                        else:  # 后续车型，合并数据
                            time_filtered_data = pd.concat([time_filtered_data, filtered_data[vehicle_mask]], ignore_index=True)
                
                if 'time_filtered_data' in locals():
                    filtered_data = time_filtered_data
            
            # 产品分类筛选
            if product_types and 'Product Name' in filtered_data.columns:
                product_mask = pd.Series([False] * len(filtered_data), index=filtered_data.index)
                
                for category in product_types:
                    if category == "增程":
                        # 产品名称中包含"新一代"和数字52或66的为增程
                        category_mask = (
                            filtered_data['Product Name'].str.contains('新一代', na=False) & 
                            (filtered_data['Product Name'].str.contains('52', na=False) | 
                             filtered_data['Product Name'].str.contains('66', na=False))
                        )
                    elif category == "纯电":
                        # 其他产品为纯电
                        category_mask = ~(
                            filtered_data['Product Name'].str.contains('新一代', na=False) & 
                            (filtered_data['Product Name'].str.contains('52', na=False) | 
                             filtered_data['Product Name'].str.contains('66', na=False))
                        )
                    else:
                        continue
                    
                    product_mask = product_mask | category_mask
                
                filtered_data = filtered_data[product_mask]
            
            # 周末锁单筛选
            if weekend_lock_filter != "全部" and 'Lock_Time' in filtered_data.columns:
                # 将Lock_Time转换为datetime格式
                filtered_data['Lock_Time_dt'] = pd.to_datetime(filtered_data['Lock_Time'], errors='coerce')
                
                if weekend_lock_filter == "仅周末锁单":
                    # 筛选周末锁单（周六=5, 周日=6）
                    weekend_mask = filtered_data['Lock_Time_dt'].dt.weekday.isin([5, 6])
                    filtered_data = filtered_data[weekend_mask]
                elif weekend_lock_filter == "仅工作日锁单":
                    # 筛选工作日锁单（周一到周五=0-4）
                    weekday_mask = filtered_data['Lock_Time_dt'].dt.weekday.isin([0, 1, 2, 3, 4])
                    filtered_data = filtered_data[weekday_mask]
                
                # 删除临时列
                filtered_data = filtered_data.drop('Lock_Time_dt', axis=1)
            
            # 最终筛选：只保留有锁单数据的记录用于统计
            if 'Lock_Time' in filtered_data.columns:
                lock_data = filtered_data[filtered_data['Lock_Time'].notna()].copy()
            else:
                return pd.DataFrame()
            
            if lock_data.empty:
                return pd.DataFrame()
            
            # 处理性别数据，统一格式
            def normalize_gender(gender):
                if pd.isna(gender):
                    return '未知性别'
                gender_str = str(gender).strip().lower()
                if gender_str in ['男', 'male', 'm', '1']:
                    return '男'
                elif gender_str in ['女', 'female', 'f', '0']:
                    return '女'
                else:
                    return '未知性别'
            
            # 为所有数据添加性别分类
            if 'order_gender' in lock_data.columns:
                lock_data['性别'] = lock_data['order_gender'].apply(normalize_gender)
            else:
                return pd.DataFrame()
            
            # 根据include_unknown参数过滤未知性别数据
            if not include_unknown:
                lock_data = lock_data[lock_data['性别'] != '未知性别']
            
            # 获取所有性别类别
            all_genders = sorted(lock_data['性别'].unique())
            
            # 按性别分组统计各车型锁单数
            result_data = []
            
            # 先计算每个车型的总锁单数（用于计算占比）
            vehicle_totals = {}
            for vehicle in selected_vehicles:
                vehicle_totals[vehicle] = len(lock_data[lock_data['车型分组'] == vehicle])
            
            for gender in all_genders:
                gender_data = lock_data[lock_data['性别'] == gender]
                
                # 构建该性别的车型对比数据
                row_data = {'性别': gender}
                
                # 为每个车型添加锁单数和占比
                for vehicle in selected_vehicles:
                    vehicle_locks = len(gender_data[gender_data['车型分组'] == vehicle])
                    # 计算该性别在该车型中的占比
                    vehicle_ratio = (vehicle_locks / vehicle_totals[vehicle] * 100) if vehicle_totals[vehicle] > 0 else 0
                    
                    row_data[f'{vehicle}_锁单数'] = vehicle_locks
                    row_data[f'{vehicle}_锁单占比(%)'] = round(vehicle_ratio, 2)
                
                result_data.append(row_data)
            
            if result_data:
                final_df = pd.DataFrame(result_data)
                # 按第一个车型的锁单数排序
                if selected_vehicles:
                    sort_column = f'{selected_vehicles[0]}_锁单数'
                    final_df = final_df.sort_values(sort_column, ascending=False)
                return final_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"准备订单性别锁单数据时出错: {e}")
            return pd.DataFrame()

    def create_gender_lock_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建订单性别锁单统计表格（车型对比格式）"""
        if data.empty:
            return pd.DataFrame({'提示': ['暂无锁单数据']})
        
        try:
            # 获取车型相关的列
            vehicle_columns = [col for col in data.columns if col.endswith('_锁单数') or col.endswith('_锁单占比(%)')]
            lock_columns = [col for col in vehicle_columns if col.endswith('_锁单数')]
            ratio_columns = [col for col in vehicle_columns if col.endswith('_锁单占比(%)')]
            
            # 计算锁单数和占比的最大值（用于高亮）
            lock_max_values = {}
            ratio_max_values = {}
            
            for _, row in data.iterrows():
                # 找出该行锁单数的最大值
                lock_values = [row[col] for col in lock_columns]
                if lock_values:
                    max_lock = max(lock_values)
                    lock_max_values[row.name] = max_lock
                
                # 找出该行占比的最大值
                ratio_values = [row[col] for col in ratio_columns]
                if ratio_values:
                    max_ratio = max(ratio_values)
                    ratio_max_values[row.name] = max_ratio
            
            # 重新组织表格格式
            table_data = []
            
            for _, row in data.iterrows():
                table_row = {'性别': row['性别']}
                
                # 添加各车型的锁单数和占比，并应用高亮
                for col in data.columns:
                    if col == '性别':
                        continue
                    
                    value = row[col]
                    
                    if col.endswith('_锁单数'):
                        # 锁单数格式化并高亮
                        formatted_value = f"{int(value):,}"
                        if value == lock_max_values.get(row.name, 0) and value > 0:
                            formatted_value = f'<span style="color: red; font-weight: bold;">{formatted_value}</span>'
                        table_row[col] = formatted_value
                    
                    elif col.endswith('_锁单占比(%)'):
                        # 占比格式化并高亮
                        formatted_value = f"{value}%"
                        if value == ratio_max_values.get(row.name, 0) and value > 0:
                            formatted_value = f'<span style="color: red; font-weight: bold;">{formatted_value}</span>'
                        table_row[col] = formatted_value
                
                table_data.append(table_row)
            
            return pd.DataFrame(table_data)
            
        except Exception as e:
            logger.error(f"创建订单性别锁单表格时出错: {e}")
            return pd.DataFrame({'错误': [f'表格生成失败: {str(e)}']})

    def prepare_region_lock_data(self, selected_vehicles: List[str], start_date: str = '', end_date: str = '', 
                               product_types: List[str] = None, lock_start_date: str = '', lock_end_date: str = '', lock_n_days: int = 30, include_unknown: bool = True, include_virtual: bool = True, include_fac: bool = True, weekend_lock_filter: str = "全部") -> pd.DataFrame:
        """准备父级区域锁单统计数据（支持多种筛选条件）"""
        try:
            if self.df.empty:
                return pd.DataFrame()
            
            # 基础筛选：车型分组
            filtered_data = self.df[self.df['车型分组'].isin(selected_vehicles)].copy()
            
            if filtered_data.empty:
                return pd.DataFrame()
            
            # 小订时间范围筛选（基于Intention_Payment_Time）
            if start_date and 'Intention_Payment_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Intention_Payment_Time'] >= start_date]
            if end_date and 'Intention_Payment_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Intention_Payment_Time'] <= end_date]
            
            # 锁单时间范围筛选（基于Lock_Time）
            if lock_start_date and 'Lock_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Lock_Time'] >= lock_start_date]
            if lock_end_date and 'Lock_Time' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Lock_Time'] <= lock_end_date]
            
            # 基于锁单后N天的筛选（基于business_definition.json最大值+N天）
            if lock_n_days and 'Lock_Time' in filtered_data.columns and hasattr(self, 'business_def'):
                for vehicle in selected_vehicles:
                    if vehicle in self.business_def:
                        # 计算该车型的预售结束日期 + N天
                        end_date_str = self.business_def[vehicle]['end']
                        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                        max_lock_date = end_date + timedelta(days=lock_n_days)
                        
                        # 筛选该车型在指定时间范围内的锁单数据
                        vehicle_mask = (filtered_data['车型分组'] == vehicle) & (filtered_data['Lock_Time'] <= max_lock_date.strftime('%Y-%m-%d'))
                        if vehicle == selected_vehicles[0]:  # 第一个车型，直接赋值
                            time_filtered_data = filtered_data[vehicle_mask]
                        else:  # 后续车型，合并数据
                            time_filtered_data = pd.concat([time_filtered_data, filtered_data[vehicle_mask]], ignore_index=True)
                
                if 'time_filtered_data' in locals():
                    filtered_data = time_filtered_data
            
            # 产品分类筛选
            if product_types and 'Product Name' in filtered_data.columns:
                product_mask = pd.Series([False] * len(filtered_data), index=filtered_data.index)
                
                for category in product_types:
                    if category == "增程":
                        # 产品名称中包含"新一代"和数字52或66的为增程
                        category_mask = (
                            filtered_data['Product Name'].str.contains('新一代', na=False) & 
                            (filtered_data['Product Name'].str.contains('52', na=False) | 
                             filtered_data['Product Name'].str.contains('66', na=False))
                        )
                    elif category == "纯电":
                        # 其他产品为纯电
                        category_mask = ~(
                            filtered_data['Product Name'].str.contains('新一代', na=False) & 
                            (filtered_data['Product Name'].str.contains('52', na=False) | 
                             filtered_data['Product Name'].str.contains('66', na=False))
                        )
                    else:
                        continue
                    
                    product_mask = product_mask | category_mask
                
                filtered_data = filtered_data[product_mask]
            
            # 周末锁单筛选
            if weekend_lock_filter != "全部" and 'Lock_Time' in filtered_data.columns:
                # 将Lock_Time转换为datetime格式
                filtered_data['Lock_Time_dt'] = pd.to_datetime(filtered_data['Lock_Time'], errors='coerce')
                
                if weekend_lock_filter == "仅周末锁单":
                    # 筛选周末锁单（周六=5, 周日=6）
                    weekend_mask = filtered_data['Lock_Time_dt'].dt.weekday.isin([5, 6])
                    filtered_data = filtered_data[weekend_mask]
                elif weekend_lock_filter == "仅工作日锁单":
                    # 筛选工作日锁单（周一到周五=0-4）
                    weekday_mask = filtered_data['Lock_Time_dt'].dt.weekday.isin([0, 1, 2, 3, 4])
                    filtered_data = filtered_data[weekday_mask]
                
                # 删除临时列
                filtered_data = filtered_data.drop('Lock_Time_dt', axis=1)
            
            # 最终筛选：只保留有锁单数据的记录用于统计
            if 'Lock_Time' in filtered_data.columns:
                lock_data = filtered_data[filtered_data['Lock_Time'].notna()].copy()
            else:
                return pd.DataFrame()
            
            if lock_data.empty:
                return pd.DataFrame()
            
            # 处理区域数据，统一格式
            def normalize_region(region):
                if pd.isna(region):
                    return '未知区域'
                region_str = str(region).strip()
                if region_str == '' or region_str.lower() == 'nan':
                    return '未知区域'
                return region_str
            
            # 定义虚拟大区和FAC大区的分类逻辑
            def classify_region_type(region):
                """根据区域名称分类为虚拟大区或FAC大区"""
                if pd.isna(region) or region == '未知区域':
                    return '未知'
                region_str = str(region).strip()
                
                # 虚拟大区的关键词（可根据实际业务需求调整）
                virtual_keywords = ['虚拟', 'Virtual', '线上', '网络', '电商', '数字']
                # FAC大区的关键词（可根据实际业务需求调整）
                fac_keywords = ['FAC', 'fac', '工厂', 'Factory', '直营']
                
                # 检查是否包含虚拟大区关键词
                for keyword in virtual_keywords:
                    if keyword in region_str:
                        return '虚拟大区'
                
                # 检查是否包含FAC大区关键词
                for keyword in fac_keywords:
                    if keyword in region_str:
                        return 'FAC大区'
                
                # 默认为传统大区
                return '传统大区'

            # 为所有数据添加区域分类
            if 'Parent Region Name' in lock_data.columns:
                lock_data['父级区域'] = lock_data['Parent Region Name'].apply(normalize_region)
                lock_data['区域类型'] = lock_data['Parent Region Name'].apply(classify_region_type)
            else:
                return pd.DataFrame()

            # 根据include_unknown参数过滤未知区域数据
            if not include_unknown:
                lock_data = lock_data[lock_data['父级区域'] != '未知区域']
            
            # 根据虚拟大区和FAC大区过滤参数过滤数据
            region_type_filters = []
            if include_virtual:
                region_type_filters.extend(['虚拟大区', '传统大区'])
            if include_fac:
                region_type_filters.append('FAC大区')
            if not include_virtual and not include_fac:
                # 如果两个都不选择，则只显示传统大区
                region_type_filters = ['传统大区']
            
            # 添加未知类型（如果include_unknown为True）
            if include_unknown:
                region_type_filters.append('未知')
            
            lock_data = lock_data[lock_data['区域类型'].isin(region_type_filters)]
            
            # 获取所有区域类别
            all_regions = sorted(lock_data['父级区域'].unique())
            
            # 按区域分组统计各车型锁单数
            result_data = []
            
            # 先计算每个车型的总锁单数（用于计算占比）
            vehicle_totals = {}
            for vehicle in selected_vehicles:
                vehicle_totals[vehicle] = len(lock_data[lock_data['车型分组'] == vehicle])
            
            for region in all_regions:
                region_data = lock_data[lock_data['父级区域'] == region]
                
                # 构建该区域的车型对比数据
                row_data = {'父级区域': region}
                
                # 为每个车型添加锁单数和占比
                for vehicle in selected_vehicles:
                    vehicle_locks = len(region_data[region_data['车型分组'] == vehicle])
                    # 计算该区域在该车型中的占比
                    vehicle_ratio = (vehicle_locks / vehicle_totals[vehicle] * 100) if vehicle_totals[vehicle] > 0 else 0
                    
                    row_data[f'{vehicle}_锁单数'] = vehicle_locks
                    row_data[f'{vehicle}_锁单占比(%)'] = round(vehicle_ratio, 2)
                
                result_data.append(row_data)
            
            if result_data:
                final_df = pd.DataFrame(result_data)
                # 按第一个车型的锁单数排序
                if selected_vehicles:
                    sort_column = f'{selected_vehicles[0]}_锁单数'
                    final_df = final_df.sort_values(sort_column, ascending=False)
                return final_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"准备父级区域锁单数据时出错: {e}")
            return pd.DataFrame()

    def create_region_lock_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建父级区域锁单统计表格（车型对比格式）"""
        if data.empty:
            return pd.DataFrame({'提示': ['暂无锁单数据']})
        
        try:
            # 获取车型相关的列
            vehicle_columns = [col for col in data.columns if col.endswith('_锁单数') or col.endswith('_锁单占比(%)')]
            lock_columns = [col for col in vehicle_columns if col.endswith('_锁单数')]
            ratio_columns = [col for col in vehicle_columns if col.endswith('_锁单占比(%)')]
            
            # 计算锁单数和占比的最大值（用于高亮）
            lock_max_values = {}
            ratio_max_values = {}
            
            for _, row in data.iterrows():
                # 找出该行锁单数的最大值
                lock_values = [row[col] for col in lock_columns]
                if lock_values:
                    max_lock = max(lock_values)
                    lock_max_values[row.name] = max_lock
                
                # 找出该行占比的最大值
                ratio_values = [row[col] for col in ratio_columns]
                if ratio_values:
                    max_ratio = max(ratio_values)
                    ratio_max_values[row.name] = max_ratio
            
            # 重新组织表格格式
            table_data = []
            
            for _, row in data.iterrows():
                table_row = {'父级区域': row['父级区域']}
                
                # 添加各车型的锁单数和占比，并应用高亮
                for col in data.columns:
                    if col == '父级区域':
                        continue
                    
                    value = row[col]
                    
                    if col.endswith('_锁单数'):
                        # 锁单数格式化并高亮
                        formatted_value = f"{int(value):,}"
                        if value == lock_max_values.get(row.name, 0) and value > 0:
                            formatted_value = f'<span style="color: red; font-weight: bold;">{formatted_value}</span>'
                        table_row[col] = formatted_value
                    
                    elif col.endswith('_锁单占比(%)'):
                        # 占比格式化并高亮
                        formatted_value = f"{value}%"
                        if value == ratio_max_values.get(row.name, 0) and value > 0:
                            formatted_value = f'<span style="color: red; font-weight: bold;">{formatted_value}</span>'
                        table_row[col] = formatted_value
                
                table_data.append(table_row)
            
            return pd.DataFrame(table_data)
            
        except Exception as e:
            logger.error(f"创建父级区域锁单表格时出错: {e}")
            return pd.DataFrame({'错误': [f'表格生成失败: {str(e)}']})
    
    def prepare_city_lock_data(self, selected_vehicles: List[str], order_start_date: str, order_end_date: str,
                              lock_start_date: str, lock_end_date: str, lock_n_days: int,
                              product_types: List[str], weekend_lock_filter: str = "全部", 
                              min_lock_count: int = 100, max_lock_count: int = 1000) -> pd.DataFrame:
        """准备License City锁单数据（车型对比格式）"""
        try:
            if self.df.empty:
                return pd.DataFrame()
            
            # 筛选车型数据
            lock_data = self.df[self.df['车型分组'].isin(selected_vehicles)].copy()
            
            if lock_data.empty:
                return pd.DataFrame()
            
            # 小订时间范围筛选
            if order_start_date and order_end_date and 'Intention_Payment_Time' in lock_data.columns:
                order_start_dt = datetime.strptime(order_start_date, '%Y-%m-%d').date()
                order_end_dt = datetime.strptime(order_end_date, '%Y-%m-%d').date()
                
                lock_data = lock_data[
                    (pd.to_datetime(lock_data['Intention_Payment_Time']).dt.date >= order_start_dt) &
                    (pd.to_datetime(lock_data['Intention_Payment_Time']).dt.date <= order_end_dt)
                ]
            elif order_start_date and 'Intention_Payment_Time' in lock_data.columns:
                order_start_dt = datetime.strptime(order_start_date, '%Y-%m-%d').date()
                lock_data = lock_data[pd.to_datetime(lock_data['Intention_Payment_Time']).dt.date >= order_start_dt]
            elif order_end_date and 'Intention_Payment_Time' in lock_data.columns:
                order_end_dt = datetime.strptime(order_end_date, '%Y-%m-%d').date()
                lock_data = lock_data[pd.to_datetime(lock_data['Intention_Payment_Time']).dt.date <= order_end_dt]
            
            # 锁单时间范围筛选
            if lock_start_date and lock_end_date and 'Lock_Time' in lock_data.columns:
                lock_start_dt = datetime.strptime(lock_start_date, '%Y-%m-%d').date()
                lock_end_dt = datetime.strptime(lock_end_date, '%Y-%m-%d').date()
                
                lock_data = lock_data[
                    (pd.to_datetime(lock_data['Lock_Time']).dt.date >= lock_start_dt) &
                    (pd.to_datetime(lock_data['Lock_Time']).dt.date <= lock_end_dt)
                ]
            elif lock_start_date and 'Lock_Time' in lock_data.columns:
                lock_start_dt = datetime.strptime(lock_start_date, '%Y-%m-%d').date()
                lock_data = lock_data[pd.to_datetime(lock_data['Lock_Time']).dt.date >= lock_start_dt]
            elif lock_end_date and 'Lock_Time' in lock_data.columns:
                lock_end_dt = datetime.strptime(lock_end_date, '%Y-%m-%d').date()
                lock_data = lock_data[pd.to_datetime(lock_data['Lock_Time']).dt.date <= lock_end_dt]
            
            # 锁单后N天数筛选
            if lock_n_days and lock_n_days > 0 and 'Lock_Time' in lock_data.columns and hasattr(self, 'business_def'):
                # 基于business_definition.json计算各车型的最大预售天数
                for vehicle in selected_vehicles:
                    if vehicle in self.business_def:
                        max_days = self.business_def[vehicle].get('max_days', 30)
                        cutoff_day = max_days + lock_n_days
                        
                        # 筛选该车型在cutoff_day之前锁单的数据
                        vehicle_data = lock_data[lock_data['车型分组'] == vehicle]
                        if not vehicle_data.empty:
                            # 计算每行的天数
                            days_from_start = vehicle_data['Lock_Time'].apply(
                                lambda x: self.calculate_days_from_start(vehicle, pd.to_datetime(x))
                            )
                            vehicle_data = vehicle_data[days_from_start <= cutoff_day]
                            # 更新lock_data，保留其他车型数据
                            lock_data = pd.concat([
                                lock_data[lock_data['车型分组'] != vehicle],
                                vehicle_data
                            ])
            
            # 产品分类筛选
            if product_types and 'Product Name' in lock_data.columns and hasattr(self, 'business_def'):
                filtered_products = []
                for vehicle in selected_vehicles:
                    if vehicle in self.business_def and 'products' in self.business_def[vehicle]:
                        for product, info in self.business_def[vehicle]['products'].items():
                            # 检查产品类型是否在筛选列表中
                            product_category = info.get('category', '')
                            if any(ptype in product_category for ptype in product_types):
                                filtered_products.append(product)
                
                if filtered_products:
                    lock_data = lock_data[lock_data['Product Name'].isin(filtered_products)]
            
            # 如果没有指定产品分类，默认只保留小订产品
            elif 'Product Name' in lock_data.columns and hasattr(self, 'business_def'):
                small_order_products = []
                for vehicle in selected_vehicles:
                    if vehicle in self.business_def and 'products' in self.business_def[vehicle]:
                        small_order_products.extend([
                            product for product, info in self.business_def[vehicle]['products'].items()
                            if info.get('type') == '小订'
                        ])
                
                if small_order_products:
                    lock_data = lock_data[lock_data['Product Name'].isin(small_order_products)]
            
            # 周末锁单筛选
            if weekend_lock_filter != "全部":
                lock_data['Lock_Time_dt'] = pd.to_datetime(lock_data['Lock_Time'])
                lock_data['is_weekend'] = lock_data['Lock_Time_dt'].dt.dayofweek >= 5
                
                if weekend_lock_filter == "仅周末":
                    lock_data = lock_data[lock_data['is_weekend']]
                elif weekend_lock_filter == "仅工作日":
                    lock_data = lock_data[~lock_data['is_weekend']]
            
            # 只保留有锁单时间的数据
            lock_data = lock_data[lock_data['Lock_Time'].notna()]
            
            # 获取所有License City
            all_cities = sorted(lock_data['License City'].dropna().unique())
            
            # 按License City分组统计各车型锁单数
            result_data = []
            
            # 先计算每个车型的总锁单数（用于计算占比）
            vehicle_totals = {}
            for vehicle in selected_vehicles:
                vehicle_totals[vehicle] = len(lock_data[lock_data['车型分组'] == vehicle])
            
            for city in all_cities:
                city_data = lock_data[lock_data['License City'] == city]
                
                # 构建该城市的车型对比数据
                row_data = {'License City': city}
                
                # 为每个车型添加锁单数和占比
                vehicle_locks_list = []
                for vehicle in selected_vehicles:
                    vehicle_locks = len(city_data[city_data['车型分组'] == vehicle])
                    vehicle_locks_list.append(vehicle_locks)
                    
                    # 计算该城市在该车型中的占比
                    vehicle_ratio = (vehicle_locks / vehicle_totals[vehicle] * 100) if vehicle_totals[vehicle] > 0 else 0
                    
                    row_data[f'{vehicle}_锁单数'] = vehicle_locks
                    row_data[f'{vehicle}_锁单占比(%)'] = round(vehicle_ratio, 2)
                
                # 应用锁单数筛选：只要有任意一个车型的锁单数在范围内就保留该城市
                if any(min_lock_count <= vehicle_locks <= max_lock_count for vehicle_locks in vehicle_locks_list):
                    result_data.append(row_data)
            
            if result_data:
                final_df = pd.DataFrame(result_data)
                # 按第一个车型的锁单数排序
                if selected_vehicles:
                    sort_column = f'{selected_vehicles[0]}_锁单数'
                    final_df = final_df.sort_values(sort_column, ascending=False)
                return final_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"准备License City锁单数据时出错: {e}")
            return pd.DataFrame()

    def create_city_lock_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建License City锁单统计表格（车型对比格式）"""
        if data.empty:
            return pd.DataFrame({'提示': ['暂无锁单数据']})
        
        try:
            # 获取车型相关的列
            vehicle_columns = [col for col in data.columns if col.endswith('_锁单数') or col.endswith('_锁单占比(%)')]
            lock_columns = [col for col in vehicle_columns if col.endswith('_锁单数')]
            ratio_columns = [col for col in vehicle_columns if col.endswith('_锁单占比(%)')]
            
            # 计算锁单数和占比的最大值（用于高亮）
            lock_max_values = {}
            ratio_max_values = {}
            
            for _, row in data.iterrows():
                # 找出该行锁单数的最大值
                lock_values = [row[col] for col in lock_columns]
                if lock_values:
                    max_lock = max(lock_values)
                    lock_max_values[row.name] = max_lock
                
                # 找出该行占比的最大值
                ratio_values = [row[col] for col in ratio_columns]
                if ratio_values:
                    max_ratio = max(ratio_values)
                    ratio_max_values[row.name] = max_ratio
            
            # 重新组织表格格式
            table_data = []
            
            for _, row in data.iterrows():
                table_row = {'License City': row['License City']}
                
                # 添加各车型的锁单数和占比，并应用高亮
                for col in data.columns:
                    if col == 'License City':
                        continue
                    
                    value = row[col]
                    
                    if col.endswith('_锁单数'):
                        # 锁单数格式化并高亮
                        formatted_value = f"{int(value):,}"
                        if value == lock_max_values.get(row.name, 0) and value > 0:
                            formatted_value = f'<span style="color: red; font-weight: bold;">{formatted_value}</span>'
                        table_row[col] = formatted_value
                    
                    elif col.endswith('_锁单占比(%)'):
                        # 占比格式化并高亮
                        formatted_value = f"{value}%"
                        if value == ratio_max_values.get(row.name, 0) and value > 0:
                            formatted_value = f'<span style="color: red; font-weight: bold;">{formatted_value}</span>'
                        table_row[col] = formatted_value
                
                table_data.append(table_row)
            
            return pd.DataFrame(table_data)
            
        except Exception as e:
            logger.error(f"创建License City锁单表格时出错: {e}")
            return pd.DataFrame({'错误': [f'表格生成失败: {str(e)}']})
    
    def prepare_lock_performance_table_data(self, selected_vehicles: List[str], n_days: int = 30) -> pd.DataFrame:
        """准备锁单表现表格数据"""
        try:
            if self.df.empty:
                return pd.DataFrame()
            
            # 获取车型的最大日期（预售结束日期）
            vehicle_max_dates = {}
            for vehicle in selected_vehicles:
                if vehicle in self.business_def:
                    vehicle_max_dates[vehicle] = datetime.strptime(
                        self.business_def[vehicle]['end'], '%Y-%m-%d'
                    ).date()
            
            result_data = []
            
            # 为每一天（0到N日）计算各种锁单数
            for day in range(n_days + 1):
                day_data = {'第N日': day}
                
                for vehicle in selected_vehicles:
                    if vehicle not in vehicle_max_dates:
                        continue
                        
                    vehicle_data = self.df[self.df['车型分组'] == vehicle].copy()
                    if vehicle_data.empty:
                        continue
                    
                    max_date = vehicle_max_dates[vehicle]
                    target_date = max_date + timedelta(days=day)
                    
                    # 筛选该日期的锁单数据
                    day_lock_data = vehicle_data[
                        (vehicle_data['Lock_Time'].notna()) &
                        (pd.to_datetime(vehicle_data['Lock_Time']).dt.date == target_date)
                    ]
                    
                    # 计算各种锁单数
                    daily_locks = len(day_lock_data)
                    
                    # 获取该车型的预售结束时间
                    vehicle_end_date = datetime.strptime(
                        self.business_def[vehicle]['end'], '%Y-%m-%d'
                    ).date()
                    
                    # 小订留存锁单数：Lock_Time和Intention_Payment_Time都非空，且Intention_Payment_Time < vehicle_end_date
                    retained_locks = len(day_lock_data[
                        (day_lock_data['Intention_Payment_Time'].notna()) &
                        (pd.to_datetime(day_lock_data['Intention_Payment_Time']).dt.date <= vehicle_end_date)
                    ])
                    
                    # 发布会后小订锁单数：Lock_Time和Intention_Payment_Time都非空，且Intention_Payment_Time >= vehicle_end_date
                    post_launch_locks = len(day_lock_data[
                        (day_lock_data['Intention_Payment_Time'].notna()) &
                        (pd.to_datetime(day_lock_data['Intention_Payment_Time']).dt.date > vehicle_end_date)
                    ])
                    
                    # 直接锁单数：含有Lock_Time但没有Intention_Payment_Time的订单数
                    direct_locks = len(day_lock_data[
                        day_lock_data['Intention_Payment_Time'].isna()
                    ])
                    
                    # 验证数据一致性：三个分类的合计应该等于当日锁单总数
                    total_classified = retained_locks + post_launch_locks + direct_locks
                    if total_classified != daily_locks:
                        logger.warning(f"第{day}日 {vehicle} 锁单分类不一致: 总数{daily_locks}, 分类合计{total_classified}")
                        logger.warning(f"  小订留存: {retained_locks}, 发布会后: {post_launch_locks}, 直接: {direct_locks}")
                    
                    # 累计锁单数（从第0日到当前日）
                    cumulative_data = vehicle_data[
                        (vehicle_data['Lock_Time'].notna()) &
                        (pd.to_datetime(vehicle_data['Lock_Time']).dt.date >= max_date) &
                        (pd.to_datetime(vehicle_data['Lock_Time']).dt.date <= target_date)
                    ]
                    cumulative_locks = len(cumulative_data)
                    
                    # 按新的表头顺序组织数据
                    day_data[f'{vehicle}锁单数'] = daily_locks
                    day_data[f'{vehicle}累计锁单数'] = cumulative_locks
                    day_data[f'{vehicle}锁单结构'] = f"{retained_locks}/{post_launch_locks}/{direct_locks}"
                
                result_data.append(day_data)
            
            if result_data:
                df = pd.DataFrame(result_data)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"准备锁单表现表格数据时出错: {e}")
            return pd.DataFrame()
    
    def create_lock_performance_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建锁单表现表格"""
        if data.empty:
            return pd.DataFrame({'提示': ['暂无锁单数据']})
        
        try:
            # 按第N日排序
            data_sorted = data.sort_values('第N日')
            
            # 获取所有车型
            vehicles = []
            for col in data.columns:
                if '锁单数' in col and '累计' not in col:
                    vehicle = col.replace('锁单数', '')
                    if vehicle not in vehicles:
                        vehicles.append(vehicle)
            
            # 按照指定顺序重新组织表格结构
            table_data = []
            
            for _, row in data_sorted.iterrows():
                table_row = {'第N日': f"第{row['第N日']}日"}
                
                # 按照用户要求的列顺序添加数据
                for vehicle in vehicles:
                    # 车型锁单数
                    daily_col = f'{vehicle}锁单数'
                    if daily_col in row.index:
                        table_row[f'{vehicle}锁单数'] = int(row[daily_col]) if pd.notna(row[daily_col]) else 0
                
                for vehicle in vehicles:
                    # 车型累计锁单数
                    cumulative_col = f'{vehicle}累计锁单数'
                    if cumulative_col in row.index:
                        table_row[f'{vehicle}累计锁单数'] = int(row[cumulative_col]) if pd.notna(row[cumulative_col]) else 0
                
                for vehicle in vehicles:
                    # 车型锁单结构
                    structure_col = f'{vehicle}锁单结构'
                    if structure_col in row.index:
                        table_row[f'{vehicle}锁单结构'] = str(row[structure_col]) if pd.notna(row[structure_col]) else "0/0/0"
                
                table_data.append(table_row)
            
            return pd.DataFrame(table_data)
            
        except Exception as e:
            logger.error(f"创建锁单表现表格时出错: {e}")
            return pd.DataFrame({'错误': [f'表格生成失败: {str(e)}']})

    def prepare_delivery_data(self, selected_vehicles: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """准备交付数据分析"""
        try:
            if self.df is None or self.df.empty:
                logger.warning("数据为空")
                return pd.DataFrame()
            
            # 筛选有交付时间的订单
            delivery_df = self.df[self.df['Invoice_Upload_Time'].notna()].copy()
            
            if delivery_df.empty:
                logger.warning("没有找到交付数据")
                return pd.DataFrame()
            
            # 确保Invoice_Upload_Time为datetime类型
            delivery_df['Invoice_Upload_Time'] = pd.to_datetime(delivery_df['Invoice_Upload_Time'])
            
            # 筛选车型
            if selected_vehicles:
                delivery_df = delivery_df[delivery_df['车型分组'].isin(selected_vehicles)]
            
            # 筛选时间范围
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            delivery_df = delivery_df[
                (delivery_df['Invoice_Upload_Time'] >= start_dt) & 
                (delivery_df['Invoice_Upload_Time'] <= end_dt)
            ]
            
            if delivery_df.empty:
                logger.warning("筛选后没有数据")
                return pd.DataFrame()
            
            # 按日期和车型分组统计
            delivery_df['交付日期'] = delivery_df['Invoice_Upload_Time'].dt.date
            
            # 计算每日交付数量和开票价格
            daily_stats = []
            
            for vehicle in selected_vehicles:
                vehicle_data = delivery_df[delivery_df['车型分组'] == vehicle]
                if vehicle_data.empty:
                    continue
                
                # 按日期分组
                daily_group = vehicle_data.groupby('交付日期').agg({
                    'Order Number': 'count',  # 交付数量
                    '开票价格': 'mean'  # 平均开票价格
                }).reset_index()
                
                daily_group['车型'] = vehicle
                daily_group = daily_group.rename(columns={
                    'Order Number': '交付数量',
                    '开票价格': '开票价格'
                })
                
                daily_stats.append(daily_group)
            
            if not daily_stats:
                return pd.DataFrame()
            
            # 合并所有车型数据
            result_df = pd.concat(daily_stats, ignore_index=True)
            
            # 转换日期为datetime以便排序
            result_df['交付日期'] = pd.to_datetime(result_df['交付日期'])
            result_df = result_df.sort_values(['车型', '交付日期'])
            
            # 计算7日滚动平均
            for vehicle in selected_vehicles:
                vehicle_mask = result_df['车型'] == vehicle
                if vehicle_mask.any():
                    # 交付数量7日滚动平均
                    result_df.loc[vehicle_mask, '交付数量_7日均值'] = (
                        result_df.loc[vehicle_mask, '交付数量']
                        .rolling(window=7, min_periods=1)
                        .mean()
                    )
                    
                    # 开票价格7日滚动平均
                    result_df.loc[vehicle_mask, '开票价格_7日均值'] = (
                        result_df.loc[vehicle_mask, '开票价格']
                        .rolling(window=7, min_periods=1)
                        .mean()
                    )
            
            return result_df
            
        except Exception as e:
            logger.error(f"准备交付数据时出错: {e}")
            return pd.DataFrame()

    def create_delivery_trend_chart(self, data: pd.DataFrame) -> go.Figure:
        """创建交付趋势分离子图"""
        try:
            if data.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text="暂无交付数据",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=20, color="gray")
                )
                return fig
            
            # 创建两个子图：上方为交付数量，下方为开票价格
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("车型交付数量趋势分析（7日滚动平均）", "车型开票价格趋势分析（7日滚动平均）"),
                vertical_spacing=0.12,
                shared_xaxes=True
            )
            
            # 颜色映射
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            vehicles = data['车型'].unique()
            
            for i, vehicle in enumerate(vehicles):
                vehicle_data = data[data['车型'] == vehicle].sort_values('交付日期')
                color = colors[i % len(colors)]
                
                # 过滤掉交付数量为0或空的数据点
                delivery_data = vehicle_data[
                    (vehicle_data['交付数量_7日均值'].notna()) & 
                    (vehicle_data['交付数量_7日均值'] > 0)
                ]
                
                # 过滤掉开票价格为0或空的数据点
                invoice_data = vehicle_data[
                    (vehicle_data['开票价格_7日均值'].notna()) & 
                    (vehicle_data['开票价格_7日均值'] > 0)
                ]
                
                # 第一个子图：交付数量（7日滚动平均）- 只有当有有效数据时才添加
                if not delivery_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=delivery_data['交付日期'],
                            y=delivery_data['交付数量_7日均值'],
                            mode='lines+markers',
                            name=f'{vehicle} 交付数量',
                            line=dict(color=color, width=2),
                            marker=dict(size=4),
                            hovertemplate=(
                                f'<b>{vehicle} 交付数量</b><br>' +
                                '日期: %{x}<br>' +
                                '7日均值: %{y:.1f}<br>' +
                                '<extra></extra>'
                            ),
                            legendgroup=f'{vehicle}',
                            showlegend=True
                        ),
                        row=1, col=1
                    )
                
                # 第二个子图：开票价格（7日滚动平均）- 只有当有有效数据时才添加
                if not invoice_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=invoice_data['交付日期'],
                            y=invoice_data['开票价格_7日均值'],
                            mode='lines+markers',
                            name=f'{vehicle} 开票价格',
                            line=dict(color=color, width=2, dash='dash'),
                            marker=dict(size=4, symbol='diamond'),
                            hovertemplate=(
                                f'<b>{vehicle} 开票价格</b><br>' +
                                '日期: %{x}<br>' +
                                '7日均值: ¥%{y:,.0f}<br>' +
                                '<extra></extra>'
                            ),
                            legendgroup=f'{vehicle}',
                            showlegend=True
                        ),
                        row=2, col=1
                    )
            
            # 设置第一个子图Y轴标题（交付数量）
            fig.update_yaxes(
                title_text="交付数量",
                showgrid=True,
                gridcolor='lightgray',
                row=1, col=1
            )
            
            # 设置第二个子图Y轴标题（开票价格）
            fig.update_yaxes(
                title_text="开票价格（元）",
                showgrid=True,
                gridcolor='lightgray',
                row=2, col=1
            )
            
            # 设置X轴（只在底部子图显示标题）
            fig.update_xaxes(
                showgrid=True,
                gridcolor='lightgray',
                row=1, col=1
            )
            
            fig.update_xaxes(
                title_text="交付日期",
                showgrid=True,
                gridcolor='lightgray',
                row=2, col=1
            )
            
            # 更新布局
            fig.update_layout(
                title={
                    'text': '车型交付数量与开票价格趋势分析',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.15,
                    xanchor="center",
                    x=0.5
                ),
                height=800,
                margin=dict(t=80, b=100, l=60, r=60),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"创建交付趋势图表时出错: {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"图表生成失败: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig

    def create_delivery_detail_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建交付详细数据表格"""
        try:
            if data.empty:
                return pd.DataFrame({'提示': ['暂无交付数据']})
            
            # 准备表格数据
            table_data = []
            
            for _, row in data.iterrows():
                table_row = {
                    '交付日期': row['交付日期'].strftime('%Y-%m-%d'),
                    '车型': row['车型'],
                    '当日交付数量': int(row['交付数量']),
                    '交付数量(7日均值)': f"{row['交付数量_7日均值']:.1f}",
                    '开票价格(7日均值)': f"¥{row['开票价格_7日均值']:,.0f}"
                }
                table_data.append(table_row)
            
            result_df = pd.DataFrame(table_data)
            
            # 按日期倒序排列，显示最新数据
            result_df = result_df.sort_values('交付日期', ascending=False)
            
            return result_df
            
        except Exception as e:
            logger.error(f"创建交付详细表格时出错: {e}")
            return pd.DataFrame({'错误': [f'表格生成失败: {str(e)}']})

# 创建监控器实例
monitor = OrderTrendMonitor()

def update_charts(selected_vehicles, days_after_launch=1):
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
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_df, empty_df
        
        # 准备数据
        daily_data = monitor.prepare_daily_data(selected_vehicles)
        
        # 创建图表
        cumulative_chart = monitor.create_cumulative_chart(daily_data)
        daily_chart = monitor.create_daily_chart(daily_data)
        change_trend_chart = monitor.create_change_trend_chart(daily_data)
        conversion_rate_chart = monitor.create_conversion_rate_chart(selected_vehicles, days_after_launch)
        summary_statistics_table = monitor.prepare_summary_statistics_data(selected_vehicles, days_after_launch)
        daily_table = monitor.create_daily_change_table(daily_data, days_after_launch)
        
        return cumulative_chart, daily_chart, change_trend_chart, conversion_rate_chart, summary_statistics_table, daily_table
        
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
        return error_fig, error_fig, error_fig, error_fig, error_df, error_df

def update_refund_charts(selected_vehicles, city_order_min=100, city_order_max=2000):
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
            return empty_fig, empty_fig, empty_fig, empty_df, empty_df, empty_df

        # 准备退订相关数据
        refund_data = monitor.prepare_refund_data(selected_vehicles)
        refund_rate_data = monitor.prepare_refund_rate_data(selected_vehicles)
        daily_refund_rate_data = monitor.prepare_daily_refund_rate_data(selected_vehicles)
        regional_summary_data = monitor.prepare_regional_summary_data(selected_vehicles)
        city_summary_data = monitor.prepare_city_summary_data(selected_vehicles)
        
        # 创建图表
        cumulative_refund_chart = monitor.create_cumulative_refund_chart(refund_data)
        cumulative_refund_rate_chart = monitor.create_cumulative_refund_rate_chart(refund_rate_data)
        daily_refund_rate_chart = monitor.create_daily_refund_rate_chart(daily_refund_rate_data)
        daily_refund_table = monitor.create_daily_refund_table(daily_refund_rate_data)
        regional_summary_table = monitor.create_regional_summary_table(regional_summary_data)
        city_summary_table = monitor.create_city_summary_table(city_summary_data, [city_order_min, city_order_max])
        
        return cumulative_refund_chart, cumulative_refund_rate_chart, daily_refund_rate_chart, daily_refund_table, regional_summary_table, city_summary_table
        
    except Exception as e:
        import traceback
        logger.error(f"退订图表更新失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"错误: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        error_df = pd.DataFrame({'错误': [str(e)]})
        return error_fig, error_fig, error_fig, error_df, error_df, error_df

def update_delivery_analysis(selected_vehicles, start_date, end_date):
    """更新交付分析"""
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
            stats_text = "请选择车型和时间范围，点击分析按钮查看统计信息"
            return empty_fig, empty_df, stats_text
        
        # 准备交付数据
        delivery_data = monitor.prepare_delivery_data(selected_vehicles, start_date, end_date)
        
        if delivery_data.empty:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="所选时间范围内暂无交付数据",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="orange")
            )
            empty_df = pd.DataFrame({'提示': ['所选时间范围内暂无交付数据']})
            stats_text = "所选时间范围内暂无交付数据，请调整筛选条件"
            return empty_fig, empty_df, stats_text
        
        # 创建图表和表格
        delivery_chart = monitor.create_delivery_trend_chart(delivery_data)
        delivery_table = monitor.create_delivery_detail_table(delivery_data)
        
        # 生成统计信息
        total_deliveries = delivery_data['交付数量'].sum()
        avg_price = delivery_data['开票价格'].mean()
        date_range = f"{delivery_data['交付日期'].min().strftime('%Y-%m-%d')} 至 {delivery_data['交付日期'].max().strftime('%Y-%m-%d')}"
        
        stats_text = f"""
        ### 📈 交付统计信息
        
        **分析时间范围**: {date_range}
        
        **选中车型**: {', '.join(selected_vehicles)}
        
        **总交付数量**: {total_deliveries:,} 台
        
        **平均开票价格**: ¥{avg_price:,.0f}
        
        **数据点数量**: {len(delivery_data)} 个
        """
        
        return delivery_chart, delivery_table, stats_text
        
    except Exception as e:
        logger.error(f"交付分析更新失败: {str(e)}")
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"分析失败: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        error_df = pd.DataFrame({'错误': [str(e)]})
        error_stats = f"分析失败: {str(e)}"
        return error_fig, error_df, error_stats

def update_lock_charts(selected_vehicles, n_days):
    """更新锁单图表"""
    try:
        if not selected_vehicles:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="请选择车型",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return empty_fig, empty_fig, empty_fig, empty_fig

        # 准备锁单相关数据
        lock_data = monitor.prepare_lock_data(selected_vehicles, n_days)
        conversion_data = monitor.prepare_lock_conversion_data(selected_vehicles, n_days)
        change_data = monitor.prepare_daily_lock_change_data(selected_vehicles, n_days)
        
        # 创建图表
        cumulative_lock_chart = monitor.create_cumulative_lock_chart(lock_data)
        conversion_chart = monitor.create_lock_conversion_chart(conversion_data)
        daily_lock_chart = monitor.create_daily_lock_chart(lock_data)
        change_chart = monitor.create_daily_lock_change_chart(change_data)
        
        return cumulative_lock_chart, conversion_chart, daily_lock_chart, change_chart
        
    except Exception as e:
        import traceback
        logger.error(f"锁单图表更新失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"错误: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return error_fig, error_fig, error_fig, error_fig

def update_lock_performance_table(selected_vehicles, n_days):
    """更新锁单表现表格"""
    try:
        if not selected_vehicles:
            return pd.DataFrame({'提示': ['请选择车型']})
        
        # 准备锁单表现数据
        performance_data = monitor.prepare_lock_performance_table_data(selected_vehicles, n_days)
        
        # 创建表格
        performance_table = monitor.create_lock_performance_table(performance_data)
        
        return performance_table
        
    except Exception as e:
        import traceback
        logger.error(f"锁单表现表格更新失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        return pd.DataFrame({'错误': [f'表格更新失败: {str(e)}']})



def update_config_table(selected_vehicles, start_date, end_date, product_categories, lock_start_date, lock_end_date, lock_n_days, age_include_unknown, gender_include_unknown, region_include_unknown, region_virtual_filter, region_fac_filter, weekend_lock_filter, min_lock_count, max_lock_count):
    """更新配置模块所有锁单统计表格"""
    try:
        if not selected_vehicles:
            empty_df = pd.DataFrame({'提示': ['请选择车型']})
            return empty_df, empty_df, empty_df, empty_df, empty_df, empty_df
        
        # 准备Product Name锁单数据
        product_data = monitor.prepare_product_name_lock_data(
            selected_vehicles, start_date, end_date, product_categories, lock_start_date, lock_end_date, lock_n_days, weekend_lock_filter
        )
        product_table = monitor.create_product_name_lock_table(product_data)
        
        # 准备中间渠道锁单数据
        channel_data = monitor.prepare_channel_lock_data(
            selected_vehicles, start_date, end_date, product_categories, lock_start_date, lock_end_date, lock_n_days, weekend_lock_filter
        )
        channel_table = monitor.create_channel_lock_table(channel_data)
        
        # 准备买家年龄锁单数据
        age_data = monitor.prepare_age_lock_data(
            selected_vehicles, start_date, end_date, product_categories, lock_start_date, lock_end_date, lock_n_days, age_include_unknown, weekend_lock_filter
        )
        age_table = monitor.create_age_lock_table(age_data)
        
        # 准备订单性别锁单数据
        gender_data = monitor.prepare_gender_lock_data(
            selected_vehicles, start_date, end_date, product_categories, lock_start_date, lock_end_date, lock_n_days, gender_include_unknown, weekend_lock_filter
        )
        gender_table = monitor.create_gender_lock_table(gender_data)
        
        # 准备父级区域锁单数据
        region_data = monitor.prepare_region_lock_data(
            selected_vehicles, start_date, end_date, product_categories, lock_start_date, lock_end_date, lock_n_days, region_include_unknown, region_virtual_filter, region_fac_filter, weekend_lock_filter
        )
        region_table = monitor.create_region_lock_table(region_data)
        
        # 准备License City锁单数据
        city_data = monitor.prepare_city_lock_data(
            selected_vehicles, start_date, end_date, lock_start_date, lock_end_date, lock_n_days, product_categories, weekend_lock_filter, min_lock_count, max_lock_count
        )
        city_table = monitor.create_city_lock_table(city_data)
        
        return product_table, channel_table, age_table, gender_table, region_table, city_table
        
    except Exception as e:
        import traceback
        logger.error(f"配置模块表格更新失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        error_df = pd.DataFrame({'错误': [f'表格更新失败: {str(e)}']})
        return error_df, error_df, error_df, error_df, error_df, error_df

# 获取车型分组
vehicle_groups = monitor.get_vehicle_groups()

# 创建Gradio界面
with gr.Blocks(title="小订订单趋势监测", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚗 小订订单趋势监测工具")
    gr.Markdown("监测各车型小订订单的趋势变化，支持多维度对比分析")
    
    with gr.Tabs():
        # 订单模块
        with gr.Tab("📊 订单"):
            with gr.Row():
                with gr.Column(scale=1):
                    vehicle_selector = gr.CheckboxGroup(
                        choices=vehicle_groups,
                        label="选择车型分组",
                        value=["CM2", "CM1"] if "CM2" in vehicle_groups and "CM1" in vehicle_groups else vehicle_groups[:2],
                        interactive=True
                    )
                with gr.Column(scale=1):
                    days_after_launch = gr.Number(
                        label="发布会后第N日",
                        value=6,
                        minimum=0,
                        maximum=30,
                        step=1,
                        info="计算小订转化率的时间点（1表示发布会当天）"
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
                    conversion_rate_plot = gr.Plot(label="车型小订转化率对比图")
           
            with gr.Row():
                summary_statistics_table = gr.DataFrame(
                    label="汇总统计表格",
                    interactive=False,
                    wrap=True,
                    datatype=["str", "number", "number", "number", "number", "number", "number", "number"]  # 车型(str) + 累计预售天数(number) + 累计预售小订数(number) + 发布会后N日累计锁单数(number) + 小订留存锁单数(number) + 发布会后小订锁单数(number) + 直接锁单数(number) + 小订转化率(number)
                )
            
            with gr.Row():
                daily_table = gr.DataFrame(
                    label="订单日变化表格",
                    interactive=False,
                    wrap=True,
                    datatype=["str"] + ["html"] * 20  # 支持更多列：第N日(str) + 多个车型的各项指标(html)
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
                        wrap=True,
                        datatype=["str"] + ["html"] * 20
                    )
            
            with gr.Accordion("📊 分区域汇总表格", open=True):
                with gr.Row():
                    regional_summary_table = gr.DataFrame(
                        label="车型对比：分区域累计订单/退订数(退订率)汇总表格",
                        interactive=False,
                        wrap=True,
                        datatype=["str"] + ["html"] * 20
                    )
            
            with gr.Accordion("🏙️ 分城市汇总表格", open=False):
                with gr.Row():
                    with gr.Row():
                        city_order_min = gr.Number(
                            label="最小小订数",
                            value=200,
                            minimum=0,
                            step=1,
                            scale=1
                        )
                        city_order_max = gr.Number(
                            label="最大小订数",
                            value=5000,
                            minimum=0,
                            step=1,
                            scale=1
                        )
                with gr.Row():
                    city_summary_table = gr.DataFrame(
                        label="车型对比：分城市累计订单/退订数(退订率)汇总表格",
                        interactive=False,
                        wrap=True,
                        datatype=["str"] + ["html"] * 20
                    )
        
        # 锁单模块
        with gr.Tab("🔒 锁单"):
            with gr.Row():
                lock_vehicle_selector = gr.CheckboxGroup(
                    choices=vehicle_groups,
                    label="选择车型分组",
                    value=["CM2", "CM1"] if "CM2" in vehicle_groups and "CM1" in vehicle_groups else vehicle_groups[:2],
                    interactive=True
                )
                lock_n_days = gr.Number(
                    label="N天数（基于business_definition.json最大值+N天）",
                    value=30,
                    minimum=1,
                    maximum=100,
                    step=1,
                    info="输入N天数，用于计算X轴第N日"
                )
            
            with gr.Row():
                with gr.Column(scale=1):
                    cumulative_lock_plot = gr.Plot(label="累计锁单数对比图")
                with gr.Column(scale=1):
                    lock_conversion_rate_plot = gr.Plot(label="累计小订转化率对比图")
            
            with gr.Row():
                with gr.Column(scale=1):
                    daily_lock_plot = gr.Plot(label="每日锁单数对比图")
                with gr.Column(scale=1):
                    daily_lock_change_plot = gr.Plot(label="每日锁单数环比变化图")
            
            with gr.Row():
                with gr.Accordion("📊 锁单表现表格", open=False):
                    lock_performance_table = gr.DataFrame(
                        label="锁单表现表格",
                        interactive=False,
                        wrap=True
                    )
        
        # 配置模块
        with gr.Tab("⚙️ 配置"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 筛选条件")
                    
                    with gr.Group():
                        gr.Markdown("#### 车型选择")
                        config_vehicle_selector = gr.CheckboxGroup(
                            choices=vehicle_groups,
                            label="选择车型分组",
                            value=["CM2", "CM1"] if "CM2" in vehicle_groups and "CM1" in vehicle_groups else vehicle_groups[:2],
                            interactive=True
                        )
                    
                    with gr.Group():
                        gr.Markdown("#### 时间范围筛选")
                        with gr.Row():
                            config_start_date = gr.Textbox(
                                label="小订开始日期",
                                placeholder="YYYY-MM-DD（可选）",
                                value="2025-08-15"
                            )
                            config_end_date = gr.Textbox(
                                label="小订结束日期",
                                placeholder="YYYY-MM-DD（可选）",
                                value=""
                            )
                        
                        with gr.Row():
                            config_lock_start_date = gr.Textbox(
                                label="锁单开始日期",
                                placeholder="YYYY-MM-DD（可选）",
                                value="2025-09-10"
                            )
                            config_lock_end_date = gr.Textbox(
                                label="锁单结束日期",
                                placeholder="YYYY-MM-DD（可选）",
                                value=""
                            )
                        
                        with gr.Row():
                            config_lock_n_days = gr.Number(
                                label="锁单后N天数（基于business_definition.json最大值+N天）",
                                value=30,
                                minimum=1,
                                maximum=100,
                                step=1,
                                info="输入N天数，用于计算锁单后第N日数据"
                            )
                    
                    with gr.Group():
                        gr.Markdown("#### 产品分类筛选")
                        config_product_types = gr.CheckboxGroup(
                            choices=["增程", "纯电"],
                            label="产品分类（增程/纯电）",
                            value=[],
                            interactive=True
                        )
                    
                    with gr.Group():
                        gr.Markdown("#### 周末锁单筛选")
                        config_weekend_lock_filter = gr.Radio(
                            choices=["全部", "仅周末锁单", "仅工作日锁单"],
                            label="锁单时间筛选",
                            value="全部",
                            interactive=True,
                            info="根据Lock_Time是否为周末（周六、周日）进行筛选"
                        )
                    
                    with gr.Row():
                        config_analyze_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")

                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Product Name锁单统计")
                    config_product_table = gr.DataFrame(
                        label="Product Name锁单统计表格",
                        interactive=False,
                        wrap=True
                    )
                    
                    gr.Markdown("### 📊 first_middle_channel_name锁单统计")
                    config_channel_table = gr.DataFrame(
                        label="first_middle_channel_name锁单统计表格",
                        interactive=False,
                        wrap=True,
                        datatype=["str"] + ["html"] * 20
                    )
                    
                    gr.Markdown("### 📊 buyer_age锁单统计")
                    with gr.Row():
                        config_age_include_unknown = gr.Checkbox(
                            label="包含未知年龄数据",
                            value=True,
                            info="取消勾选将过滤掉年龄为'未知年龄'的数据"
                        )
                    config_age_table = gr.DataFrame(
                        label="buyer_age锁单统计表格",
                        interactive=False,
                        wrap=True,
                        datatype=["str"] + ["html"] * 20
                    )
                    
                    gr.Markdown("### 📊 order_gender锁单统计")
                    with gr.Row():
                        config_gender_include_unknown = gr.Checkbox(
                            label="包含未知性别数据",
                            value=True,
                            info="取消勾选将过滤掉性别为'未知性别'的数据"
                        )
                    config_gender_table = gr.DataFrame(
                        label="order_gender锁单统计表格",
                        interactive=False,
                        wrap=True,
                        datatype=["str"] + ["html"] * 20
                    )
                    
                    gr.Markdown("### 📊 Parent Region Name锁单统计")
                    with gr.Row():
                        config_region_include_unknown = gr.Checkbox(
                            label="包含未知区域数据",
                            value=True,
                            info="取消勾选将过滤掉区域为'未知区域'的数据"
                        )
                    with gr.Row():
                        config_region_virtual_filter = gr.Checkbox(
                            label="虚拟大区",
                            value=True,
                            info="勾选显示虚拟大区数据"
                        )
                        config_region_fac_filter = gr.Checkbox(
                            label="FAC大区",
                            value=True,
                            info="勾选显示FAC大区数据"
                        )
                    config_region_table = gr.DataFrame(
                        label="Parent Region Name锁单统计表格",
                        interactive=False,
                        wrap=True,
                        datatype=["str"] + ["html"] * 20
                    )
                
                    with gr.Accordion("📊 License City锁单统计", open=False):
                        with gr.Row():
                            config_city_lock_min = gr.Number(
                                label="最小锁单数",
                                value=100,
                                minimum=0,
                                step=1,
                                scale=1
                            )
                            config_city_lock_max = gr.Number(
                                label="最大锁单数",
                                value=1000,
                                minimum=0,
                                step=1,
                                scale=1
                            )
                        with gr.Row():
                            config_city_table = gr.DataFrame(
                                label="License City锁单统计表格",
                                interactive=False,
                                wrap=True,
                                datatype=["str"] + ["html"] * 20
                            )
        
        # 交付模块
        with gr.Tab("📦 交付"):
            gr.Markdown("### 车型分组交付数量和开票价格分析")
            gr.Markdown("""
            📊 **功能说明**：
            - 分析不同车型分组的交付数量趋势（基于Invoice_Upload_Time）
            - 展示开票价格的变化趋势（7日滚动平均）
            - 支持车型筛选和时间范围自定义
            
            💡 **使用方法**：选择车型和时间范围，点击分析按钮查看交付趋势
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 📋 筛选条件")
                    
                    # 车型选择器
                    delivery_vehicle_selector = gr.CheckboxGroup(
                        label="🚗 车型选择",
                        choices=monitor.get_vehicle_groups(),
                        value=["CM2", "CM1"],
                        info="选择要分析的车型分组"
                    )
                    
                    # 时间范围选择器
                    with gr.Row():
                        delivery_start_date = gr.Textbox(
                            label="📅 开始日期",
                            value="2024-05-13",
                            placeholder="YYYY-MM-DD",
                            info="选择交付分析的开始日期"
                        )
                        delivery_end_date = gr.Textbox(
                            label="📅 结束日期", 
                            value="2025-12-31",
                            placeholder="YYYY-MM-DD",
                            info="选择交付分析的结束日期"
                        )
                    
                    # 分析按钮
                    delivery_analyze_btn = gr.Button(
                        "📊 开始分析",
                        variant="primary",
                        size="lg"
                    )
                    
                    # 统计信息显示
                    delivery_stats = gr.Markdown(
                        label="📈 统计信息",
                        value="请选择车型和时间范围，点击分析按钮查看统计信息"
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("#### 📈 交付趋势图表")
                    
                    # 双轴折线图
                    delivery_trend_plot = gr.Plot(
                        label="交付数量与开票价格趋势图",
                        value=None
                    )
                    
                    # 详细数据表格
                    delivery_detail_table = gr.DataFrame(
                        label="交付详细数据",
                        interactive=False,
                        wrap=True,
                        datatype=["str", "number", "number", "number", "number"]
                    )

        # 预测模块（占位）
        with gr.Tab("🔮 预测"):
            gr.Markdown("### 基于CM1历史数据和CM2部分样本的CM2全周期小订数预测")
            gr.Markdown("""
            📊 **功能说明**：
            - 基于CM1的完整28天历史数据和CM2已有的24天数据
            - 使用S型增长曲线和分段回归模型进行预测
            - 支持预测CM2在任意目标天数的累计订单数
            
            💡 **使用方法**：输入目标天数，系统将自动生成预测图表和结果分析
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    target_days_input = gr.Number(
                        label="目标天数",
                        value=28,
                        minimum=1,
                        maximum=50,
                        step=1,
                        info="输入要预测的CM2目标天数（1-50天）"
                    )
                    predict_button = gr.Button("🔮 开始预测", variant="primary", size="lg")
                    
                    prediction_result = gr.Markdown(
                        label="预测结果",
                        value="请输入目标天数并点击预测按钮"
                    )
                
                with gr.Column(scale=2):
                    prediction_plot = gr.Plot(
                        label="CM1 vs CM2 订单分析与预测图表",
                        value=None
                    )
            
            # 预测按钮点击事件
            predict_button.click(
                fn=predict_orders,
                inputs=[target_days_input],
                outputs=[prediction_plot, prediction_result]
            )
            
            # 输入改变时自动预测
            target_days_input.change(
                fn=predict_orders,
                inputs=[target_days_input],
                outputs=[prediction_plot, prediction_result]
            )
    
    # 绑定事件
    vehicle_selector.change(
        fn=update_charts,
        inputs=[vehicle_selector, days_after_launch],
        outputs=[cumulative_plot, daily_plot, change_trend_plot, conversion_rate_plot, summary_statistics_table, daily_table]
    )
    
    days_after_launch.change(
        fn=update_charts,
        inputs=[vehicle_selector, days_after_launch],
        outputs=[cumulative_plot, daily_plot, change_trend_plot, conversion_rate_plot, summary_statistics_table, daily_table]
    )
    
    refund_vehicle_selector.change(
        fn=update_refund_charts,
        inputs=[refund_vehicle_selector, city_order_min, city_order_max],
        outputs=[cumulative_refund_plot, cumulative_refund_rate_plot, daily_refund_rate_plot, daily_refund_table, regional_summary_table, city_summary_table]
    )
    
    city_order_min.change(
        fn=update_refund_charts,
        inputs=[refund_vehicle_selector, city_order_min, city_order_max],
        outputs=[cumulative_refund_plot, cumulative_refund_rate_plot, daily_refund_rate_plot, daily_refund_table, regional_summary_table, city_summary_table]
    )
    
    city_order_max.change(
        fn=update_refund_charts,
        inputs=[refund_vehicle_selector, city_order_min, city_order_max],
        outputs=[cumulative_refund_plot, cumulative_refund_rate_plot, daily_refund_rate_plot, daily_refund_table, regional_summary_table, city_summary_table]
    )
    
    # 锁单模块事件绑定
    lock_vehicle_selector.change(
        fn=update_lock_charts,
        inputs=[lock_vehicle_selector, lock_n_days],
        outputs=[cumulative_lock_plot, lock_conversion_rate_plot, daily_lock_plot, daily_lock_change_plot]
    )
    
    lock_n_days.change(
        fn=update_lock_charts,
        inputs=[lock_vehicle_selector, lock_n_days],
        outputs=[cumulative_lock_plot, lock_conversion_rate_plot, daily_lock_plot, daily_lock_change_plot]
    )
    
    # 锁单表现表格事件绑定
    lock_vehicle_selector.change(
        fn=update_lock_performance_table,
        inputs=[lock_vehicle_selector, lock_n_days],
        outputs=[lock_performance_table]
    )
    
    lock_n_days.change(
        fn=update_lock_performance_table,
        inputs=[lock_vehicle_selector, lock_n_days],
        outputs=[lock_performance_table]
    )
    
    # 配置模块事件绑定 - 仅通过按钮触发分析
    config_analyze_btn.click(
        fn=update_config_table,
        inputs=[config_vehicle_selector, config_start_date, config_end_date, config_product_types, config_lock_start_date, config_lock_end_date, config_lock_n_days, config_age_include_unknown, config_gender_include_unknown, config_region_include_unknown, config_region_virtual_filter, config_region_fac_filter, config_weekend_lock_filter, config_city_lock_min, config_city_lock_max],
        outputs=[config_product_table, config_channel_table, config_age_table, config_gender_table, config_region_table, config_city_table]
    )
    
    # 交付模块事件绑定 - 仅通过按钮触发分析
    delivery_analyze_btn.click(
        fn=update_delivery_analysis,
        inputs=[delivery_vehicle_selector, delivery_start_date, delivery_end_date],
        outputs=[delivery_trend_plot, delivery_detail_table, delivery_stats]
    )
    
    # 页面加载时自动更新
    demo.load(
        fn=update_charts,
        inputs=[vehicle_selector, days_after_launch],
        outputs=[cumulative_plot, daily_plot, change_trend_plot, conversion_rate_plot, summary_statistics_table, daily_table]
    )
    
    demo.load(
        fn=update_refund_charts,
        inputs=[refund_vehicle_selector, city_order_min, city_order_max],
        outputs=[cumulative_refund_plot, cumulative_refund_rate_plot, daily_refund_rate_plot, daily_refund_table, regional_summary_table, city_summary_table]
    )
    
    demo.load(
        fn=update_lock_charts,
        inputs=[lock_vehicle_selector, lock_n_days],
        outputs=[cumulative_lock_plot, lock_conversion_rate_plot, daily_lock_plot, daily_lock_change_plot]
    )
    
    demo.load(
        fn=update_lock_performance_table,
        inputs=[lock_vehicle_selector, lock_n_days],
        outputs=[lock_performance_table]
    )
    
    demo.load(
        fn=update_config_table,
        inputs=[config_vehicle_selector, config_start_date, config_end_date, config_product_types, config_lock_start_date, config_lock_end_date, config_lock_n_days, config_age_include_unknown, config_gender_include_unknown, config_region_include_unknown, config_region_virtual_filter, config_region_fac_filter, config_weekend_lock_filter, config_city_lock_min, config_city_lock_max],
        outputs=[config_product_table, config_channel_table, config_age_table, config_gender_table, config_region_table, config_city_table]
    )
    
    # 界面加载时初始化预测模块
    def init_prediction():
        predictor.train_models()
        fig, result_text = predict_orders(28)
        return fig, result_text
    
    demo.load(
        fn=init_prediction,
        outputs=[prediction_plot, prediction_result]
    )
    
    with gr.Accordion("📖 使用说明", open=False):
        gr.Markdown("""
        ### 功能说明
        
        **订单模块**包含五个核心图表：
        1. **累计小订订单数对比图**: 展示各车型从预售开始的累计订单趋势
        2. **每日小订单数对比图**: 对比各车型每日的订单量
        3. **每日小订单数环比变化趋势图**: 显示订单量的日环比变化率
        4. **车型小订转化率对比图**: 对比各车型每日的小订转化率趋势
        5. **订单日变化表格**: 详细的数据表格，包含emoji标记的变化趋势
        
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