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
            business_def = json.load(f)
        
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
    
    def create_daily_change_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建订单的日变化表格"""
        if data.empty:
            return pd.DataFrame({'提示': ['暂无数据']})
        
        # 准备表格数据
        table_data = []
        
        for vehicle in data['车型分组'].unique():
            vehicle_data = data[data['车型分组'] == vehicle].sort_values('days_from_start')
            
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
            vehicle_data = vehicle_data[vehicle_data['days_from_end'] > 1]
            
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
            
            with gr.Accordion("📊 分区域汇总表格", open=True):
                with gr.Row():
                    regional_summary_table = gr.DataFrame(
                        label="车型对比：分区域累计订单/退订数(退订率)汇总表格",
                        interactive=False,
                        wrap=True
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
                        wrap=True
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
        
        # 配置模块（占位）
        with gr.Tab("⚙️ 配置"):
            gr.Markdown("### 配置模块")
            gr.Markdown("*此模块待开发*")
        
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
        inputs=[vehicle_selector],
        outputs=[cumulative_plot, daily_plot, change_trend_plot, daily_table]
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
    
    # 页面加载时自动更新
    demo.load(
        fn=update_charts,
        inputs=[vehicle_selector],
        outputs=[cumulative_plot, daily_plot, change_trend_plot, daily_table]
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
    
    # 界面加载时初始化预测模块
    def init_prediction():
        predictor.train_models()
        return predict_orders(28)
    
    demo.load(
        fn=init_prediction,
        outputs=[prediction_plot, prediction_result]
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