import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# 数据准备
cm1_data = [
    (1, 2165, 2165), (2, 1089, 3254), (3, 936, 4190), (4, 573, 4763),
    (5, 502, 5265), (6, 463, 5728), (7, 433, 6161), (8, 481, 6642),
    (9, 865, 7507), (10, 955, 8462), (11, 446, 8908), (12, 384, 9292),
    (13, 370, 9662), (14, 372, 10034), (15, 365, 10399), (16, 531, 10930),
    (17, 721, 11651), (18, 621, 12272), (19, 730, 13002), (20, 473, 13475),
    (21, 528, 14003), (22, 600, 14603), (23, 1164, 15767), (24, 1331, 17098),
    (25, 1169, 18267), (26, 1503, 19770), (27, 3000, 22770), (28, 4167, 26937)
]

cm2_data = [
    (1, 5351, 5351), (2, 3126, 8477), (3, 2207, 10684), (4, 1079, 11763),
    (5, 845, 12608), (6, 880, 13488), (7, 873, 14361), (8, 695, 15056),
    (9, 1300, 16356), (10, 1278, 17634), (11, 684, 18318), (12, 700, 19018),
    (13, 701, 19719), (14, 601, 20320), (15, 1029, 21349), (16, 1656, 23005),
    (17, 1773, 24778), (18, 864, 25642), (19, 886, 26528), (20, 730, 27258),
    (21, 911, 28169), (22, 1157, 29326), (23, 2124, 31459), (24, 2448, 33907)
]

# 转换为DataFrame
cm1_df = pd.DataFrame(cm1_data, columns=['day', 'daily_orders', 'cumulative_orders'])
cm2_df = pd.DataFrame(cm2_data, columns=['day', 'daily_orders', 'cumulative_orders'])

class OrderPredictor:
    def __init__(self):
        self.cm1_model = None
        self.cm2_model = None
        self.cm1_growth_phases = None
        self.cm2_growth_phases = None
        self.train_models()
    
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
        
        # 分析增长阶段
        self.cm1_growth_phases = self.analyze_growth_phases(cm1_df, 'CM1')
        self.cm2_growth_phases = self.analyze_growth_phases(cm2_df, 'CM2')
        
        # 为CM1训练S型曲线模型
        X_cm1 = cm1_df['day'].values
        y_cm1 = cm1_df['cumulative_orders'].values
        
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
        X_cm2 = cm2_df['day'].values
        y_cm2 = cm2_df['cumulative_orders'].values
        
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
        if target_days <= len(cm2_data):
            # 如果目标天数在已有数据范围内，直接返回实际数据
            return cm2_df[cm2_df['day'] <= target_days]['cumulative_orders'].iloc[-1]
        
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
        if target_days > len(cm2_data):
            prediction_days = np.arange(len(cm2_data) + 1, target_days + 1)
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
    if target_days > len(cm2_data):
        last_cumulative = cm2_df['cumulative_orders'].iloc[-1]
        for day in range(len(cm2_data) + 1, target_days + 1):
            current_cumulative = predictor.predict_cm2(target_days, current_day=day)
            if day == len(cm2_data) + 1:
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
        prediction_days = list(range(len(cm2_data) + 1, target_days + 1))
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
        # width=1000,
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
        if target_days <= len(cm2_data):
            result_text = f"""📊 **预测结果**
            
**目标天数**: {target_days} 天
**累计订单数**: {predicted_cumulative:,.0f} 单 (实际数据)

💡 **说明**: 该天数在现有数据范围内，显示的是实际累计订单数。
            """
        else:
            # 计算相比最后一天的增长
            last_actual = cm2_df['cumulative_orders'].iloc[-1]
            growth = predicted_cumulative - last_actual
            growth_rate = (growth / last_actual) * 100
            
            result_text = f"""📊 **预测结果**
            
**目标天数**: {target_days} 天
**预测累计订单数**: {predicted_cumulative:,.0f} 单
**相比第{len(cm2_data)}天增长**: {growth:,.0f} 单 (+{growth_rate:.1f}%)

💡 **说明**: 预测基于CM1全量数据和CM2已有{len(cm2_data)}天数据的趋势分析。
            """
        
        return fig, result_text
        
    except ValueError:
        return None, "请输入有效的数字"
    except Exception as e:
        return None, f"预测过程中出现错误: {str(e)}"

# 创建Gradio界面
def create_interface():
    with gr.Blocks(title="CM2订单数预测工具", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎯 CM2订单累计数预测工具 (S型曲线建模)
        
        基于CM1的全量数据（28天）和CM2的部分数据（24天）进行**S型曲线**和**分段建模**预测。
        
        **🔍 识别的关键特征**:
        - 📈 **典型S型增长**: 初期爆发 → 中段平台期 → 后期重新加速
        - 🚀 **CM1特征**: 0-0.107分位值快速积累(2165→4190)，0.143-0.536分位值平稳，0.571分位值后重新加速，在0.964分位值处大幅跃升
        - 🔄 **CM2特征**: 参考CM1的S型模式，0-0.107分位值初期爆发，0.143-0.5分位值平台期，0.536分位值后预期进入拉升期
        - 📊 两个车型都经历约10天的"稳定吸单"低速增长期，然后迎来二次爆发
        
        **🚀 功能特点**:
        - 📈 显示CM1完整S型趋势曲线作为参考，包含0.571分位值后重新加速和0.964分位值处的大幅跃升
        - 📊 CM2已发生数据用实线显示，预测数据用虚线显示
        - 🎯 **分段预测**: 初期冲高期、平台期、后期拉升期分别建模
        - 🚀 **加速预测**: 基于归一化时间分位值，参考CM1在0.964分位值处的爆发模式进行智能加速预测
        - 🔮 **S型曲线拟合**: 捕捉长期增长的饱和特征和阶段性加速
        - 🎛️ 交互式输入目标天数获得智能预测结果
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                target_days_input = gr.Number(
                    label="输入CM2目标天数",
                    value=27,
                    minimum=1,
                    maximum=100,
                    step=1,
                    info="输入您想预测的CM2总天数"
                )
                
                predict_btn = gr.Button(
                    "🔮 开始预测",
                    variant="primary",
                    size="lg"
                )
                
                result_text = gr.Markdown(
                    value="请输入目标天数并点击预测按钮",
                    label="预测结果"
                )
            
            with gr.Column(scale=2):
                plot_output = gr.Plot(
                    label="订单趋势预测图"
                )
        
        # 添加数据说明
        with gr.Accordion("📋 数据说明与建模方法", open=False):
            gr.Markdown("""
            **📊 数据概况**:
            - **CM1数据**: 完整28天数据，展现完整的S型增长周期
            - **CM2数据**: 已有24天实际数据，正处于拉升期加速阶段
            
            **🔬 建模方法**: 
            - **S型曲线拟合**: 使用Sigmoid函数捕捉整体增长饱和特征
            - **分段建模**: 针对不同增长阶段分别训练预测模型
              - 🚀 **初期冲高期** (1-3天): 高增长率的启动阶段
              - 📊 **平台期** (4-14天): 稳定吸单的低速增长阶段  
              - 📈 **后期拉升期** (15天+): 二次增长加速阶段
            - **趋势外推**: 基于最近增长率进行长期预测
            - **连续性保证**: 确保预测曲线的平滑过渡
            
            **📈 图表说明**:
            - 🔵 **蓝色线**: CM1参考曲线 (完整S型周期)
            - 🔴 **红色实线**: CM2实际数据 (已发生)
            - 🔴 **红色虚线**: CM2预测数据 (基于分段模型)
            
            **💡 预测逻辑**:
            根据目标天数自动选择最适合的预测模型，充分考虑S型增长的阶段性特征。
            """)
        
        # 绑定事件
        predict_btn.click(
            fn=predict_orders,
            inputs=[target_days_input],
            outputs=[plot_output, result_text]
        )
        
        # 实时预测（当输入改变时）
        target_days_input.change(
            fn=predict_orders,
            inputs=[target_days_input],
            outputs=[plot_output, result_text]
        )
        
        # 界面加载时初始化预测
        demo.load(
            fn=predict_orders,
            inputs=[target_days_input],
            outputs=[plot_output, result_text]
        )
    
    return demo

if __name__ == "__main__":
    # 创建并启动界面
    demo = create_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )