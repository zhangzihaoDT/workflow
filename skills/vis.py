import plotly.graph_objects as go
import numpy as np

# 车型列表
models = ['CM0', 'CM1', 'CM2']

# 锁单率数据
# CM0/CM1 使用线性插值
days_points_base = [3, 5, 30]
rates_base = {
    'CM0': [9.7, 13.5, 31.1],
    'CM1': [10.9, 12.8, 24.5],
}

# CM2 使用贝叶斯更新：3日先验 → 5日观测 → 30日预测
# 贝叶斯后验 5日：15.65%
# 30日预测：32.9%
days_cm2 = [3, 5, 30]
rates_cm2 = [13.7, 15.65, 32.9]

# 创建 figure
fig = go.Figure()

# 绘制 CM0/CM1 曲线
for model in ['CM0', 'CM1']:
    day_curve = np.linspace(days_points_base[0], days_points_base[-1], 100)
    rate_curve = np.interp(day_curve, days_points_base, rates_base[model])
    
    fig.add_trace(go.Scatter(
        x=day_curve,
        y=rate_curve,
        mode='lines',
        name=f'{model} 预测曲线'
    ))
    
    # 数据点
    fig.add_trace(go.Scatter(
        x=days_points_base,
        y=rates_base[model],
        mode='markers',
        marker=dict(size=10),
        name=f'{model} 数据点'
    ))

# 绘制 CM2 贝叶斯曲线
day_curve_cm2 = np.linspace(days_cm2[0], days_cm2[-1], 100)
rate_curve_cm2 = np.interp(day_curve_cm2, days_cm2, rates_cm2)

fig.add_trace(go.Scatter(
    x=day_curve_cm2,
    y=rate_curve_cm2,
    mode='lines',
    name='CM2 贝叶斯预测曲线',
    line=dict(color='red', dash='dash')
))

# CM2 数据点
fig.add_trace(go.Scatter(
    x=days_cm2,
    y=rates_cm2,
    mode='markers',
    marker=dict(size=10, color='red'),
    name='CM2 数据点/预测'
))

# 布局
fig.update_layout(
    title='CM0/CM1/CM2 锁单率随天数变化（CM2 贝叶斯预测）',
    xaxis_title='天数',
    yaxis_title='锁单率 (%)',
    yaxis=dict(range=[0, 50]),
    template='plotly_white'
)

fig.show()
