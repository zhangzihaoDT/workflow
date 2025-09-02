import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ====== 模拟多维度销量数据 ======
def mock_sales_data():
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", "2025-01-15")

    regions = ["华东", "华南", "西南"]
    provinces = {"华东": ["上海", "江苏"], "华南": ["广东", "广西"], "西南": ["四川", "重庆"]}
    channels = ["直营", "经销商", "电商"]

    data = []
    for date in dates:
        for region in regions:
            for prov in provinces[region]:
                for ch in channels:
                    order_vol = np.random.randint(50, 200)
                    lock_vol = np.random.randint(30, min(150, order_vol))  # 锁单量不超过订单量
                    data.append({
                        "date": date,
                        "region": region,
                        "province": prov,
                        "channel": ch,
                        "order_volume": order_vol,  # 订单量（领先指标）
                        "small_order_volume": np.random.randint(10, order_vol//2),  # 小订量
                        "lock_volume": lock_vol,  # 锁单量（确认指标）
                        "avg_price": np.random.randint(15, 30) * 1e4,  # 成交均价（价值指标）
                        "refund_volume": np.random.randint(0, min(20, lock_vol//3)),  # 退订量（风险指标）
                    })
    return pd.DataFrame(data)

df = mock_sales_data()

# ====== 趋势绘图函数 ======
def plot_trends(region, province, channel, metric):
    dff = df.copy()
    if region != "全部":
        dff = dff[dff["region"] == region]
    if province != "全部":
        dff = dff[dff["province"] == province]
    if channel != "全部":
        dff = dff[dff["channel"] == channel]

    grouped = dff.groupby("date")[metric].sum().reset_index()
    
    # 计算边际变化率
    grouped['change_rate'] = grouped[metric].pct_change() * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grouped["date"], 
        y=grouped[metric], 
        mode='lines+markers',
        name=metric,
        line=dict(width=2),
        marker=dict(size=6),
        hovertemplate='日期: %{x}<br>数值: %{y}<br>变化率: %{customdata:.1f}%<extra></extra>',
        customdata=grouped['change_rate']
    ))
    
    # 获取指标中文名称
    metric_names = {
        "order_volume": "订单量",
        "small_order_volume": "小订量", 
        "lock_volume": "锁单量",
        "avg_price": "成交均价",
        "refund_volume": "退订量"
    }
    
    fig.update_layout(
        title=f"{metric_names.get(metric, metric)} 趋势 - {region}/{province}/{channel}",
        xaxis_title="日期",
        yaxis_title=metric_names.get(metric, metric),
        width=700,
        height=350,
        showlegend=False,
        hovermode='x unified'
    )
    return fig

# ====== 异动监测函数 ======
def detect_anomaly(region, province, channel, metric, threshold=1.5):
    dff = df.copy()
    if region != "全部":
        dff = dff[dff["region"] == region]
    if province != "全部":
        dff = dff[dff["province"] == province]
    if channel != "全部":
        dff = dff[dff["channel"] == channel]

    grouped = dff.groupby("date")[metric].sum().reset_index()

    series = grouped[metric]
    mean, std = series.mean(), series.std()
    anomalies = grouped[series > mean + threshold * std]
    if anomalies.empty:
        return "暂无显著异常"
    return anomalies.to_string(index=False)

# ====== Gradio 界面 ======
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("## 🚗 汽车销量监测小程序（多维度选择版）\n核心逻辑：领先（订单）- 确认（锁单）- 价值（均价）- 风险（退订）")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                region_choice = gr.Dropdown(["全部", "华东", "华南", "西南"], label="区域选择", value="全部")
                province_choice = gr.Dropdown(["全部"], label="省份选择", value="全部")
                channel_choice = gr.Dropdown(["全部", "直营", "经销商", "电商"], label="渠道选择", value="全部")

            metric_choice = gr.Radio(
                ["order_volume", "small_order_volume", "lock_volume", "avg_price", "refund_volume"],
                label="选择监测指标",
                value="lock_volume"
            )
            plot_out = gr.Plot(label="📈 趋势图")
            anomaly_out = gr.Textbox(label="📊 异动监测结果", interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("### 📊 指标体系说明")
            gr.Markdown("""
            **🔸 领先指标**
            - **订单量**: 客户提交的购车订单总数
            - **小订量**: 支付定金但未完全确认的订单
            
            **🔸 确认指标** 
            - **锁单量**: 已确认成交的订单量（核心指标）
            
            **🔸 价值指标**
            - **成交均价**: 实际成交价格均值
            
            **🔸 风险指标**
            - **退订量**: 用户取消订单的数量
            """)
            threshold_slider = gr.Slider(0.5, 3.0, value=1.5, step=0.1, label="异常检测阈值 (倍数 std)")

    # ====== 联动逻辑（省份跟随区域变化） ======
    def update_province(region):
        if region == "全部":
            return gr.Dropdown.update(choices=["全部"], value="全部")
        provs = ["全部"] + sorted(df[df["region"] == region]["province"].unique().tolist())
        return gr.Dropdown.update(choices=provs, value="全部")

    region_choice.change(fn=update_province, inputs=region_choice, outputs=province_choice)

    # ====== 交互逻辑 ======
    inputs = [region_choice, province_choice, channel_choice, metric_choice]
    region_choice.change(fn=plot_trends, inputs=inputs, outputs=plot_out)
    province_choice.change(fn=plot_trends, inputs=inputs, outputs=plot_out)
    channel_choice.change(fn=plot_trends, inputs=inputs, outputs=plot_out)
    metric_choice.change(fn=plot_trends, inputs=inputs, outputs=plot_out)

    region_choice.change(fn=detect_anomaly, inputs=inputs+[threshold_slider], outputs=anomaly_out)
    province_choice.change(fn=detect_anomaly, inputs=inputs+[threshold_slider], outputs=anomaly_out)
    channel_choice.change(fn=detect_anomaly, inputs=inputs+[threshold_slider], outputs=anomaly_out)
    metric_choice.change(fn=detect_anomaly, inputs=inputs+[threshold_slider], outputs=anomaly_out)
    threshold_slider.change(fn=detect_anomaly, inputs=inputs+[threshold_slider], outputs=anomaly_out)

# 启动
demo.launch()
