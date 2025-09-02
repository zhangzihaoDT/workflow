import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime

# 从 order_trend_monitor 模块导入数据处理函数
from order_trend_monitor import filter_and_aggregate_data, detect_data_anomaly, initialize_database, get_db_connection
import os

# ====== 加载预处理的聚合数据 ======
def load_processed_data():
    """从工作区文件加载预处理的聚合数据"""
    processed_file = "/Users/zihao_/Documents/github/W35_workflow/processed_order_data.parquet"
    
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"聚合数据文件不存在: {processed_file}。请先运行 order_trend_monitor.py 生成聚合数据。")
    
    # 读取预处理的数据
    df = pd.read_parquet(processed_file)
    
    # 初始化数据库并将数据加载到 DuckDB
    initialize_database()
    conn = get_db_connection()
    
    # 清空并重新加载数据到 processed_order_data 表
    conn.execute("DELETE FROM processed_order_data")
    conn.register('temp_df', df)
    conn.execute("""
        INSERT INTO processed_order_data 
        SELECT * FROM temp_df
    """)
    
    return df

df = load_processed_data()

# ====== 趋势绘图函数 ======
def plot_trends(region, province, channel, metric, start_date, end_date):
    # 验证日期格式
    start_date = start_date.strip() if start_date and start_date.strip() else None
    end_date = end_date.strip() if end_date and end_date.strip() else None
    
    # 使用导入的数据处理函数
    grouped, actual_metric = filter_and_aggregate_data(df, region, province, channel, metric, start_date, end_date)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grouped["date"], 
        y=grouped[actual_metric], 
        mode='lines+markers',
        name=actual_metric,
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
        # width=700,
        height=450,
        showlegend=False,
        hovermode='x unified'
    )
    return fig

# ====== 异动监测函数 ======
def detect_anomaly(region, province, channel, metric, threshold=1.5, start_date=None, end_date=None):
    # 验证日期格式，处理可能的 float 类型输入
    if isinstance(start_date, str):
        start_date = start_date.strip() if start_date and start_date.strip() else None
    else:
        start_date = None
        
    if isinstance(end_date, str):
        end_date = end_date.strip() if end_date and end_date.strip() else None
    else:
        end_date = None
    
    # 使用导入的异动检测函数
    return detect_data_anomaly(df, region, province, channel, metric, threshold, start_date, end_date)

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
            
            # 添加日期范围选择器
            with gr.Row():
                start_date = gr.Textbox(
                    label="开始日期",
                    value="2024-01-01",
                    placeholder="YYYY-MM-DD",
                    scale=1
                )
                end_date = gr.Textbox(
                    label="结束日期",
                    value="2024-12-31",
                    placeholder="YYYY-MM-DD",
                    scale=1
                )
            
            plot_out = gr.Plot(label="📈 趋势图")
            
            # 异动检测按钮和结果
            with gr.Row():
                anomaly_btn = gr.Button("🔍 执行异动检测", variant="primary")
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
    inputs = [region_choice, province_choice, channel_choice, metric_choice, start_date, end_date]
    region_choice.change(fn=plot_trends, inputs=inputs, outputs=plot_out)
    province_choice.change(fn=plot_trends, inputs=inputs, outputs=plot_out)
    channel_choice.change(fn=plot_trends, inputs=inputs, outputs=plot_out)
    metric_choice.change(fn=plot_trends, inputs=inputs, outputs=plot_out)
    start_date.change(fn=plot_trends, inputs=inputs, outputs=plot_out)
    end_date.change(fn=plot_trends, inputs=inputs, outputs=plot_out)

    # 异动检测按钮点击事件
    anomaly_inputs = inputs + [threshold_slider]
    anomaly_btn.click(fn=detect_anomaly, inputs=anomaly_inputs, outputs=anomaly_out)

# 启动
demo.launch()
