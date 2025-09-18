import pandas as pd
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 数据路径
DATA_PATH = "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"

def load_and_analyze_data():
    """加载数据并进行分析"""
    try:
        # 加载数据
        df = pd.read_parquet(DATA_PATH)
        
        # 确保store_create_date是datetime类型
        df['store_create_date'] = pd.to_datetime(df['store_create_date'])
        
        # 提取年份
        df['year'] = df['store_create_date'].dt.year
        
        return df
    except Exception as e:
        print(f"数据加载错误: {e}")
        return None

def determine_store_status_unified(store_create_date, last_order_time, analysis_date, threshold_days=30):
    """
    统一的门店状态判断函数
    
    参数:
    - store_create_date: 门店创建日期
    - last_order_time: 最后订单时间
    - analysis_date: 分析日期（观察日期）
    - threshold_days: 判断闭店的天数阈值，默认30天
    
    返回:
    - '未开业': 门店创建日期 > 观察日期
    - '疑似闭店': 已开业但最后订单时间超过threshold_days天
    - '正常营业': 已开业且最后订单时间在threshold_days天内
    """
    # 确保日期格式正确
    store_create_date = pd.to_datetime(store_create_date)
    last_order_time = pd.to_datetime(last_order_time)
    analysis_date = pd.to_datetime(analysis_date)
    
    # 判断是否已开业
    is_opened = store_create_date <= analysis_date
    
    if not is_opened:
        return '未开业'
    
    # 计算距今天数
    days_since_last_order = (analysis_date - last_order_time).days
    
    if days_since_last_order > threshold_days:
        return '疑似闭店'
    else:
        return '正常营业'

def apply_store_status_analysis(df, analysis_date, last_order_time_column='Order_Create_Time'):
    """
    应用门店状态分析的通用函数
    
    参数:
    - df: 包含门店信息的DataFrame
    - analysis_date: 分析日期
    - last_order_time_column: 最后订单时间列名，默认为'Order_Create_Time'
    
    返回:
    - 添加了门店状态相关列的DataFrame
    """
    # 确保必要的列存在
    if 'store_create_date' not in df.columns:
        raise ValueError("DataFrame中缺少'store_create_date'列")
    if last_order_time_column not in df.columns:
        raise ValueError(f"DataFrame中缺少'{last_order_time_column}'列")
    
    # 确保日期格式正确
    df['store_create_date'] = pd.to_datetime(df['store_create_date'])
    df[last_order_time_column] = pd.to_datetime(df[last_order_time_column])
    
    # 计算距今天数和是否已开业
    df['距今天数'] = (analysis_date - df[last_order_time_column]).dt.days
    df['是否已开业'] = df['store_create_date'] <= analysis_date
    
    # 应用统一的门店状态判断
    df['门店状态'] = df.apply(
        lambda row: determine_store_status_unified(
            row['store_create_date'], 
            row[last_order_time_column], 
            analysis_date
        ), 
        axis=1
    )
    
    return df

def analyze_store_count_by_year(df, observation_date=None):
    """按年份统计门店数量，包含开闭店状态分析"""
    if df is None:
        return None
    
    # 确保Order_Create_Time是datetime类型
    df['Order_Create_Time'] = pd.to_datetime(df['Order_Create_Time'])
    
    # 计算每个门店的最大订单创建时间和年份
    store_info = df.groupby('Store Name').agg({
        'Order_Create_Time': 'max',
        'year': 'max',  # 门店最后活跃的年份
        'store_create_date': 'first'
    }).reset_index()
    
    # 确定分析基准日期
    if observation_date is not None:
        analysis_date = pd.to_datetime(observation_date)
    else:
        analysis_date = df['Order_Create_Time'].max()
    
    # 确保门店创建日期是datetime类型
    store_info['store_create_date'] = pd.to_datetime(store_info['store_create_date'])
    
    # 使用统一的门店状态判断函数
    store_info = apply_store_status_analysis(store_info, analysis_date, 'Order_Create_Time')
    
    # 按年份和门店状态统计
    yearly_status_stats = store_info.groupby(['year', '门店状态'], observed=False).size().reset_index(name='门店数量')
    
    # 创建透视表，便于展示
    yearly_pivot = yearly_status_stats.pivot(index='year', columns='门店状态', values='门店数量').fillna(0)
    yearly_pivot = yearly_pivot.reset_index()
    yearly_pivot.columns.name = None
    
    # 计算总数
    yearly_pivot['门店总数'] = yearly_pivot.get('正常营业', 0) + yearly_pivot.get('疑似闭店', 0)
    
    # 重命名列
    yearly_pivot = yearly_pivot.rename(columns={'year': '年份'})
    
    return yearly_pivot, yearly_status_stats

def analyze_by_parent_region(df, observation_date=None):
    """按Parent Region Name分组统计，包含门店状态分析"""
    if df is None:
        return None, None
    
    # 确保Order_Create_Time是datetime类型
    df['Order_Create_Time'] = pd.to_datetime(df['Order_Create_Time'])
    
    # 如果没有指定观察日期，使用数据中的最大日期
    if observation_date is None:
        analysis_date = df['Order_Create_Time'].max()
    else:
        analysis_date = pd.to_datetime(observation_date)
    
    # 获取每个门店的基本信息和最后订单时间
    store_info = df.groupby('Store Name').agg({
        'Order_Create_Time': 'max',
        'Parent Region Name': 'first',  # 门店的区域信息
        'store_create_date': 'first'
    }).reset_index()
    
    # 确保门店创建日期是datetime类型
    store_info['store_create_date'] = pd.to_datetime(store_info['store_create_date'])
    
    # 使用统一的门店状态判断函数
    store_info = apply_store_status_analysis(store_info, analysis_date, 'Order_Create_Time')
    
    # 计算每个区域的锁单数（Lock_Time不为空的订单数）
    if 'Lock_Time' in df.columns:
        region_lock_stats = df[df['Lock_Time'].notna()].groupby('Parent Region Name').size().reset_index(name='锁单数')
    else:
        region_lock_stats = pd.DataFrame(columns=['Parent Region Name', '锁单数'])
    
    # 按区域和门店状态统计
    region_status_stats = store_info.groupby(['Parent Region Name', '门店状态'], observed=False).size().reset_index(name='门店数量')
    
    # 创建透视表，便于展示
    region_pivot = region_status_stats.pivot(index='Parent Region Name', columns='门店状态', values='门店数量').fillna(0)
    region_pivot = region_pivot.reset_index()
    region_pivot.columns.name = None
    
    # 确保包含所有状态列
    if '正常营业' not in region_pivot.columns:
        region_pivot['正常营业'] = 0
    if '疑似闭店' not in region_pivot.columns:
        region_pivot['疑似闭店'] = 0
    
    # 合并锁单数据
    region_pivot = region_pivot.merge(region_lock_stats, on='Parent Region Name', how='left')
    region_pivot['锁单数'] = region_pivot['锁单数'].fillna(0).astype(int)
    
    # 重新排列列顺序
    region_pivot = region_pivot[['Parent Region Name', '正常营业', '疑似闭店', '锁单数']]
    region_pivot = region_pivot.rename(columns={'Parent Region Name': '区域名称'})
    
    # 按正常营业列降序排序
    region_pivot = region_pivot.sort_values('正常营业', ascending=False).reset_index(drop=True)
    
    return region_pivot, region_status_stats

def analyze_by_store_city(df, observation_date=None):
    """按Store City分组统计，包含门店状态分析"""
    if df is None:
        return None, None
    
    # 确保Order_Create_Time是datetime类型
    df['Order_Create_Time'] = pd.to_datetime(df['Order_Create_Time'])
    
    # 如果没有指定观察日期，使用数据中的最大日期
    if observation_date is None:
        analysis_date = df['Order_Create_Time'].max()
    else:
        analysis_date = pd.to_datetime(observation_date)
    
    # 获取每个门店的基本信息和最后订单时间
    store_info = df.groupby('Store Name').agg({
        'Order_Create_Time': 'max',
        'Store City': 'first',  # 门店的城市信息
        'store_create_date': 'first'
    }).reset_index()
    
    # 使用统一的门店状态判断函数
    store_info = apply_store_status_analysis(store_info, analysis_date, 'Order_Create_Time')
    
    # 计算每个城市的锁单数（Lock_Time不为空的订单数）
    if 'Lock_Time' in df.columns:
        city_lock_stats = df[df['Lock_Time'].notna()].groupby('Store City').size().reset_index(name='锁单数')
    else:
        city_lock_stats = pd.DataFrame(columns=['Store City', '锁单数'])
    
    # 按城市和门店状态统计
    city_status_stats = store_info.groupby(['Store City', '门店状态'], observed=False).size().reset_index(name='门店数量')
    
    # 创建透视表，便于展示
    city_pivot = city_status_stats.pivot(index='Store City', columns='门店状态', values='门店数量').fillna(0)
    city_pivot = city_pivot.reset_index()
    city_pivot.columns.name = None
    
    # 确保包含所有状态列
    if '正常营业' not in city_pivot.columns:
        city_pivot['正常营业'] = 0
    if '疑似闭店' not in city_pivot.columns:
        city_pivot['疑似闭店'] = 0
    if '未开业' not in city_pivot.columns:
        city_pivot['未开业'] = 0
    
    # 合并锁单数据
    city_pivot = city_pivot.merge(city_lock_stats, on='Store City', how='left')
    city_pivot['锁单数'] = city_pivot['锁单数'].fillna(0).astype(int)
    
    # 重新排列列顺序
    city_pivot = city_pivot[['Store City', '正常营业', '疑似闭店', '未开业', '锁单数']]
    city_pivot = city_pivot.rename(columns={'Store City': '城市名称'})
    
    # 按正常营业列降序排序
    city_pivot = city_pivot.sort_values('正常营业', ascending=False).reset_index(drop=True)
    
    return city_pivot, city_status_stats

def analyze_store_status(df, observation_date=None):
    """分析门店开闭店情况"""
    if df is None:
        return None, None
    
    # 确保Order_Create_Time是datetime类型
    df['Order_Create_Time'] = pd.to_datetime(df['Order_Create_Time'])
    
    # 计算每个门店的最大订单创建时间
    store_last_order = df.groupby('Store Name')['Order_Create_Time'].max().reset_index()
    store_last_order.columns = ['门店名称', '最后订单时间']
    
    # 确定分析基准日期
    if observation_date is not None:
        analysis_date = pd.to_datetime(observation_date)
    else:
        analysis_date = df['Order_Create_Time'].max()
    
    # 添加门店的其他信息
    store_info = df.groupby('Store Name').agg({
        'Store City': 'first',
        'Parent Region Name': 'first',
        'store_create_date': 'first',
        'Order Number': 'count'
    }).reset_index()
    store_info.columns = ['门店名称', '门店城市', '父级区域名称', '门店创建日期', '总订单数']
    
    # 计算每个门店的锁单数（Lock_Time不为空的订单数）
    if 'Lock_Time' in df.columns:
        store_lock_stats = df[df['Lock_Time'].notna()].groupby('Store Name').size().reset_index(name='锁单数')
        store_lock_stats.columns = ['门店名称', '锁单数']
        
        # 计算每个门店的最大锁单时间
        store_max_lock_time = df[df['Lock_Time'].notna()].groupby('Store Name')['Lock_Time'].max().reset_index()
        store_max_lock_time.columns = ['门店名称', '最大锁单时间']
    else:
        store_lock_stats = pd.DataFrame(columns=['门店名称', '锁单数'])
        store_max_lock_time = pd.DataFrame(columns=['门店名称', '最大锁单时间'])
    
    # 合并信息
    store_status = store_last_order.merge(store_info, on='门店名称', how='left')
    store_status = store_status.merge(store_lock_stats, on='门店名称', how='left')
    store_status = store_status.merge(store_max_lock_time, on='门店名称', how='left')
    store_status['锁单数'] = store_status['锁单数'].fillna(0).astype(int)
    
    # 确保门店创建日期是datetime类型
    store_status['门店创建日期'] = pd.to_datetime(store_status['门店创建日期'])
    
    # 使用统一的门店状态判断函数
    # 计算距今天数和是否已开业（为了保持兼容性）
    store_status['距今天数'] = (analysis_date - store_status['最后订单时间']).dt.days
    store_status['是否已开业'] = store_status['门店创建日期'] <= analysis_date
    
    # 应用统一的门店状态判断
    store_status['门店状态'] = store_status.apply(
        lambda row: determine_store_status_unified(
            row['门店创建日期'], 
            row['最后订单时间'], 
            analysis_date
        ), 
        axis=1
    )
    
    # 计算门店营业天数
    store_status['营业天数'] = (store_status['最后订单时间'] - pd.to_datetime(store_status['门店创建日期'])).dt.days
    
    # 重新排列列的顺序
    store_status = store_status[['门店名称', '门店城市', '父级区域名称', '门店创建日期', 
                               '最后订单时间', '距今天数', '营业天数', '总订单数', '锁单数', '最大锁单时间', '门店状态']]
    
    # 统计汇总
    status_summary = store_status['门店状态'].value_counts().reset_index()
    status_summary.columns = ['门店状态', '门店数量']
    
    return store_status, status_summary

def create_yearly_chart(yearly_status_stats):
    """创建年度统计图表，显示门店开闭店状态"""
    if yearly_status_stats is None or yearly_status_stats.empty:
        return None
    
    # 创建堆叠柱状图显示每年的门店状态分布
    fig = px.bar(yearly_status_stats, x='year', y='门店数量', color='门店状态',
                 title='不同年份门店总数统计（按开闭店状态分类）',
                 labels={'year': '年份', '门店数量': '门店数量', '门店状态': '门店状态'},
                 color_discrete_map={'正常营业': '#2E8B57', '疑似闭店': '#DC143C', '未开业': '#FFA500'})
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_region_chart(region_status_stats):
    """创建按区域分组的图表，显示门店状态分类"""
    if region_status_stats is None or region_status_stats.empty:
        return None
    
    # 创建堆叠柱状图显示每个区域的门店状态分布
    fig = px.bar(region_status_stats, x='Parent Region Name', y='门店数量', color='门店状态',
                 title='按区域分组的门店状态统计',
                 labels={'Parent Region Name': '区域名称', '门店数量': '门店数量', '门店状态': '门店状态'},
                 color_discrete_map={'正常营业': '#2E8B57', '疑似闭店': '#DC143C', '未开业': '#FFA500'})
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_city_chart(city_status_stats):
    """创建按城市分组的图表，显示门店状态分类"""
    if city_status_stats is None or city_status_stats.empty:
        return None
    
    # 由于城市可能很多，我们只显示前20个城市（按总门店数排序）
    city_totals = city_status_stats.groupby('Store City')['门店数量'].sum().nlargest(20)
    top_cities = city_totals.index
    filtered_city_stats = city_status_stats[city_status_stats['Store City'].isin(top_cities)]
    
    # 创建堆叠柱状图显示每个城市的门店状态分布
    fig = px.bar(filtered_city_stats, x='Store City', y='门店数量', color='门店状态',
                 title='按城市分组的门店状态统计（前20个城市）',
                 labels={'Store City': '城市名称', '门店数量': '门店数量', '门店状态': '门店状态'},
                 color_discrete_map={'正常营业': '#2E8B57', '疑似闭店': '#DC143C', '未开业': '#FFA500'})
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_store_status_chart(status_summary):
    """创建门店状态统计图表"""
    if status_summary is None or status_summary.empty:
        return None
    
    # 饼图显示门店状态分布
    fig = px.pie(status_summary, values='门店数量', names='门店状态',
                 title='门店开闭店状态分布',
                 color_discrete_map={'正常营业': '#2E8B57', '疑似闭店': '#DC143C', '未开业': '#FFA500'})
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_store_timeline_chart(store_status):
    """创建门店时间线图表"""
    if store_status is None or store_status.empty:
        return None
    
    # 按城市分组显示门店状态
    city_status = store_status.groupby(['门店城市', '门店状态'], observed=False).size().reset_index(name='门店数量')
    
    fig = px.bar(city_status, x='门店城市', y='门店数量', color='门店状态',
                 title='各城市门店开闭店状态分布',
                 labels={'门店城市': '城市', '门店数量': '门店数量', '门店状态': '门店状态'},
                 color_discrete_map={'正常营业': '#2E8B57', '疑似闭店': '#DC143C'})
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_days_since_histogram(store_status):
    """创建距今时间的门店数频数直方图"""
    if store_status is None or store_status.empty:
        return None
    
    import numpy as np
    
    # 定义自定义区间
    bins = [0, 1, 7, 30, 60, 180, 365, float('inf')]
    labels = ['0-1天', '1-7天', '7-30天', '30-60天', '60-180天', '180-365天', '365天以上']
    
    # 对距今天数进行分组
    store_status_copy = store_status.copy()
    store_status_copy['天数区间'] = pd.cut(store_status_copy['距今天数'], 
                                      bins=bins, 
                                      labels=labels, 
                                      right=False)
    
    # 统计每个区间的门店数量
    interval_counts = store_status_copy['天数区间'].value_counts().sort_index()
    
    # 创建柱状图
    fig = px.bar(x=interval_counts.index, y=interval_counts.values,
                 title='门店最后订单距今天数分布直方图',
                 labels={'x': '距今天数区间', 'y': '门店数量'},
                 color_discrete_sequence=['#4472C4'])
    
    # 添加30天分界线（在第3个柱子处）
    fig.add_vline(x=2.5, line_dash="dash", line_color="red", 
                  annotation_text="30天分界线", annotation_position="top")
    
    # 更新布局
    fig.update_layout(
        xaxis_title="距今天数区间",
        yaxis_title="门店数量",
        showlegend=False,
        xaxis_tickangle=-45
    )
    
    return fig

def main_analysis(observation_date_str=None, lock_start_date_str=None, lock_end_date_str=None):
    """主分析函数"""
    # 加载数据
    df = load_and_analyze_data()
    
    if df is None:
        return "数据加载失败", None, None, None, None, None, None, None, None, None, None
    
    # 处理观察日期输入
    observation_date = None
    if observation_date_str and observation_date_str.strip():
        try:
            observation_date = pd.to_datetime(observation_date_str.strip())
        except:
            observation_date = None
    
    # 处理Lock_Time范围筛选
    if lock_start_date_str and lock_start_date_str.strip() and lock_end_date_str and lock_end_date_str.strip():
        try:
            lock_start_date = pd.to_datetime(lock_start_date_str.strip())
            lock_end_date = pd.to_datetime(lock_end_date_str.strip())
            
            # 应用Lock_Time范围筛选
            if 'Lock_Time' in df.columns:
                df['Lock_Time'] = pd.to_datetime(df['Lock_Time'], errors='coerce')
                lock_mask = (df['Lock_Time'] >= lock_start_date) & (df['Lock_Time'] <= lock_end_date)
                df = df[lock_mask]
        except:
            pass  # 如果日期格式错误，忽略筛选
    
    # 进行各种分析
    yearly_stats, yearly_status_stats = analyze_store_count_by_year(df, observation_date)
    region_stats, region_status_stats = analyze_by_parent_region(df, observation_date)
    city_stats, city_status_stats = analyze_by_store_city(df, observation_date)
    store_status, status_summary = analyze_store_status(df, observation_date)
    
    # 创建图表
    yearly_chart = create_yearly_chart(yearly_status_stats)
    region_chart = create_region_chart(region_status_stats)
    city_chart = create_city_chart(city_status_stats)
    status_chart = create_store_status_chart(status_summary)
    timeline_chart = create_store_timeline_chart(store_status)
    histogram_chart = create_days_since_histogram(store_status)
    
    # 数据概览信息
    total_stores = df['Store Name'].nunique()
    total_years = df['year'].nunique()
    year_range = f"{df['year'].min()} - {df['year'].max()}"
    
    # 门店状态统计
    active_stores = len(store_status[store_status['门店状态'] == '正常营业'])
    closed_stores = len(store_status[store_status['门店状态'] == '疑似闭店'])
    
    # Lock_Time筛选信息
    lock_filter_info = ""
    if lock_start_date_str and lock_start_date_str.strip() and lock_end_date_str and lock_end_date_str.strip():
        if 'Lock_Time' in df.columns:
            lock_count = df['Lock_Time'].notna().sum()
            lock_filter_info = f"""
    锁单时间筛选：
    - 筛选范围: {lock_start_date_str} 至 {lock_end_date_str}
    - 筛选后数据量: {len(df)} 条记录
    - 包含锁单时间的记录: {lock_count} 条
    """
    
    summary = f"""
    数据概览：
    - 总门店数量: {total_stores}
    - 年份范围: {year_range}
    - 涵盖年份数: {total_years}
    - 父级区域数量: {df['Parent Region Name'].nunique()}
    - 城市数量: {df['Store City'].nunique()}{lock_filter_info}
    
    门店状态分析：
    - 正常营业门店: {active_stores} 家
    - 疑似闭店门店: {closed_stores} 家
    - 闭店率: {closed_stores/total_stores*100:.1f}%
    """
    
    return (summary, yearly_chart, region_chart, city_chart, status_chart, timeline_chart, histogram_chart,
            yearly_stats, region_stats, city_stats, store_status, status_summary)

# 创建Gradio界面
def create_interface():
    with gr.Blocks(title="门店分析统计", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 门店分析统计系统")
        gr.Markdown("分析不同年份下的门店数量，并按父级区域和城市进行分组统计，同时分析门店开闭店情况")
    
        with gr.Row():    
            with gr.Column(scale=1):
                observation_date_input = gr.Textbox(
                    label="观察日期 (可选，格式: YYYY-MM-DD)", 
                    placeholder="留空则使用数据中的最新日期",
                    value=""
                )

                with gr.Group():
                    gr.Markdown("### 锁单时间范围筛选 (可选)")
                    with gr.Row():
                        lock_start_date_input = gr.Textbox(
                            label="锁单开始日期", 
                            placeholder="YYYY-MM-DD (可选)",
                            value=""
                        )
                        lock_end_date_input = gr.Textbox(
                            label="锁单结束日期", 
                            placeholder="YYYY-MM-DD (可选)",
                            value=""
                        )

                analyze_btn = gr.Button("开始分析", variant="primary", size="lg")
                summary_output = gr.Textbox(label="数据概览", lines=12, interactive=False)            

            with gr.Column(scale=4):
            
                with gr.Tabs():
                    with gr.TabItem("年度统计"):
                        yearly_plot = gr.Plot(label="年度门店总数统计")
                        yearly_table = gr.Dataframe(label="年度统计数据表")
                    
                    with gr.TabItem("按区域统计"):
                        region_plot = gr.Plot(label="按父级区域分组统计")
                        region_table = gr.Dataframe(label="区域统计数据表")
                    
                    with gr.TabItem("按城市统计"):
                        city_plot = gr.Plot(label="按城市分组统计（前20个城市）")
                        city_table = gr.Dataframe(label="城市统计数据表")
                    
                    with gr.TabItem("门店状态分析"):
                        with gr.Row():
                            status_pie_chart = gr.Plot(label="门店状态分布")
                            timeline_chart = gr.Plot(label="各城市门店状态分布")
                        
                        with gr.Row():
                            histogram_chart = gr.Plot(label="门店最后订单距今天数分布直方图")
                        
                        with gr.Row():
                            store_status_table = gr.Dataframe(label="门店详细状态表")
                        
                        with gr.Row():
                            status_summary_table = gr.Dataframe(label="门店状态汇总")
            
            # 绑定分析函数
            analyze_btn.click(
                fn=main_analysis,
                inputs=[observation_date_input, lock_start_date_input, lock_end_date_input],
                outputs=[summary_output, yearly_plot, region_plot, city_plot, 
                        status_pie_chart, timeline_chart, histogram_chart,
                        yearly_table, region_table, city_table, 
                        store_status_table, status_summary_table]
            )
        
        return demo

if __name__ == "__main__":
    # 创建并启动界面
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7865, share=False)