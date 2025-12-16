# ... (imports remain the same)
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import statsmodels.api as sm  # For LOWESS

# 数据路径
DATA_PATH = "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"
COMBINED_HTML = "combined_lock_analysis.html"

def map_age_group(age, current_year):
    """根据年龄计算分组（低龄/中年/高龄）"""
    if pd.isna(age):
        return "未知"
    try:
        birth_year = current_year - int(age)
    except (ValueError, TypeError):
        return "未知"
        
    if birth_year >= 1990:
        return "低龄段"  # 00后(>=2000), 95后(>=1995), 90后(>=1990)
    elif birth_year >= 1980:
        return "中年段"  # 85后(>=1985), 80后(>=1980)
    else:
        return "高龄段"  # 75后, 70后, 70前 (<1980)

def analyze_age_group_trends(df=None, target_models=None, start_date_str="2023-10-12"):
    """
    分析年龄组锁单趋势
    如果传入 df，则直接使用；否则从 DATA_PATH 加载
    返回 fig 对象，而不是直接保存文件
    """
    if target_models is None:
        target_models = ['CM0', 'CM1', 'CM2']
        
    print(f"\n=== 开始执行年龄组锁单趋势分析 (车型: {target_models}, 起始: {start_date_str}) ===")
    
    if df is None:
        print("正在加载数据...")
        try:
            df = pd.read_parquet(DATA_PATH)
        except Exception as e:
            print(f"读取数据失败: {e}")
            return None

    # 1. 筛选数据
    # 筛选“车型分组”
    if '车型分组' in df.columns:
        df = df[df['车型分组'].isin(target_models)]
        print(f"筛选车型分组后数据量: {len(df)}")
    else:
        print("警告：未找到 '车型分组' 列，跳过车型筛选")

    # 筛选 Lock_Time
    if 'Lock_Time' not in df.columns:
        print("错误：缺少 'Lock_Time' 列")
        return None
        
    df = df.copy() # 避免 SettingWithCopyWarning
    df['Lock_Time'] = pd.to_datetime(df['Lock_Time'])
    start_date = pd.Timestamp(start_date_str)
    df = df[df['Lock_Time'] >= start_date]
    print(f"筛选时间范围后数据量: {len(df)}")
    
    # 2. 处理年龄分组
    if 'owner_age' not in df.columns:
        print("错误：缺少 'owner_age' 列")
        return None
        
    # 过滤年龄异常值 [16, 85]
    df['owner_age_num'] = pd.to_numeric(df['owner_age'], errors='coerce')
    df = df[(df['owner_age_num'] >= 16) & (df['owner_age_num'] <= 85)]
    print(f"筛选年龄[16, 85]后数据量: {len(df)}")
        
    current_year = datetime.now().year
    df['age_group'] = df['owner_age_num'].apply(lambda x: map_age_group(x, current_year))
    
    # 3. 按日聚合统计锁单数
    # 将 Lock_Time 转换为日期
    df['lock_date'] = df['Lock_Time'].dt.date
    
    # Pivot Table: 行=日期, 列=年龄组, 值=计数
    daily_counts = df.pivot_table(
        index='lock_date', 
        columns='age_group', 
        values='Lock_Time', 
        aggfunc='count', 
        fill_value=0
    )
    
    # 4. 计算占比 (行归一化)
    # 每一行的总和
    daily_totals = daily_counts.sum(axis=1)
    # 除以总和得到占比
    daily_ratios = daily_counts.div(daily_totals, axis=0)
    
    # 5. MA7 处理 (7日移动平均)
    daily_ratios_ma7 = daily_ratios.rolling(window=7).mean()
    
    # 清理数据：去掉最开始因为 rolling 产生的 NaN
    daily_ratios_ma7 = daily_ratios_ma7.dropna()
    
    # 重置索引以便绘图
    plot_data = daily_ratios_ma7.reset_index()
    
    # 转换回长格式 (Long Format) 以便 Plotly 使用
    plot_data_melted = plot_data.melt(
        id_vars=['lock_date'], 
        var_name='age_group', 
        value_name='ratio'
    )
    
    # 6. 绘制折线图
    # 定义特定排序的类别顺序
    age_order = ["低龄段", "中年段", "高龄段"]
    
    # 过滤掉未知组，避免干扰
    plot_data_melted = plot_data_melted[plot_data_melted['age_group'].isin(age_order)]
    
    # 颜色映射 (使用用户指定的颜色，补充第三个)
    color_map = {
        "低龄段": "#27AD00",  # 用户指定
        "中年段": "#005783",  # 用户指定
        "高龄段": "#F4A460"   # 补充颜色 (SandyBrown)
    }

    # 使用 graph_objects 创建图表，以便叠加 LOWESS 曲线
    fig = go.Figure()

    for age_group in age_order:
        # 筛选当前年龄段数据
        group_data = plot_data_melted[plot_data_melted['age_group'] == age_group].copy()
        if group_data.empty:
            continue
            
        color = color_map.get(age_group, '#000000')
        
        # 1. 绘制原始 MA7 折线 (设置透明度或线宽，作为背景)
        fig.add_trace(go.Scatter(
            x=group_data['lock_date'],
            y=group_data['ratio'],
            mode='lines',
            name=f'{age_group} (MA7)',
            line=dict(color=color, width=1.5),
            opacity=0.4, # 让原始线稍微淡一点，突出趋势线
            legendgroup=age_group
        ))
        
        # 2. 计算并绘制 LOWESS 趋势线
        # LOWESS 需要数值型的 x 轴
        group_data['date_ordinal'] = pd.to_datetime(group_data['lock_date']).map(datetime.toordinal)
        
        # lowess 返回 (y, x) 数组
        # frac 参数控制平滑度，值越大越平滑。默认 2/3 左右，这里取 0.1-0.3 适合观察中短期趋势
        lowess_result = sm.nonparametric.lowess(
            group_data['ratio'], 
            group_data['date_ordinal'], 
            frac=0.2
        )
        
        fig.add_trace(go.Scatter(
            x=[datetime.fromordinal(int(d)) for d in lowess_result[:, 0]],
            y=lowess_result[:, 1],
            mode='lines',
            name=f'{age_group} (Trend)',
            line=dict(color=color, width=3),
            legendgroup=age_group
        ))

    models_str = "/".join(target_models)
    fig.update_layout(
        title=f'各年龄段日锁单数占比趋势 (车型分组: {models_str}, MA7 + LOWESS Trend)',
        xaxis_title='日期',
        yaxis_title='锁单占比',
        yaxis_tickformat='.1%',
        plot_bgcolor='#FFFFFF',
        xaxis=dict(
            gridcolor='#ebedf0',
            zerolinecolor='#ebedf0',
            tickfont=dict(color='#7B848F'),
            title_font=dict(color='#7B848F')
        ),
        yaxis=dict(
            gridcolor='#ebedf0',
            zerolinecolor='#ebedf0',
            tickfont=dict(color='#7B848F'),
            title_font=dict(color='#7B848F')
        ),
        legend=dict(
            bordercolor='#7B848F',
            borderwidth=1,
            font=dict(color='#7B848F')
        ),
        height=380
    )
    
    return fig

def analyze_delivery_duration(df=None):
    """
    分析 Invoice_Upload_Time 和 Lock_Time 的时间差
    对比三组车型分组的趋势
    """
    print("\n=== 开始执行交付效率分析 (Invoice - Lock Time Gap) ===")
    
    if df is None:
        return None

    # 1. 基础筛选与计算
    required_cols = ['Lock_Time', 'Invoice_Upload_Time', '车型分组']
    for col in required_cols:
        if col not in df.columns:
            print(f"错误：缺少 '{col}' 列")
            return None
            
    df = df.copy()
    
    # 转换时间
    df['Lock_Time'] = pd.to_datetime(df['Lock_Time'])
    df['Invoice_Upload_Time'] = pd.to_datetime(df['Invoice_Upload_Time'])
    
    # 筛选 Lock_Time >= 2023-10-12
    start_date = pd.Timestamp("2023-10-12")
    df = df[df['Lock_Time'] >= start_date]
    
    # 计算时间差 (天)
    df['gap_days'] = (df['Invoice_Upload_Time'] - df['Lock_Time']).dt.total_seconds() / (24 * 3600)
    
    # 过滤无效数据 (可选: 过滤掉负数或极值，这里暂不过滤，保留真实情况)
    # df = df[df['gap_days'] >= 0] 
    
    df['lock_date'] = df['Lock_Time'].dt.date
    
    # 定义分组
    groups = {
        "CM0/CM1/CM2": ['CM0', 'CM1', 'CM2'],
        "DM0/CM1": ['DM0', 'CM1'],
        "LS9": ['LS9']
    }
    
    # 颜色映射 (与之前保持一致或自定义)
    color_map = {
        "CM0/CM1/CM2": "#27AD00", 
        "DM0/CM1": "#005783",
        "LS9": "#F4A460"
    }
    
    fig = go.Figure()
    
    for group_name, models in groups.items():
        # 筛选该组车型
        group_df = df[df['车型分组'].isin(models)]
        
        if group_df.empty:
            print(f"警告: 组 {group_name} 无数据")
            continue
            
        # 按日聚合计算平均时间差
        daily_gap = group_df.groupby('lock_date')['gap_days'].mean()
        
        # 计算 MA7
        daily_gap_ma7 = daily_gap.rolling(window=7).mean().dropna()
        
        if daily_gap_ma7.empty:
             continue
             
        # 绘制曲线
        fig.add_trace(go.Scatter(
            x=daily_gap_ma7.index,
            y=daily_gap_ma7.values,
            mode='lines',
            name=f'{group_name} (MA7)',
            line=dict(color=color_map.get(group_name, '#000000'), width=2)
        ))
        
    fig.update_layout(
        title='各车型分组 发票-锁单时间差 趋势对比 (MA7)',
        xaxis_title='锁单日期 (Lock Time)',
        yaxis_title='平均时间差 (天)',
        plot_bgcolor='#FFFFFF',
        xaxis=dict(
            gridcolor='#ebedf0',
            zerolinecolor='#ebedf0',
            tickfont=dict(color='#7B848F'),
            title_font=dict(color='#7B848F')
        ),
        yaxis=dict(
            gridcolor='#ebedf0',
            zerolinecolor='#ebedf0',
            tickfont=dict(color='#7B848F'),
            title_font=dict(color='#7B848F')
        ),
        legend=dict(
            bordercolor='#7B848F',
            borderwidth=1,
            font=dict(color='#7B848F')
        ),
        height=380
    )
    
    return fig

def analyze_delivery_histogram(df=None):
    """
    分析交付周期频数分布 (Histogram)
    筛选车型: CM0, CM1, CM2
    Bin: 7天
    Y轴: 交付率 (Bin Count / Total Locked)
    """
    print("\n=== 开始执行交付周期频数分布分析 (Histogram) ===")
    
    if df is None:
        return None
        
    target_models = ['CM0', 'CM1', 'CM2']
    
    # 1. 筛选车型
    if '车型分组' not in df.columns:
        print("错误：缺少 '车型分组' 列")
        return None
        
    df_sub = df[df['车型分组'].isin(target_models)].copy()
    
    if df_sub.empty:
        print("警告: 指定车型无数据")
        return None
        
    # 2. 转换时间
    if 'Lock_Time' not in df_sub.columns or 'Invoice_Upload_Time' not in df_sub.columns:
        print("错误：缺少时间列")
        return None
        
    df_sub['Lock_Time'] = pd.to_datetime(df_sub['Lock_Time'])
    df_sub['Invoice_Upload_Time'] = pd.to_datetime(df_sub['Invoice_Upload_Time'])
    
    # 3. 计算总锁单数 (分母)
    # 只要 Lock_Time 非空就算锁单
    total_locked = df_sub['Lock_Time'].notna().sum()
    
    if total_locked == 0:
        print("警告: 无锁单数据")
        return None
        
    print(f"CM0/CM1/CM2 总锁单数: {total_locked}")
    
    # 4. 计算交付周期 (分子来源)
    # 必须有 Lock_Time 和 Invoice_Upload_Time
    df_delivered = df_sub.dropna(subset=['Lock_Time', 'Invoice_Upload_Time']).copy()
    
    # 计算天数差
    df_delivered['gap_days'] = (df_delivered['Invoice_Upload_Time'] - df_delivered['Lock_Time']).dt.total_seconds() / (24 * 3600)
    
    # 5. 分箱 (Binning) - 7天为一组，大于98天的归为一组
    if df_delivered.empty:
        print("警告: 无已交付数据")
        return None
        
    # 定义 bins: 0, 7, 14, ..., 98, inf
    bins = list(range(0, 99, 7)) + [np.inf]
    
    # 定义 labels: [0,7), [7,14), ..., [91,98), 98+
    labels = [f"[{i},{i+7})" for i in range(0, 98, 7)] + ["98+"]
    
    # 使用 pd.cut 分箱
    df_delivered['gap_bin'] = pd.cut(df_delivered['gap_days'], bins=bins, labels=labels, right=False)
    
    # 统计每个 bin 的数量
    bin_counts = df_delivered['gap_bin'].value_counts().sort_index()
    
    # 6. 计算交付率
    bin_rates = bin_counts / total_locked
    
    # 计算累计交付率
    cumulative_rates = bin_rates.cumsum()
    
    # 准备绘图数据
    plot_data = pd.DataFrame({
        'bin_range': bin_rates.index.astype(str),
        'rate': bin_rates.values,
        'cumulative_rate': cumulative_rates.values,
        'count': bin_counts.values
    })
    
    # 7. 绘图 (使用 make_subplots 支持双轴)
    # 颜色: CM0/CM1/CM2 使用 #27AD00
    primary_color = "#27AD00"
    secondary_color = "#005783" # 使用深蓝色作为累计曲线，对比明显
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bar Chart (左轴: 频数占比)
    fig.add_trace(
        go.Bar(
            x=plot_data['bin_range'],
            y=plot_data['rate'],
            name="区间交付率",
            marker_color=primary_color,
            text=plot_data['rate'].apply(lambda x: f'{x:.1%}'),
            textposition='outside'
        ),
        secondary_y=False
    )
    
    # Line Chart (右轴: 累计占比)
    fig.add_trace(
        go.Scatter(
            x=plot_data['bin_range'],
            y=plot_data['cumulative_rate'],
            name="累计交付率",
            mode='lines+markers+text',
            line=dict(color=secondary_color, width=3),
            marker=dict(size=8),
            text=plot_data['cumulative_rate'].apply(lambda x: f'{x:.1%}'),
            textposition='top center',
            textfont=dict(color=secondary_color)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title='CM0/CM1/CM2 交付周期分布 & 累计交付率',
        plot_bgcolor='#FFFFFF',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # 设置 X 轴
    fig.update_xaxes(
        title_text="交付周期区间 (天)",
        gridcolor='#ebedf0',
        tickangle=-45,
        tickfont=dict(color='#7B848F'),
        title_font=dict(color='#7B848F')
    )
    
    # 设置左 Y 轴
    fig.update_yaxes(
        title_text="区间交付率 (占总锁单)",
        tickformat='.1%',
        gridcolor='#ebedf0',
        zerolinecolor='#ebedf0',
        tickfont=dict(color=primary_color),
        title_font=dict(color=primary_color),
        secondary_y=False,
        range=[0, max(plot_data['rate'].max() * 1.2, 0.1)] # 稍微留点空间给 text
    )
    
    # 设置右 Y 轴
    fig.update_yaxes(
        title_text="累计交付率",
        tickformat='.1%',
        showgrid=False, # 右轴不显示网格，避免乱
        tickfont=dict(color=secondary_color),
        title_font=dict(color=secondary_color),
        secondary_y=True,
        range=[0, 1.1] # 累计率最高是 1 (或者接近1)，留点空间
    )
    
    return fig

def analyze_undelivered_duration(df=None):
    """
    分析未交付订单的等待时长分布
    筛选: 
      - 车型: CM0, CM1, CM2
      - Lock_Time >= 2025-01-01
      - Invoice_Upload_Time is Null
    计算:
      - 等待时长 = 当前时间 - Lock_Time
    图表:
      - Histogram (Bin=7)
    """
    print("\n=== 开始执行未交付订单积压时长分析 (Undelivered Duration) ===")
    
    if df is None:
        return None
        
    target_models = ['CM0', 'CM1', 'CM2']
    start_date_str = "2025-01-01"
    
    # 1. 筛选车型
    if '车型分组' not in df.columns:
        print("错误：缺少 '车型分组' 列")
        return None
        
    df_sub = df[df['车型分组'].isin(target_models)].copy()
    
    # 2. 筛选时间 & 状态
    if 'Lock_Time' not in df_sub.columns or 'Invoice_Upload_Time' not in df_sub.columns:
        print("错误：缺少时间列")
        return None
        
    df_sub['Lock_Time'] = pd.to_datetime(df_sub['Lock_Time'])
    
    # 筛选 Lock_Time >= 2025-01-01
    start_date = pd.Timestamp(start_date_str)
    df_sub = df_sub[df_sub['Lock_Time'] >= start_date]
    
    # 筛选未交付 (Invoice_Upload_Time 为空)
    # 注意：NaT 也是空
    df_undelivered = df_sub[df_sub['Invoice_Upload_Time'].isna() | (df_sub['Invoice_Upload_Time'].astype(str) == 'NaT')].copy()
    
    total_undelivered = len(df_undelivered)
    print(f"CM0/CM1/CM2 (Lock >= {start_date_str}) 未交付总数: {total_undelivered}")
    
    if total_undelivered == 0:
        print("警告: 无符合条件的未交付数据")
        return None
        
    # 3. 计算等待时长
    # 使用当前时间 (datetime.now()) 作为截止点
    # 为了保持一致性，如果数据中有明显的"最新时间"，也可以用那个。
    # 这里默认使用脚本运行时间。
    current_time = datetime.now()
    
    df_undelivered['wait_days'] = (current_time - df_undelivered['Lock_Time']).dt.total_seconds() / (24 * 3600)
    
    # 4. 分箱 (Binning) - 7天为一组，大于98天的归为一组 (保持统一)
    # 定义 bins: 0, 7, 14, ..., 98, inf
    bins = list(range(0, 99, 7)) + [np.inf]
    labels = [f"[{i},{i+7})" for i in range(0, 98, 7)] + ["98+"]
    
    df_undelivered['wait_bin'] = pd.cut(df_undelivered['wait_days'], bins=bins, labels=labels, right=False)
    
    bin_counts = df_undelivered['wait_bin'].value_counts().sort_index()
    
    # 计算占比
    bin_rates = bin_counts / total_undelivered
    cumulative_rates = bin_rates.cumsum()
    
    # 5. 绘图
    plot_data = pd.DataFrame({
        'bin_range': bin_rates.index.astype(str),
        'rate': bin_rates.values,
        'cumulative_rate': cumulative_rates.values,
        'count': bin_counts.values
    })
    
    primary_color = "#27AD00"
    secondary_color = "#005783"
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bar Chart (左轴: 占比)
    fig.add_trace(
        go.Bar(
            x=plot_data['bin_range'],
            y=plot_data['rate'],
            name="区间占比",
            marker_color=primary_color,
            text=plot_data['rate'].apply(lambda x: f'{x:.1%}'),
            textposition='outside'
        ),
        secondary_y=False
    )
    
    # Line Chart (右轴: 累计占比)
    fig.add_trace(
        go.Scatter(
            x=plot_data['bin_range'],
            y=plot_data['cumulative_rate'],
            name="累计占比",
            mode='lines+markers+text',
            line=dict(color=secondary_color, width=3),
            marker=dict(size=8),
            text=plot_data['cumulative_rate'].apply(lambda x: f'{x:.1%}'),
            textposition='top center',
            textfont=dict(color=secondary_color)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title=f'CM0/CM1/CM2 未交付订单积压时长分布 (Lock >= {start_date_str})',
        plot_bgcolor='#FFFFFF',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(
        title_text="已等待时长 (天)",
        gridcolor='#ebedf0',
        tickangle=-45,
        tickfont=dict(color='#7B848F'),
        title_font=dict(color='#7B848F')
    )
    
    fig.update_yaxes(
        title_text="占比 (占未交付总数)",
        tickformat='.1%',
        gridcolor='#ebedf0',
        zerolinecolor='#ebedf0',
        tickfont=dict(color=primary_color),
        title_font=dict(color=primary_color),
        secondary_y=False,
        range=[0, max(plot_data['rate'].max() * 1.2, 0.1)]
    )
    
    fig.update_yaxes(
        title_text="累计占比",
        tickformat='.1%',
        showgrid=False,
        tickfont=dict(color=secondary_color),
        title_font=dict(color=secondary_color),
        secondary_y=True,
        range=[0, 1.1]
    )
    
    return fig

def analyze_combined_delivery_backlog(df=None):
    """
    组合分析 CM0/CM1/CM2 已交付与未交付订单在交付/等待时长上的分布
    Lock_Time >= 2025-01-01
    使用相同的 7 天分桶和 98+ 规则
    """
    print("\n=== 开始执行交付与未交付合并分布分析 (Combined Delivery & Backlog) ===")
    
    if df is None:
        return None
        
    target_models = ['CM0', 'CM1', 'CM2']
    start_date_str = "2025-01-01"
    
    if '车型分组' not in df.columns:
        print("错误：缺少 '车型分组' 列")
        return None
        
    if 'Lock_Time' not in df.columns or 'Invoice_Upload_Time' not in df.columns:
        print("错误：缺少时间列")
        return None
    
    df_sub = df[df['车型分组'].isin(target_models)].copy()
    if df_sub.empty:
        print("警告: 指定车型无数据")
        return None
    
    df_sub['Lock_Time'] = pd.to_datetime(df_sub['Lock_Time'])
    df_sub['Invoice_Upload_Time'] = pd.to_datetime(df_sub['Invoice_Upload_Time'])
    
    start_date = pd.Timestamp(start_date_str)
    df_sub = df_sub[df_sub['Lock_Time'] >= start_date]
    
    locked_total = df_sub['Lock_Time'].notna().sum()
    if locked_total == 0:
        print("警告: 无锁单数据")
        return None
    
    df_delivered = df_sub.dropna(subset=['Invoice_Upload_Time']).copy()
    df_undelivered = df_sub[df_sub['Invoice_Upload_Time'].isna()].copy()
    
    current_time = datetime.now()
    
    bins = list(range(0, 99, 7)) + [np.inf]
    labels = [f"[{i},{i+7})" for i in range(0, 98, 7)] + ["98+"]
    idx = pd.Index(labels, name='bin_range')
    
    if not df_delivered.empty:
        df_delivered['gap_days'] = (df_delivered['Invoice_Upload_Time'] - df_delivered['Lock_Time']).dt.total_seconds() / (24 * 3600)
        df_delivered['bin'] = pd.cut(df_delivered['gap_days'], bins=bins, labels=labels, right=False)
        delivered_counts = df_delivered['bin'].value_counts().sort_index()
    else:
        delivered_counts = pd.Series(0, index=idx)
    
    if not df_undelivered.empty:
        df_undelivered['wait_days'] = (current_time - df_undelivered['Lock_Time']).dt.total_seconds() / (24 * 3600)
        df_undelivered['bin'] = pd.cut(df_undelivered['wait_days'], bins=bins, labels=labels, right=False)
        undelivered_counts = df_undelivered['bin'].value_counts().sort_index()
    else:
        undelivered_counts = pd.Series(0, index=idx)
    
    delivered_counts = delivered_counts.reindex(idx, fill_value=0)
    undelivered_counts = undelivered_counts.reindex(idx, fill_value=0)
    total_counts = delivered_counts + undelivered_counts
    
    delivered_rate = delivered_counts / locked_total
    undelivered_rate = undelivered_counts / locked_total
    total_rate = total_counts / locked_total
    cumulative_total_rate = total_rate.cumsum()
    
    conversion_prob = delivered_counts / total_counts.replace(0, np.nan)
    delivered_total = delivered_counts.sum()
    backlog_total = undelivered_counts.sum()
    if backlog_total > 0:
        expected_converted_from_backlog = (undelivered_counts * conversion_prob).sum(skipna=True)
        backlog_conversion_rate = expected_converted_from_backlog / backlog_total
        print(f"预计当前未交付订单未来可转交付数量: {expected_converted_from_backlog:.1f} / {backlog_total} ({backlog_conversion_rate:.1%})")
        total_expected_delivered = delivered_total + expected_converted_from_backlog
        total_expected_delivered_rate = total_expected_delivered / locked_total
        print(f"截至当前历史已完成交付锁单数: {delivered_total} / {locked_total} ({delivered_total/locked_total:.1%})")
        print(f"累计预计完成交付锁单数(历史+未来): {total_expected_delivered:.1f} / {locked_total} ({total_expected_delivered_rate:.1%})")
    else:
        expected_converted_from_backlog = 0.0
        backlog_conversion_rate = np.nan
        print("无未交付订单，无法计算转化概率")
    
    plot_data = pd.DataFrame({
        "bin_range": idx.astype(str),
        "delivered_rate": delivered_rate.values,
        "undelivered_rate": undelivered_rate.values,
        "total_rate": total_rate.values,
        "cum_total_rate": cumulative_total_rate.values,
        "conversion_prob": conversion_prob.values,
        "delivered_count": delivered_counts.values,
        "undelivered_count": undelivered_counts.values,
        "total_count": total_counts.values,
    })
    
    primary_color = "#27AD00"
    backlog_color = "#F4A460"
    line_color = "#005783"
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=plot_data["bin_range"],
            y=plot_data["delivered_rate"],
            name="已交付 (占总锁单)",
            marker_color=primary_color,
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Bar(
            x=plot_data["bin_range"],
            y=plot_data["undelivered_rate"],
            name="未交付 (占总锁单)",
            marker_color=backlog_color,
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=plot_data["bin_range"],
            y=plot_data["cum_total_rate"],
            name="累计覆盖率",
            mode="lines+markers+text",
            line=dict(color=line_color, width=3),
            marker=dict(size=8),
            text=plot_data["cum_total_rate"].apply(lambda x: f"{x:.1%}"),
            textposition="top center",
            textfont=dict(color=line_color),
        ),
        secondary_y=True,
    )
    
    fig.add_trace(
        go.Scatter(
            x=plot_data["bin_range"],
            y=plot_data["conversion_prob"],
            name="转化概率估计",
            mode="lines+markers",
            line=dict(color="#AA00FF", width=2, dash="dot"),
            marker=dict(size=6),
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        title=f"CM0/CM1/CM2 锁单交付/未交付合并分布 (Lock >= {start_date_str})",
        barmode="stack",
        plot_bgcolor="#FFFFFF",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    
    fig.update_xaxes(
        title_text="交付周期 / 已等待时长区间 (天)",
        gridcolor="#ebedf0",
        tickangle=-45,
        tickfont=dict(color="#7B848F"),
        title_font=dict(color="#7B848F"),
    )
    
    fig.update_yaxes(
        title_text="占比 (占总锁单)",
        tickformat=".1%",
        gridcolor="#ebedf0",
        zerolinecolor="#ebedf0",
        secondary_y=False,
    )
    
    fig.update_yaxes(
        title_text="累计覆盖率",
        tickformat=".1%",
        showgrid=False,
        secondary_y=True,
        range=[0, 1.1],
    )
    
    return fig

def analyze():
    print("正在加载数据...")
    try:
        df = pd.read_parquet(DATA_PATH)
    except Exception as e:
        print(f"读取数据失败: {e}")
        return

    print("数据加载成功，开始处理...")
    
    # === 新增：保存原始 DataFrame 用于年龄趋势分析，避免重复读取 ===
    df_raw = df.copy()
    
    # 确保日期列格式正确
    if 'store_create_date' not in df.columns:
        print("错误：缺少 'store_create_date' 列")
        return
        
    df['store_create_date'] = pd.to_datetime(df['store_create_date'])
    
    # 当前时间
    current_time = datetime.now()
    print(f"当前时间: {current_time}")

    # 按门店分组统计
    # 1. 获取每个门店的创建时间 (取第一个非空值或最小值)
    # 2. 计算锁单数 (Lock_Time 不为空的行数)
    store_stats = df.groupby('Store Name').agg({
        'store_create_date': 'min',
        'Lock_Time': lambda x: x.notna().sum()
    }).reset_index()
    
    store_stats.rename(columns={'Lock_Time': 'lock_count'}, inplace=True)
    
    # 计算门店在营时间 (天)
    # 确保结果为浮点数或整数，避免 Timedelta
    store_stats['days_open'] = (current_time - store_stats['store_create_date']).dt.total_seconds() / (24 * 3600)
    
    # 过滤掉异常数据（如未来创建的门店，或者在营时间极短导致除零风险）
    # 这里保留所有数据，但处理除零
    store_stats = store_stats[store_stats['days_open'] > 0].copy()
    
    # 计算日均锁单数
    store_stats['daily_lock_rate'] = store_stats['lock_count'] / store_stats['days_open']
    
    # 每 10 天分组 (Binning)
    # 例如：0-10天为一组 (标签 0), 10-20天为一组 (标签 10)
    store_stats['age_group_10d'] = (store_stats['days_open'] // 10 * 10).astype(int)
    
    print(f"分析完成。门店总数: {len(store_stats)}")
    print(store_stats[['Store Name', 'days_open', 'lock_count', 'daily_lock_rate', 'age_group_10d']].head())

    # 绘制图表
    # 需求：使用 Plotly 绘制频数分布图
    # 解释 1: 门店在营时间分布 (每10天分组)
    # 解释 2: 日均锁单数的分布
    # 解释 3: 不同在营时长分组下的日均锁单数情况
    
    # 我们创建一个包含多个子图或多图的 HTML
    
    # 图表 1: 门店在营时长分布 (直方图) - 对应 "每 10 天分组"
    fig1 = px.histogram(
        store_stats, 
        x='days_open', 
        nbins=int(store_stats['days_open'].max() / 10),
        title='门店在营时长分布 (每10天分组)',
        labels={'days_open': '门店在营时间 (天)', 'count': '门店数量'}
    )
    fig1.update_traces(xbins=dict(size=10)) # 强制 bin size 为 10
    fig1.update_layout(height=380)

    # 图表 2: 日均锁单数分布 (直方图) - 对应 "日均锁单数" 的 "频数分布图"
    fig2 = px.histogram(
        store_stats, 
        x='daily_lock_rate',
        title='门店日均锁单数频数分布图',
        labels={'daily_lock_rate': '日均锁单数'}
    )
    fig2.update_layout(height=380)
    
    # 图表 3: 各时长分组(10天)的平均日均锁单数
    # 先聚合
    group_stats = store_stats.groupby('age_group_10d')['daily_lock_rate'].mean().reset_index()
    fig3 = px.bar(
        group_stats,
        x='age_group_10d',
        y='daily_lock_rate',
        title='各在营时长分组(10天)的平均日均锁单数',
        labels={'age_group_10d': '在营时长分组 (起始天数)', 'daily_lock_rate': '平均日均锁单数'}
    )
    fig3.update_layout(height=380)
    
    # === 获取年龄趋势图 ===
    # 1. CM0/CM1/CM2 (默认, 2023-10-12)
    fig4 = analyze_age_group_trends(df_raw, ['CM0', 'CM1', 'CM2'], "2023-10-12")
    
    # 2. DM0/DM1 (2024-05-13)
    fig5 = analyze_age_group_trends(df_raw, ['DM0', 'DM1'], "2024-05-13")
    
    # 3. LS9 (2025-11-12)
    fig6 = analyze_age_group_trends(df_raw, ['LS9'], "2025-11-12")
    
    # === 获取交付效率分析图 ===
    fig7 = analyze_delivery_duration(df_raw)
    
    # === 获取交付周期直方图 ===
    fig8 = analyze_delivery_histogram(df_raw)
    
    # === 获取未交付订单积压时长直方图 ===
    fig9 = analyze_undelivered_duration(df_raw)
    
    # === 获取交付/未交付合并分布图 ===
    fig10 = analyze_combined_delivery_backlog(df_raw)

    # 保存为 HTML
    with open(COMBINED_HTML, 'w') as f:
        f.write("<html><head><title>门店锁单及年龄分析报告</title></head><body>")
        f.write("<h1>门店锁单及年龄分析报告</h1>")
        f.write("<h2>1. 门店在营时长与锁单效率分析</h2>")
        f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<hr>")
        f.write(fig2.to_html(full_html=False, include_plotlyjs=False))
        f.write("<hr>")
        f.write(fig3.to_html(full_html=False, include_plotlyjs=False))
        
        f.write("<h2>2. 年龄段锁单趋势分析</h2>")
        
        if fig4:
            f.write("<h3>2.1 车型分组: CM0/CM1/CM2 (起始: 2023-10-12)</h3>")
            f.write(fig4.to_html(full_html=False, include_plotlyjs=False))
            
        if fig5:
            f.write("<hr>")
            f.write("<h3>2.2 车型分组: DM0/DM1 (起始: 2024-05-13)</h3>")
            f.write(fig5.to_html(full_html=False, include_plotlyjs=False))
            
        if fig6:
            f.write("<hr>")
            f.write("<h3>2.3 车型分组: LS9 (起始: 2025-11-12)</h3>")
            f.write(fig6.to_html(full_html=False, include_plotlyjs=False))
            
        if fig7:
            f.write("<hr>")
            f.write("<h2>3. 交付效率分析: 发票与锁单时间差</h2>")
            f.write(fig7.to_html(full_html=False, include_plotlyjs=False))
            
        if fig8:
            f.write("<hr>")
            f.write("<h3>3.1 CM0/CM1/CM2 交付周期分布 (直方图)</h3>")
            f.write(fig8.to_html(full_html=False, include_plotlyjs=False))
            
        if fig9:
            f.write("<hr>")
            f.write("<h3>3.2 CM0/CM1/CM2 未交付订单积压时长分布 (Lock >= 2025-01-01)</h3>")
            f.write(fig9.to_html(full_html=False, include_plotlyjs=False))
        
        if fig10:
            f.write("<hr>")
            f.write("<h3>3.3 CM0/CM1/CM2 锁单交付/未交付合并分布 (Lock >= 2025-01-01)</h3>")
            f.write(fig10.to_html(full_html=False, include_plotlyjs=False))
        
        f.write("</body></html>")
    
    print(f"所有图表已合并保存至: {COMBINED_HTML}")

if __name__ == "__main__":
    analyze()
