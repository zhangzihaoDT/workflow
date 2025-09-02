#!/usr/bin/env python3
"""
订单趋势线监测脚本
用于处理订单观察数据并进行基本描述性分析
包含数据加载、处理和计算逻辑
"""

import pandas as pd
import pyarrow.parquet as pq
from typing import Dict, Any, List
import logging
from pathlib import Path
import datetime
import os
import numpy as np
import duckdb

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局 DuckDB 连接
_db_connection = None

def get_db_connection():
    """获取 DuckDB 数据库连接（单例模式）"""
    global _db_connection
    if _db_connection is None:
        _db_connection = duckdb.connect(':memory:')
        logger.info("创建新的 DuckDB 内存数据库连接")
    return _db_connection

def initialize_database():
    """初始化数据库表结构"""
    conn = get_db_connection()
    
    # 创建原始数据表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_order_data (
            "日(Intention Payment Time)" TIMESTAMP,
            "日(Order Create Time)" TIMESTAMP,
            "日(Lock Time)" TIMESTAMP,
            "日(intention_refund_time)" TIMESTAMP,
            "日(Actual Refund Time)" TIMESTAMP,
            "DATE([Invoice Upload Time])" TIMESTAMP,
            "Parent Region Name" VARCHAR,
            "Province Name" VARCHAR,
            "first_middle_channel_name" VARCHAR,
            "平均值 开票价格" DOUBLE
        )
    """)
    
    # 创建聚合数据表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processed_order_data (
            date DATE,
            region VARCHAR,
            province VARCHAR,
            channel VARCHAR,
            订单数 INTEGER,
            小订数 INTEGER,
            锁单数 INTEGER,
            开票价格 DOUBLE,
            退订数 INTEGER
        )
    """)
    
    logger.info("数据库表结构初始化完成")

def parse_chinese_date(date_str):
    """解析中文日期格式，如'2025年8月25日'"""
    if pd.isna(date_str) or date_str == 'nan':
        return pd.NaT
    try:
        # 处理中文日期格式
        if '年' in str(date_str) and '月' in str(date_str) and '日' in str(date_str):
            date_str = str(date_str).replace('年', '-').replace('月', '-').replace('日', '')
            return pd.to_datetime(date_str)
        else:
            return pd.to_datetime(date_str, errors='coerce')
    except:
        return pd.NaT

def load_real_order_data():
    """使用 DuckDB 懒加载订单观察数据"""
    conn = get_db_connection()
    
    # 检查是否已经加载过数据
    result = conn.execute("SELECT COUNT(*) FROM processed_order_data").fetchone()
    if result[0] > 0:
        logger.info("数据已存在，直接返回")
        return conn.execute("SELECT * FROM processed_order_data").df()
    
    logger.info("开始加载原始数据到 DuckDB...")
    
    # 使用 DuckDB 直接读取 parquet 文件
    parquet_path = '/Users/zihao_/Documents/coding/dataset/formatted/order_observation_data_merged.parquet'
    
    # 清空原始数据表
    conn.execute("DELETE FROM raw_order_data")
    
    # 直接从 parquet 文件插入数据到 DuckDB
    conn.execute(f"""
        INSERT INTO raw_order_data 
        SELECT 
            "日(Intention Payment Time)",
            "日(Order Create Time)",
            "日(Lock Time)",
            "日(intention_refund_time)",
            "日(Actual Refund Time)",
            "DATE([Invoice Upload Time])",
            "Parent Region Name",
            "Province Name",
            "first_middle_channel_name",
            "平均值 开票价格"
        FROM read_parquet('{parquet_path}')
        WHERE "Parent Region Name" IS NOT NULL 
        AND "Province Name" IS NOT NULL 
        AND "first_middle_channel_name" IS NOT NULL
    """)
    
    logger.info("原始数据加载完成，开始聚合处理...")
    
    # 使用 SQL 进行数据聚合处理
    conn.execute("""
        INSERT INTO processed_order_data
        WITH date_series AS (
            SELECT DISTINCT date_trunc('day', coalesce(
                "日(Order Create Time)",
                "日(Intention Payment Time)", 
                "日(Lock Time)",
                "DATE([Invoice Upload Time])",
                "日(Actual Refund Time)",
                "日(intention_refund_time)"
            )) as date
            FROM raw_order_data
            WHERE coalesce(
                "日(Order Create Time)",
                "日(Intention Payment Time)", 
                "日(Lock Time)",
                "DATE([Invoice Upload Time])",
                "日(Actual Refund Time)",
                "日(intention_refund_time)"
            ) IS NOT NULL
        ),
        combinations AS (
            SELECT DISTINCT 
                "Parent Region Name" as region,
                "Province Name" as province,
                "first_middle_channel_name" as channel
            FROM raw_order_data
        )
        SELECT 
            ds.date::DATE as date,
            c.region,
            c.province,
            c.channel,
            COUNT(CASE WHEN date_trunc('day', "日(Order Create Time)") = ds.date THEN 1 END) as 订单数,
            COUNT(CASE WHEN date_trunc('day', "日(Intention Payment Time)") = ds.date THEN 1 END) as 小订数,
            COUNT(CASE WHEN date_trunc('day', "日(Lock Time)") = ds.date THEN 1 END) as 锁单数,
            AVG(CASE WHEN date_trunc('day', "DATE([Invoice Upload Time])") = ds.date AND "平均值 开票价格" IS NOT NULL 
                THEN "平均值 开票价格" END) as 开票价格,
            COUNT(CASE WHEN (date_trunc('day', "日(Actual Refund Time)") = ds.date OR 
                           date_trunc('day', "日(intention_refund_time)") = ds.date) THEN 1 END) as 退订数
        FROM date_series ds
        CROSS JOIN combinations c
        LEFT JOIN raw_order_data r ON (
            c.region = r."Parent Region Name" AND 
            c.province = r."Province Name" AND 
            c.channel = r."first_middle_channel_name" AND
            ds.date IN (
                date_trunc('day', r."日(Order Create Time)"),
                date_trunc('day', r."日(Intention Payment Time)"),
                date_trunc('day', r."日(Lock Time)"),
                date_trunc('day', r."DATE([Invoice Upload Time])"),
                date_trunc('day', r."日(Actual Refund Time)"),
                date_trunc('day', r."日(intention_refund_time)")
            )
        )
        GROUP BY ds.date, c.region, c.province, c.channel
        HAVING (订单数 + 小订数 + 锁单数 + 退订数) > 0
        ORDER BY ds.date, c.region, c.province, c.channel
    """)
    
    logger.info("数据聚合处理完成")
    
    # 获取处理后的数据
    processed_data = conn.execute("SELECT * FROM processed_order_data").df()
    
    # 保存聚合结果到工作区文件
    output_file = "/Users/zihao_/Documents/github/W35_workflow/processed_order_data.parquet"
    processed_data.to_parquet(output_file, index=False)
    logger.info(f"聚合结果已保存到: {output_file}")
    
    # 返回处理后的数据
    return processed_data

def load_order_data(data_path: str) -> pd.DataFrame:
    """
    加载订单数据
    """
    try:
        logger.info(f"正在加载数据文件: {data_path}")
        
        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        # 加载parquet文件
        data = pd.read_parquet(data_path)
        logger.info(f"数据加载成功，形状: {data.shape}")
        
        return data
        
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        raise

def filter_and_aggregate_data(df, region, province, channel, metric, start_date=None, end_date=None):
    """使用 DuckDB 筛选和聚合数据"""
    conn = get_db_connection()
    
    # 映射新的字段名称
    metric_mapping = {
        "order_volume": "订单数",
        "small_order_volume": "小订数", 
        "lock_volume": "锁单数",
        "avg_price": "开票价格",
        "refund_volume": "退订数"
    }
    
    actual_metric = metric_mapping.get(metric, metric)
    
    # 构建 WHERE 条件
    where_conditions = []
    if region != "全部":
        where_conditions.append(f"region = '{region}'")
    if province != "全部":
        where_conditions.append(f"province = '{province}'")
    if channel != "全部":
        where_conditions.append(f"channel = '{channel}'")
    if start_date:
        where_conditions.append(f"date >= '{start_date}'")
    if end_date:
        where_conditions.append(f"date <= '{end_date}'")
    
    where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
    
    # 使用 SQL 进行聚合查询
    query = f"""
        WITH aggregated AS (
            SELECT 
                date,
                SUM("{actual_metric}") as "{actual_metric}"
            FROM processed_order_data
            {where_clause}
            GROUP BY date
            ORDER BY date
        ),
        with_change_rate AS (
            SELECT 
                date,
                "{actual_metric}",
                (("{actual_metric}" - LAG("{actual_metric}") OVER (ORDER BY date)) / 
                 NULLIF(LAG("{actual_metric}") OVER (ORDER BY date), 0)) * 100 as change_rate
            FROM aggregated
        )
        SELECT * FROM with_change_rate
    """
    
    grouped = conn.execute(query).df()
    
    return grouped, actual_metric

def detect_data_anomaly(df, region, province, channel, metric, threshold=1.5, start_date=None, end_date=None):
    """使用 DuckDB 检测数据异动"""
    conn = get_db_connection()
    
    # 映射新的字段名称
    metric_mapping = {
        "order_volume": "订单数",
        "small_order_volume": "小订数", 
        "lock_volume": "锁单数",
        "avg_price": "开票价格",
        "refund_volume": "退订数"
    }
    
    actual_metric = metric_mapping.get(metric, metric)
    
    # 构建 WHERE 条件
    where_conditions = []
    if region != "全部":
        where_conditions.append(f"region = '{region}'")
    if province != "全部":
        where_conditions.append(f"province = '{province}'")
    if channel != "全部":
        where_conditions.append(f"channel = '{channel}'")
    if start_date:
        where_conditions.append(f"date >= '{start_date}'")
    if end_date:
        where_conditions.append(f"date <= '{end_date}'")
    
    where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
    
    # 使用 SQL 检测异常值
    query = f"""
        WITH aggregated AS (
            SELECT 
                date,
                SUM("{actual_metric}") as "{actual_metric}"
            FROM processed_order_data
            {where_clause}
            GROUP BY date
        ),
        stats AS (
            SELECT 
                AVG("{actual_metric}") as mean_val,
                STDDEV("{actual_metric}") as std_val
            FROM aggregated
        ),
        anomalies AS (
            SELECT 
                a.date,
                a."{actual_metric}",
                s.mean_val,
                s.std_val,
                (a."{actual_metric}" - s.mean_val) / s.std_val as z_score
            FROM aggregated a
            CROSS JOIN stats s
            WHERE a."{actual_metric}" > s.mean_val + {threshold} * s.std_val
            ORDER BY a.date
        )
        SELECT * FROM anomalies
    """
    
    anomalies = conn.execute(query).df()
    
    if anomalies.empty:
        return "暂无显著异常"
    
    # 格式化输出
    result_lines = []
    for _, row in anomalies.iterrows():
        result_lines.append(f"日期: {row['date']}, {actual_metric}: {row[actual_metric]:.2f}, Z-Score: {row['z_score']:.2f}")
    
    return "\n".join(result_lines)

def analyze_data_structure(data: pd.DataFrame) -> Dict[str, Any]:
    """
    分析数据结构和基本信息
    """
    logger.info("开始分析数据结构...")
    
    # 基本信息
    basic_info = {
        'shape': data.shape,
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    # 列信息
    column_info = []
    for col in data.columns:
        col_info = {
            'column_name': col,
            'data_type': str(data[col].dtype),
            'non_null_count': data[col].count(),
            'null_count': data[col].isnull().sum(),
            'null_percentage': (data[col].isnull().sum() / len(data)) * 100,
            'unique_count': data[col].nunique()
        }
        
        # 如果是数值类型，添加统计信息
        if pd.api.types.is_numeric_dtype(data[col]):
            col_info.update({
                'min_value': data[col].min(),
                'max_value': data[col].max(),
                'mean_value': data[col].mean(),
                'std_value': data[col].std()
            })
        
        column_info.append(col_info)
    
    # 数据类型分类
    numeric_columns = data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_columns = data.select_dtypes(include=['datetime64']).columns.tolist()
    
    # 缺失值统计
    missing_values = data.isnull().sum()
    missing_cells = missing_values.sum()
    
    # 重复行检查
    duplicate_rows = data.duplicated().sum()
    
    # 数据完整性
    total_cells = data.shape[0] * data.shape[1]
    data_completeness = ((total_cells - missing_cells) / total_cells) * 100 if total_cells > 0 else 0
    
    analysis_result = {
        'basic_info': basic_info,
        'column_info': column_info,
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'datetime_columns': datetime_columns,
        'missing_values': missing_values,
        'missing_cells': missing_cells,
        'duplicate_rows': duplicate_rows,
        'data_completeness': data_completeness
    }
    
    logger.info("数据结构分析完成")
    return analysis_result

def generate_order_trend_report(data: pd.DataFrame, analysis: Dict[str, Any], data_path: str) -> None:
    """
    生成订单趋势线监测报告
    """
    # 获取当前时间
    current_time = datetime.datetime.now().isoformat()
    
    # 构建报告内容
    report_content = f"""# 订单趋势线监测报告

## 报告概览
- **生成时间**: {current_time}
- **数据文件**: {data_path}
- **分析类型**: 订单观察数据基本描述性分析

---

# 数据基本信息

## 数据概览
- **数据形状**: {analysis['basic_info']['shape'][0]:,} 行 × {analysis['basic_info']['shape'][1]} 列
- **内存使用**: {analysis['basic_info']['memory_usage_mb']:.2f} MB
- **数据完整性**: {analysis['data_completeness']:.2f}%
- **重复行数**: {analysis['duplicate_rows']:,}

## 数据类型分布
- **数值列**: {len(analysis['numeric_columns'])} 个
- **分类列**: {len(analysis['categorical_columns'])} 个
- **日期列**: {len(analysis['datetime_columns'])} 个

## 字段信息详情

### 全部字段列表

| 序号 | 字段名称 | 数据类型 | 非空数量 | 缺失率 | 唯一值数量 |
|------|----------|----------|----------|--------|------------|
"""
    
    # 添加字段信息表格
    for i, col_info in enumerate(analysis['column_info'], 1):
        report_content += f"| {i} | {col_info['column_name']} | {col_info['data_type']} | {col_info['non_null_count']:,} | {col_info['null_percentage']:.2f}% | {col_info['unique_count']:,} |\n"
    
    # 数值列统计信息
    if analysis['numeric_columns']:
        report_content += "\n### 数值列统计信息\n\n"
        report_content += "| 字段名称 | 最小值 | 最大值 | 平均值 | 标准差 |\n"
        report_content += "|----------|--------|--------|--------|--------|\n"
        
        for col_info in analysis['column_info']:
            if col_info['column_name'] in analysis['numeric_columns']:
                min_val = col_info.get('min_value', 'N/A')
                max_val = col_info.get('max_value', 'N/A')
                mean_val = col_info.get('mean_value', 'N/A')
                std_val = col_info.get('std_value', 'N/A')
                
                # 格式化数值显示
                if isinstance(min_val, (int, float)):
                    min_val = f"{min_val:,.2f}" if isinstance(min_val, float) else f"{min_val:,}"
                if isinstance(max_val, (int, float)):
                    max_val = f"{max_val:,.2f}" if isinstance(max_val, float) else f"{max_val:,}"
                if isinstance(mean_val, (int, float)):
                    mean_val = f"{mean_val:,.2f}"
                if isinstance(std_val, (int, float)):
                    std_val = f"{std_val:,.2f}"
                
                report_content += f"| {col_info['column_name']} | {min_val} | {max_val} | {mean_val} | {std_val} |\n"
    else:
        report_content += "\n### 数值列统计信息\n\n无数值列\n"
    
    # 分类列信息
    if analysis['categorical_columns']:
        report_content += "\n### 分类列详情\n\n"
        report_content += ', '.join(analysis['categorical_columns'])
        report_content += "\n"
    
    # 日期列信息
    if analysis['datetime_columns']:
        report_content += "\n### 日期列详情\n\n"
        report_content += ', '.join(analysis['datetime_columns'])
        report_content += "\n"
    
    # 数据质量分析
    report_content += "\n## 数据质量分析\n\n"
    
    # 缺失值分析
    if analysis['missing_cells'] > 0:
        report_content += "### 缺失值分析\n\n"
        report_content += "| 字段名称 | 缺失数量 | 缺失比例 |\n"
        report_content += "|----------|----------|----------|\n"
        
        for col, missing_count in analysis['missing_values'].items():
            if missing_count > 0:
                missing_pct = (missing_count / len(data)) * 100
                report_content += f"| {col} | {missing_count:,} | {missing_pct:.2f}% |\n"
    else:
        report_content += "### 缺失值检查\n\n✅ 未发现缺失值\n"
    
    # 重复数据检查
    if analysis['duplicate_rows'] > 0:
        report_content += f"\n### 重复数据检查\n\n⚠️ 发现 {analysis['duplicate_rows']:,} 行重复数据\n"
    else:
        report_content += "\n### 重复数据检查\n\n✅ 未发现重复数据\n"
    
    # 数据质量评级
    data_completeness = analysis['data_completeness']
    if data_completeness >= 95:
        quality_grade = "优秀 ✅"
    elif data_completeness >= 90:
        quality_grade = "良好 ⚠️"
    elif data_completeness >= 80:
        quality_grade = "一般 ⚠️"
    else:
        quality_grade = "较差 ❌"
    
    report_content += f"\n## 数据质量评估\n\n- **整体评级**: {quality_grade}\n- **完整性得分**: {data_completeness:.2f}%\n"
    
    # 保存报告
    report_path = "/Users/zihao_/Documents/github/W35_workflow/order_trend_analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"订单趋势线监测报告已生成: {report_path}")
    print(f"\n📊 订单趋势线监测报告已生成: {report_path}")
    
    # 同时打印简要摘要到控制台
    print("\n" + "="*80)
    print("📊 订单趋势线监测 - 数据基本描述性信息")
    print("="*80)
    print(f"\n📋 数据概览: {analysis['basic_info']['shape'][0]:,} 行 × {analysis['basic_info']['shape'][1]} 列")
    print(f"📈 数据完整性: {analysis['data_completeness']:.2f}%")
    print(f"📝 详细报告已保存至: {report_path}")
    print("="*80)

def main():
    """
    主函数
    """
    # 数据文件路径
    data_path = "/Users/zihao_/Documents/coding/dataset/formatted/order_observation_data_merged.parquet"
    
    try:
        logger.info("启动订单趋势线监测脚本...")
        
        # 初始化数据库
        initialize_database()
        
        # 加载并聚合数据（这会生成聚合数据文件）
        aggregated_data = load_real_order_data()
        
        # 加载原始数据用于分析
        data = load_order_data(data_path)
        
        # 分析数据结构
        analysis = analyze_data_structure(data)
        
        # 生成报告
        generate_order_trend_report(data, analysis, data_path)
        
        logger.info("订单趋势线监测脚本执行完成")
        
    except Exception as e:
        logger.error(f"脚本执行失败: {str(e)}")
        print(f"\n❌ 错误: {str(e)}")

if __name__ == "__main__":
    main()