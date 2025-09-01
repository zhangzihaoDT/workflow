#!/usr/bin/env python3
"""
订单趋势线监测脚本
用于处理订单观察数据并进行基本描述性分析
"""

import pandas as pd
import pyarrow.parquet as pq
from typing import Dict, Any, List
import logging
from pathlib import Path
import datetime
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    data_path = "/Users/zihao_/Documents/coding/dataset/formatted/order_observation_data.parquet"
    
    try:
        logger.info("启动订单趋势线监测脚本...")
        
        # 加载数据
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