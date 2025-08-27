#!/usr/bin/env python3
"""
LangGraph异常检测工作流脚本
用于处理数据集并进行异常检测和数据完整性检查
"""

import pandas as pd
import pyarrow.parquet as pq
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
import logging
from pathlib import Path
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义工作流状态类型
from typing import TypedDict

class WorkflowState(TypedDict):
    data: pd.DataFrame
    data_path: str
    integrity_check_results: Dict[str, Any]
    errors: List[str]
    status: str
    metadata: Dict[str, Any]

# 生成异常检测报告
def generate_anomaly_report(state: WorkflowState, data_completeness: float, duplicate_rows: int, 
                           numeric_columns: list, categorical_columns: list, datetime_columns: list, 
                           missing_values: pd.Series, missing_cells: int) -> None:
    """
    生成异常检测MD报告
    """
    report_content = f"""# 异常检测报告

## 数据概览
- **数据文件**: {state['data_path']}
- **生成时间**: {state['metadata']['processing_timestamp']}
- **文件大小**: {state['metadata']['file_size_mb']:.2f} MB

## 数据基本信息
- **数据形状**: {state['data'].shape[0]} 行 × {state['data'].shape[1]} 列
- **数据完整性**: {data_completeness:.2f}%
- **重复行数**: {duplicate_rows}

## 数据类型分布
- **数值列**: {len(numeric_columns)} 个
- **分类列**: {len(categorical_columns)} 个  
- **日期列**: {len(datetime_columns)} 个

## 列信息详情
### 数值列
{', '.join(numeric_columns) if numeric_columns else '无'}

### 分类列
{', '.join(categorical_columns) if categorical_columns else '无'}

### 日期列
{', '.join(datetime_columns) if datetime_columns else '无'}

## 异常检测结果
"""
    
    if missing_cells > 0:
        report_content += "\n### 缺失值异常\n\n| 列名 | 缺失数量 | 缺失比例 |\n|------|----------|----------|\n"
        for col, missing_count in missing_values.items():
            if missing_count > 0:
                missing_pct = missing_count / len(state["data"]) * 100
                report_content += f"| {col} | {missing_count} | {missing_pct:.2f}% |\n"
    else:
        report_content += "\n### 缺失值检查\n✅ 未发现缺失值\n"
    
    if duplicate_rows > 0:
        report_content += f"\n### 重复数据异常\n⚠️ 发现 {duplicate_rows} 行重复数据\n"
    else:
        report_content += "\n### 重复数据检查\n✅ 未发现重复数据\n"
    
    # 数据质量评级
    if data_completeness >= 95:
        quality_grade = "优秀 ✅"
    elif data_completeness >= 90:
        quality_grade = "良好 ⚠️"
    elif data_completeness >= 80:
        quality_grade = "一般 ⚠️"
    else:
        quality_grade = "较差 ❌"
    
    report_content += f"\n## 数据质量评估\n- **整体评级**: {quality_grade}\n- **完整性得分**: {data_completeness:.2f}%\n"
    
    # 保存报告
    report_path = "/Users/zihao_/Documents/github/W35_workflow/anomaly_detection_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"异常检测报告已生成: {report_path}")
    print(f"\n📊 异常检测报告已生成: {report_path}")

def generate_complete_report(state: WorkflowState) -> None:
    """
    生成完整的综合报告，整合异常检测和结构检查结果
    """
    from datetime import datetime
    import os
    
    try:
        # 读取异常检测报告
        anomaly_report_path = "/Users/zihao_/Documents/github/W35_workflow/anomaly_detection_report.md"
        structure_report_path = "/Users/zihao_/Documents/github/W35_workflow/structure_check_report.md"
        
        anomaly_content = ""
        structure_content = ""
        
        if os.path.exists(anomaly_report_path):
            with open(anomaly_report_path, 'r', encoding='utf-8') as f:
                anomaly_content = f.read()
        
        if os.path.exists(structure_report_path):
            with open(structure_report_path, 'r', encoding='utf-8') as f:
                structure_content = f.read()
        
        # 生成综合报告
        complete_report = f"""# W35 异常检测工作流 - 综合分析报告

## 报告概览
- **生成时间**: {datetime.now().isoformat()}
- **工作流版本**: W35 Anomaly Detection Workflow
- **分析范围**: 数据质量检测 + 结构异常分析

---

{anomaly_content}

---

{structure_content}

---

## 综合结论

### 数据质量状况
从数据质量检测结果来看，数据的基本完整性和一致性情况。

### 结构异常状况
从结构检查结果来看，CM2车型相对于历史车型在地区分布、渠道结构、人群结构方面的变化情况。

### 建议措施
1. **数据质量方面**: 根据异常检测结果，对发现的数据质量问题进行相应处理
2. **结构异常方面**: 对发现的结构异常进行深入分析，确定是否为正常的业务变化或需要关注的异常情况
3. **持续监控**: 建议定期运行此工作流，持续监控数据质量和结构变化

---

*本报告由 W35 异常检测工作流自动生成*
"""
        
        # 保存综合报告
        complete_report_path = "/Users/zihao_/Documents/github/W35_workflow/complete_analysis_report.md"
        with open(complete_report_path, 'w', encoding='utf-8') as f:
            f.write(complete_report)
        
        logger.info(f"综合分析报告已生成: {complete_report_path}")
        print(f"\n📋 综合分析报告已生成: {complete_report_path}")
        
        # 更新状态
        state["complete_report_path"] = complete_report_path
        
    except Exception as e:
        error_msg = f"生成综合报告时发生错误: {str(e)}"
        logger.error(error_msg)
        print(f"❌ {error_msg}")

# 更新README中的mermaid图示节点
def update_readme_mermaid_node(state: WorkflowState) -> WorkflowState:
    """
    更新README.md中的mermaid工作流图示
    """
    logger.info("开始更新README.md中的mermaid图示...")
    
    try:
        mermaid_content = """# W35 异常检测工作流

## 工作流架构

```mermaid
flowchart TD
    A[开始] --> B[异常检测节点]
    B --> C[结构检查节点]
    C --> D[生成综合报告]
    D --> E[更新README图示]
    E --> F[结束]
    
    B --> B1[数据质量检查]
    B --> B2[缺失值检测]
    B --> B3[重复数据检测]
    B --> B4[生成异常检测报告]
    
    C --> C1[地区分布异常检查]
    C --> C2[渠道结构异常检查]
    C --> C3[人群结构异常检查]
    C --> C4[生成结构检查报告]
    
    D --> D1[整合异常检测报告]
    D --> D2[整合结构检查报告]
    D --> D3[生成综合分析报告]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#ffebee
    style F fill:#f1f8e9
```

## 功能说明

### 异常检测节点
- 文件存在性检查
- 数据读取验证
- 基本统计信息收集
- 缺失值检测与分析
- 重复数据检测
- 数据类型分布分析
- 异常值识别

### 结构检查节点
- 地区分布异常检测：对比CM2车型与历史车型的地区订单分布变化
- 渠道结构异常检测：分析各渠道销量占比的突变情况
- 人群结构异常检测：检测性别比例和年龄段结构的大幅变化
- 基于业务定义的时间范围进行数据筛选

### 综合报告生成
- 整合异常检测和结构检查结果
- 提供数据质量和结构异常的综合评估
- 基于检测结果提供相应的处理建议

### 报告生成
- 自动生成详细的MD格式异常检测报告
- 生成结构检查报告
- 生成综合分析报告
- 包含数据质量评级
- 提供可视化的异常统计信息

### 工作流特性
- 基于LangGraph框架构建
- 模块化节点设计
- 多维度异常检测（数据质量+业务结构）
- 完整的错误处理机制
- 详细的日志记录

## 使用方法

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行异常检测工作流
python main.py
```

## 输出文件
- `anomaly_detection_report.md`: 详细的异常检测报告
- `structure_check_report.md`: 结构检查详细报告
- `complete_analysis_report.md`: 综合分析报告
- 控制台日志: 实时处理状态信息

## 配置文件
- `business_definition.json`: 业务时间范围定义，用于结构检查的数据筛选
"""
        
        readme_path = "/Users/zihao_/Documents/github/W35_workflow/README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(mermaid_content)
        
        state["status"] = "readme_updated"
        logger.info("README.md中的mermaid图示更新完成")
        print(f"\n📝 README.md已更新: {readme_path}")
        
    except Exception as e:
        error_msg = f"更新README.md过程中发生错误: {str(e)}"
        logger.error(error_msg)
        state["errors"].append(error_msg)
        state["status"] = "failed"
    
    return state

# 异常检测节点
def anomaly_detection_node(state: WorkflowState) -> WorkflowState:
    """
    异常检测节点 - 工作流的第一个节点
    检查数据集的异常情况，包括：
    1. 文件是否存在
    2. 数据是否可以正常读取
    3. 数据基本统计信息
    4. 缺失值检查
    5. 数据类型检查
    6. 异常值检测
    """
    logger.info("开始执行异常检测...")
    
    try:
        # 1. 检查文件是否存在
        data_path = Path(state["data_path"])
        if not data_path.exists():
            error_msg = f"数据文件不存在: {state['data_path']}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["status"] = "failed"
            return state
        
        logger.info(f"数据文件存在: {state['data_path']}")
        
        # 2. 读取数据
        try:
            state["data"] = pd.read_parquet(state["data_path"])
            logger.info(f"成功读取数据，数据形状: {state['data'].shape}")
        except Exception as e:
            error_msg = f"读取数据文件失败: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["status"] = "failed"
            return state
        
        # 3. 基本统计信息
        state["integrity_check_results"]["shape"] = state["data"].shape
        state["integrity_check_results"]["columns"] = list(state["data"].columns)
        state["integrity_check_results"]["dtypes"] = state["data"].dtypes.to_dict()
        
        # 4. 缺失值检查
        missing_values = state["data"].isnull().sum()
        state["integrity_check_results"]["missing_values"] = missing_values.to_dict()
        state["integrity_check_results"]["missing_percentage"] = (missing_values / len(state["data"]) * 100).to_dict()
        
        # 5. 数据类型分布
        numeric_columns = state["data"].select_dtypes(include=['number']).columns.tolist()
        categorical_columns = state["data"].select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = state["data"].select_dtypes(include=['datetime']).columns.tolist()
        
        state["integrity_check_results"]["numeric_columns"] = numeric_columns
        state["integrity_check_results"]["categorical_columns"] = categorical_columns
        state["integrity_check_results"]["datetime_columns"] = datetime_columns
        
        # 6. 数据质量评估
        total_cells = state["data"].shape[0] * state["data"].shape[1]
        missing_cells = state["data"].isnull().sum().sum()
        data_completeness = (total_cells - missing_cells) / total_cells * 100
        
        state["integrity_check_results"]["data_completeness_percentage"] = data_completeness
        
        # 7. 基本描述性统计
        if numeric_columns:
            state["integrity_check_results"]["numeric_summary"] = state["data"][numeric_columns].describe().to_dict()
        
        # 8. 重复行检查
        duplicate_rows = state["data"].duplicated().sum()
        state["integrity_check_results"]["duplicate_rows"] = duplicate_rows
        
        # 记录元数据
        state["metadata"]["file_size_mb"] = data_path.stat().st_size / (1024 * 1024)
        state["metadata"]["processing_timestamp"] = pd.Timestamp.now().isoformat()
        
        state["status"] = "anomaly_detection_completed"
        logger.info("异常检测完成")
        
        # 生成MD报告
        generate_anomaly_report(state, data_completeness, duplicate_rows, numeric_columns, categorical_columns, datetime_columns, missing_values, missing_cells)
        
    except Exception as e:
        error_msg = f"异常检测过程中发生错误: {str(e)}"
        logger.error(error_msg)
        state["errors"].append(error_msg)
        state["status"] = "failed"
    
    return state

# 结构检查节点
def structure_check_node(state: WorkflowState) -> WorkflowState:
    """
    结构检查节点：分析CM2车型相对于历史车型的结构异常
    """
    logger.info("开始执行结构检查...")
    
    try:
        # 读取业务定义文件
        business_def_path = "/Users/zihao_/Documents/github/W35_workflow/business_definition.json"
        with open(business_def_path, 'r', encoding='utf-8') as f:
            import json
            presale_periods = json.load(f)
        
        # 读取数据
        data_path = "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"
        df = pd.read_parquet(data_path)
        
        # 转换时间列
        df['Intention_Payment_Time'] = pd.to_datetime(df['Intention_Payment_Time'])
        
        # 筛选各车型数据
        vehicle_data = {}
        for vehicle, period in presale_periods.items():
            start_date = pd.to_datetime(period['start'])
            end_date = pd.to_datetime(period['end'])
            mask = (df['Intention_Payment_Time'] >= start_date) & (df['Intention_Payment_Time'] <= end_date)
            vehicle_data[vehicle] = df[mask].copy()
        
        # 分析结果
        anomalies = []
        
        # 1. 地区分布异常检查
        region_anomalies = analyze_region_distribution(vehicle_data)
        if region_anomalies:
            anomalies.extend(region_anomalies)
        
        # 2. 渠道结构异常检查
        channel_anomalies = analyze_channel_structure(vehicle_data)
        if channel_anomalies:
            anomalies.extend(channel_anomalies)
        
        # 3. 人群结构异常检查
        demographic_anomalies = analyze_demographic_structure(vehicle_data)
        if demographic_anomalies:
            anomalies.extend(demographic_anomalies)
        
        # 4. 同比/环比异常检查
        time_series_anomalies = analyze_time_series_anomalies(vehicle_data, presale_periods)
        if time_series_anomalies:
            anomalies.extend(time_series_anomalies)
        
        # 将presale_periods添加到state中
        state['presale_periods'] = presale_periods
        
        # 生成结构检查报告
        try:
            generate_structure_report(state, vehicle_data, anomalies)
        except Exception as report_error:
    
            import traceback
            traceback.print_exc()
            raise report_error
        
        state["status"] = "structure_check_completed"
        logger.info("结构检查完成")
        
    except Exception as e:
        error_msg = f"结构检查过程中发生错误: {str(e)}"
        logger.error(error_msg)
        state["errors"].append(error_msg)
        state["status"] = "failed"
    
    return state

# 地区分布异常分析
def analyze_region_distribution(vehicle_data):
    """
    分析地区分布异常
    """
    anomalies = []
    
    # 获取CM2数据
    cm2_data = vehicle_data.get('CM2')
    if cm2_data is None or len(cm2_data) == 0:
        return ["CM2车型数据为空，无法进行地区分布分析"]
    
    # 1. CM2 vs 历史平均异常检测
    historical_vehicles = ['CM0', 'DM0', 'CM1', 'DM1']
    
    for region_col in ['Parent Region Name', 'License Province', 'license_city_level', 'License City']:
        if region_col not in cm2_data.columns:
            continue
            
        # CM2地区分布
        cm2_region_dist = cm2_data[region_col].value_counts(normalize=True)
        
        # 历史车型平均分布
        historical_dists = []
        for vehicle in historical_vehicles:
            if vehicle in vehicle_data and len(vehicle_data[vehicle]) > 0:
                hist_dist = vehicle_data[vehicle][region_col].value_counts(normalize=True)
                historical_dists.append(hist_dist)
        
        if historical_dists:
            # 计算历史平均分布
            all_regions = set()
            for dist in historical_dists:
                all_regions.update(dist.index)
            all_regions.update(cm2_region_dist.index)
            
            avg_historical = pd.Series(0.0, index=list(all_regions))
            for dist in historical_dists:
                for region in all_regions:
                    avg_historical[region] += dist.get(region, 0) / len(historical_dists)
            
            # 检查异常（变化幅度超过20%）
            for region in all_regions:
                cm2_ratio = cm2_region_dist.get(region, 0)
                hist_ratio = avg_historical.get(region, 0)
                
                # 计算变化幅度（相对变化率）
                if hist_ratio > 0 and cm2_ratio > 0.01:  # 增加占比超过1%的条件
                    change_rate = abs(cm2_ratio - hist_ratio) / hist_ratio
                    if change_rate > 0.2:  # 20%变化幅度阈值
                        change_direction = "增长" if cm2_ratio > hist_ratio else "下降"
                        anomalies.append(f"[历史对比]{region_col}中{region}地区订单占比异常{change_direction}：CM2为{cm2_ratio:.2%}，历史平均为{hist_ratio:.2%}，变化幅度{change_rate:.1%}")
                elif cm2_ratio > 0.01:  # 新出现的地区，占比超过1%
                    anomalies.append(f"[历史对比]{region_col}中{region}地区为新出现区域：CM2占比{cm2_ratio:.2%}，历史无数据")
    
    # 2. CM2 vs CM1直接对比异常检测
    cm1_data = vehicle_data.get('CM1')
    if cm1_data is not None and len(cm1_data) > 0:
        for region_col in ['Parent Region Name', 'License Province', 'license_city_level', 'License City']:
            if region_col not in cm2_data.columns or region_col not in cm1_data.columns:
                continue
                
            # CM2和CM1地区分布
            cm2_region_dist = cm2_data[region_col].value_counts(normalize=True)
            cm1_region_dist = cm1_data[region_col].value_counts(normalize=True)
            
            # 获取所有地区
            all_regions = set(cm2_region_dist.index) | set(cm1_region_dist.index)
            
            # 检查异常（变化幅度超过20%）
            for region in all_regions:
                cm2_ratio = cm2_region_dist.get(region, 0)
                cm1_ratio = cm1_region_dist.get(region, 0)
                
                # 计算变化幅度（相对变化率）
                if cm1_ratio > 0 and cm2_ratio > 0.01:  # 增加占比超过1%的条件
                    change_rate = abs(cm2_ratio - cm1_ratio) / cm1_ratio
                    if change_rate > 0.2:  # 20%变化幅度阈值
                        change_direction = "增长" if cm2_ratio > cm1_ratio else "下降"
                        anomalies.append(f"[CM1对比]{region_col}中{region}地区订单占比异常{change_direction}：CM2为{cm2_ratio:.2%}，CM1为{cm1_ratio:.2%}，变化幅度{change_rate:.1%}")
                elif cm2_ratio > 0.01:  # 新出现的地区，占比超过1%
                    anomalies.append(f"[CM1对比]{region_col}中{region}地区为新出现区域：CM2占比{cm2_ratio:.2%}，CM1无数据")
    
    return anomalies

# 渠道结构异常分析
def analyze_channel_structure(vehicle_data):
    """
    分析渠道结构异常
    """
    anomalies = []
    
    cm2_data = vehicle_data.get('CM2')
    if cm2_data is None or len(cm2_data) == 0:
        return ["CM2车型数据为空，无法进行渠道结构分析"]
    
    channel_col = 'first_middle_channel_name'
    
    if channel_col not in cm2_data.columns:
        return [f"缺少{channel_col}列，无法进行渠道分析"]
    
    # 1. CM2 vs 历史平均异常检测
    historical_vehicles = ['CM0', 'DM0', 'CM1', 'DM1']
    
    # CM2渠道分布
    cm2_channel_dist = cm2_data[channel_col].value_counts(normalize=True)
    
    # 历史渠道分布
    historical_dists = []
    for vehicle in historical_vehicles:
        if vehicle in vehicle_data and len(vehicle_data[vehicle]) > 0:
            hist_dist = vehicle_data[vehicle][channel_col].value_counts(normalize=True)
            historical_dists.append(hist_dist)
    
    if historical_dists:
        # 计算历史平均分布
        all_channels = set()
        for dist in historical_dists:
            all_channels.update(dist.index)
        all_channels.update(cm2_channel_dist.index)
        
        avg_historical = pd.Series(0.0, index=list(all_channels))
        for dist in historical_dists:
            for channel in all_channels:
                avg_historical[channel] += dist.get(channel, 0) / len(historical_dists)
        
        # 检查异常（变化幅度超过15%）
        for channel in all_channels:
            cm2_ratio = cm2_channel_dist.get(channel, 0)
            hist_ratio = avg_historical.get(channel, 0)
            
            # 计算变化幅度（相对变化率）
            if hist_ratio > 0 and cm2_ratio > 0.01:  # 增加占比超过1%的条件
                change_rate = abs(cm2_ratio - hist_ratio) / hist_ratio
                if change_rate > 0.15:  # 15%变化幅度阈值
                    change_direction = "增长" if cm2_ratio > hist_ratio else "下降"
                    anomalies.append(f"[历史对比]渠道{channel}销量占比异常{change_direction}：CM2为{cm2_ratio:.2%}，历史平均为{hist_ratio:.2%}，变化幅度{change_rate:.1%}")
            elif cm2_ratio > 0.01:  # 新出现的渠道，占比超过1%
                anomalies.append(f"[历史对比]渠道{channel}为新出现渠道：CM2占比{cm2_ratio:.2%}，历史无数据")
    
    # 2. CM2 vs CM1直接对比异常检测
    cm1_data = vehicle_data.get('CM1')
    if cm1_data is not None and len(cm1_data) > 0 and channel_col in cm1_data.columns:
        # CM2和CM1渠道分布
        cm2_channel_dist = cm2_data[channel_col].value_counts(normalize=True)
        cm1_channel_dist = cm1_data[channel_col].value_counts(normalize=True)
        
        # 获取所有渠道
        all_channels = set(cm2_channel_dist.index) | set(cm1_channel_dist.index)
        
        # 检查异常（变化幅度超过15%）
        for channel in all_channels:
            cm2_ratio = cm2_channel_dist.get(channel, 0)
            cm1_ratio = cm1_channel_dist.get(channel, 0)
            
            # 计算变化幅度（相对变化率）
            if cm1_ratio > 0 and cm2_ratio > 0.01:  # 增加占比超过1%的条件
                change_rate = abs(cm2_ratio - cm1_ratio) / cm1_ratio
                if change_rate > 0.15:  # 15%变化幅度阈值
                    change_direction = "增长" if cm2_ratio > cm1_ratio else "下降"
                    anomalies.append(f"[CM1对比]渠道{channel}销量占比异常{change_direction}：CM2为{cm2_ratio:.2%}，CM1为{cm1_ratio:.2%}，变化幅度{change_rate:.1%}")
            elif cm2_ratio > 0.01:  # 新出现的渠道，占比超过1%
                anomalies.append(f"[CM1对比]渠道{channel}为新出现渠道：CM2占比{cm2_ratio:.2%}，CM1无数据")
    
    return anomalies

# 人群结构异常分析
def analyze_demographic_structure(vehicle_data):
    """
    分析人群结构异常
    """
    anomalies = []
    
    cm2_data = vehicle_data.get('CM2')
    if cm2_data is None or len(cm2_data) == 0:
        return ["CM2车型数据为空，无法进行人群结构分析"]
    
    # 1. CM2 vs 历史平均异常检测
    historical_vehicles = ['CM0', 'DM0', 'CM1', 'DM1']
    
    # 1.1 性别比例分析
    if 'order_gender' in cm2_data.columns:
        cm2_gender_dist = cm2_data['order_gender'].value_counts(normalize=True)
        
        historical_gender_dists = []
        for vehicle in historical_vehicles:
            if vehicle in vehicle_data and len(vehicle_data[vehicle]) > 0:
                hist_dist = vehicle_data[vehicle]['order_gender'].value_counts(normalize=True)
                historical_gender_dists.append(hist_dist)
        
        if historical_gender_dists:
            all_genders = set()
            for dist in historical_gender_dists:
                all_genders.update(dist.index)
            all_genders.update(cm2_gender_dist.index)
            
            avg_historical = pd.Series(0.0, index=list(all_genders))
            for dist in historical_gender_dists:
                for gender in all_genders:
                    avg_historical[gender] += dist.get(gender, 0) / len(historical_gender_dists)
            
            # 检查性别比例异常（变化幅度超过10%）
            for gender in all_genders:
                cm2_ratio = cm2_gender_dist.get(gender, 0)
                hist_ratio = avg_historical.get(gender, 0)
                
                # 计算变化幅度（相对变化率）
                if hist_ratio > 0:
                    change_rate = abs(cm2_ratio - hist_ratio) / hist_ratio
                    if change_rate > 0.1:  # 10%变化幅度阈值
                        change_direction = "增长" if cm2_ratio > hist_ratio else "下降"
                        anomalies.append(f"[历史对比]性别{gender}比例异常{change_direction}：CM2为{cm2_ratio:.2%}，历史平均为{hist_ratio:.2%}，变化幅度{change_rate:.1%}")
    
    # 1.2 年龄段结构分析
    if 'buyer_age' in cm2_data.columns:
        # 定义年龄段
        def age_group(age):
            if pd.isna(age):
                return '未知'
            elif age < 25:
                return '25岁以下'
            elif age < 35:
                return '25-34岁'
            elif age < 45:
                return '35-44岁'
            elif age < 55:
                return '45-54岁'
            else:
                return '55岁以上'
        
        cm2_data_copy = cm2_data.copy()
        cm2_data_copy['age_group'] = cm2_data_copy['buyer_age'].apply(age_group)
        cm2_age_dist = cm2_data_copy['age_group'].value_counts(normalize=True)
        
        historical_age_dists = []
        for vehicle in historical_vehicles:
            if vehicle in vehicle_data and len(vehicle_data[vehicle]) > 0:
                vehicle_data_copy = vehicle_data[vehicle].copy()
                vehicle_data_copy['age_group'] = vehicle_data_copy['buyer_age'].apply(age_group)
                hist_dist = vehicle_data_copy['age_group'].value_counts(normalize=True)
                historical_age_dists.append(hist_dist)
        
        if historical_age_dists:
            all_age_groups = set()
            for dist in historical_age_dists:
                all_age_groups.update(dist.index)
            all_age_groups.update(cm2_age_dist.index)
            
            avg_historical = pd.Series(0.0, index=list(all_age_groups))
            for dist in historical_age_dists:
                for age_group_name in all_age_groups:
                    avg_historical[age_group_name] += dist.get(age_group_name, 0) / len(historical_age_dists)
            
            # 检查年龄段异常（变化幅度超过10%）
            for age_group_name in all_age_groups:
                cm2_ratio = cm2_age_dist.get(age_group_name, 0)
                hist_ratio = avg_historical.get(age_group_name, 0)
                
                # 计算变化幅度（相对变化率）
                if hist_ratio > 0:
                    change_rate = abs(cm2_ratio - hist_ratio) / hist_ratio
                    if change_rate > 0.1:  # 10%变化幅度阈值
                        change_direction = "增长" if cm2_ratio > hist_ratio else "下降"
                        anomalies.append(f"[历史对比]年龄段{age_group_name}比例异常{change_direction}：CM2为{cm2_ratio:.2%}，历史平均为{hist_ratio:.2%}，变化幅度{change_rate:.1%}")
    
    # 2. CM2 vs CM1直接对比异常检测
    cm1_data = vehicle_data.get('CM1')
    if cm1_data is not None and len(cm1_data) > 0:
        # 2.1 性别比例分析
        if 'order_gender' in cm2_data.columns and 'order_gender' in cm1_data.columns:
            cm2_gender_dist = cm2_data['order_gender'].value_counts(normalize=True)
            cm1_gender_dist = cm1_data['order_gender'].value_counts(normalize=True)
            
            # 获取所有性别
            all_genders = set(cm2_gender_dist.index) | set(cm1_gender_dist.index)
            
            # 检查性别比例异常（变化幅度超过10%）
            for gender in all_genders:
                cm2_ratio = cm2_gender_dist.get(gender, 0)
                cm1_ratio = cm1_gender_dist.get(gender, 0)
                
                # 计算变化幅度（相对变化率）
                if cm1_ratio > 0:
                    change_rate = abs(cm2_ratio - cm1_ratio) / cm1_ratio
                    if change_rate > 0.1:  # 10%变化幅度阈值
                        change_direction = "增长" if cm2_ratio > cm1_ratio else "下降"
                        anomalies.append(f"[CM1对比]性别{gender}比例异常{change_direction}：CM2为{cm2_ratio:.2%}，CM1为{cm1_ratio:.2%}，变化幅度{change_rate:.1%}")
        
        # 2.2 年龄段结构分析
        if 'buyer_age' in cm2_data.columns and 'buyer_age' in cm1_data.columns:
            # 定义年龄段
            def age_group(age):
                if pd.isna(age):
                    return '未知'
                elif age < 25:
                    return '25岁以下'
                elif age < 35:
                    return '25-34岁'
                elif age < 45:
                    return '35-44岁'
                elif age < 55:
                    return '45-54岁'
                else:
                    return '55岁以上'
            
            cm2_data_copy = cm2_data.copy()
            cm2_data_copy['age_group'] = cm2_data_copy['buyer_age'].apply(age_group)
            cm2_age_dist = cm2_data_copy['age_group'].value_counts(normalize=True)
            
            cm1_data_copy = cm1_data.copy()
            cm1_data_copy['age_group'] = cm1_data_copy['buyer_age'].apply(age_group)
            cm1_age_dist = cm1_data_copy['age_group'].value_counts(normalize=True)
            
            # 获取所有年龄段
            all_age_groups = set(cm2_age_dist.index) | set(cm1_age_dist.index)
            
            # 检查年龄段异常（变化幅度超过10%）
            for age_group_name in all_age_groups:
                cm2_ratio = cm2_age_dist.get(age_group_name, 0)
                cm1_ratio = cm1_age_dist.get(age_group_name, 0)
                
                # 计算变化幅度（相对变化率）
                if cm1_ratio > 0:
                    change_rate = abs(cm2_ratio - cm1_ratio) / cm1_ratio
                    if change_rate > 0.1:  # 10%变化幅度阈值
                        change_direction = "增长" if cm2_ratio > cm1_ratio else "下降"
                        anomalies.append(f"[CM1对比]年龄段{age_group_name}比例异常{change_direction}：CM2为{cm2_ratio:.2%}，CM1为{cm1_ratio:.2%}，变化幅度{change_rate:.1%}")
    
    return anomalies

# 同比/环比异常分析
def analyze_time_series_anomalies(vehicle_data, presale_periods):
    """
    分析CM2车型的同比/环比异常
    包括：日环比、同周期对比、累计同周期对比
    """
    anomalies = []
    
    cm2_data = vehicle_data.get('CM2')
    if cm2_data is None or len(cm2_data) == 0:
        return ["CM2车型数据为空，无法进行同比/环比分析"]
    
    # 获取CM2的起始日期
    cm2_start = pd.to_datetime(presale_periods['CM2']['start'])
    
    # 准备CM2每日订单数据
    cm2_data_copy = cm2_data.copy()
    cm2_data_copy['date'] = cm2_data_copy['Intention_Payment_Time'].dt.date
    cm2_daily = cm2_data_copy.groupby('date').size().reset_index(name='orders')
    cm2_daily['date'] = pd.to_datetime(cm2_daily['date'])
    cm2_daily = cm2_daily.sort_values('date')
    
    # 1. 日环比异常检查（CM2内部对比）
    for i in range(1, len(cm2_daily)):
        current_orders = cm2_daily.iloc[i]['orders']
        previous_orders = cm2_daily.iloc[i-1]['orders']
        current_date = cm2_daily.iloc[i]['date']
        
        if previous_orders > 0:
            change_rate = (current_orders - previous_orders) / previous_orders
            if abs(change_rate) > 0.5:  # 50%阈值
                change_type = "骤增" if change_rate > 0 else "骤降"
                anomalies.append(f"[日环比]{current_date.strftime('%Y-%m-%d')}订单量异常{change_type}：当日{current_orders}单，前日{previous_orders}单，变化幅度{change_rate*100:.1f}%")
    
    # 2. 同周期对比异常检查（CM2 vs CM0, CM1, DM0, DM1）
    for vehicle in ['CM0', 'CM1', 'DM0', 'DM1']:
        if vehicle not in vehicle_data or len(vehicle_data[vehicle]) == 0:
            continue
            
        vehicle_start = pd.to_datetime(presale_periods[vehicle]['start'])
        vehicle_data_copy = vehicle_data[vehicle].copy()
        vehicle_data_copy['date'] = vehicle_data_copy['Intention_Payment_Time'].dt.date
        vehicle_daily = vehicle_data_copy.groupby('date').size().reset_index(name='orders')
        vehicle_daily['date'] = pd.to_datetime(vehicle_daily['date'])
        vehicle_daily = vehicle_daily.sort_values('date')
        
        # 对比相同相对天数的订单量
        for _, cm2_row in cm2_daily.iterrows():
            cm2_date = cm2_row['date']
            cm2_orders = cm2_row['orders']
            
            # 计算相对于起始日的天数
            days_from_start = (cm2_date - cm2_start).days
            
            # 找到对应的历史车型日期
            target_date = vehicle_start + pd.Timedelta(days=days_from_start)
            
            # 查找对应日期的订单量
            vehicle_orders_on_date = vehicle_daily[vehicle_daily['date'] == target_date]
            
            if not vehicle_orders_on_date.empty:
                vehicle_orders = vehicle_orders_on_date.iloc[0]['orders']
                
                if vehicle_orders > 0:
                    change_rate = (cm2_orders - vehicle_orders) / vehicle_orders
                    if abs(change_rate) > 1.0:  # 100%阈值
                        change_type = "骤增" if change_rate > 0 else "骤降"
                        anomalies.append(f"[同周期对比]CM2在{cm2_date.strftime('%Y-%m-%d')}相对{vehicle}同期异常{change_type}：CM2为{cm2_orders}单，{vehicle}同期为{vehicle_orders}单，变化幅度{change_rate*100:.1f}%")
    
    # 3. 累计同周期对比异常检查
    for vehicle in ['CM0', 'CM1', 'DM0', 'DM1']:
        if vehicle not in vehicle_data or len(vehicle_data[vehicle]) == 0:
            continue
            
        vehicle_start = pd.to_datetime(presale_periods[vehicle]['start'])
        vehicle_data_copy = vehicle_data[vehicle].copy()
        vehicle_data_copy['date'] = vehicle_data_copy['Intention_Payment_Time'].dt.date
        vehicle_daily = vehicle_data_copy.groupby('date').size().reset_index(name='orders')
        vehicle_daily['date'] = pd.to_datetime(vehicle_daily['date'])
        vehicle_daily = vehicle_daily.sort_values('date')
        
        # 计算累计订单量对比
        for i, cm2_row in cm2_daily.iterrows():
            cm2_date = cm2_row['date']
            
            # 计算相对于起始日的天数
            days_from_start = (cm2_date - cm2_start).days
            
            # CM2累计订单量（从起始日到当前日）
            cm2_cumulative = cm2_daily[cm2_daily['date'] <= cm2_date]['orders'].sum()
            
            # 历史车型对应期间的累计订单量
            target_end_date = vehicle_start + pd.Timedelta(days=days_from_start)
            vehicle_cumulative_data = vehicle_daily[
                (vehicle_daily['date'] >= vehicle_start) & 
                (vehicle_daily['date'] <= target_end_date)
            ]
            
            if not vehicle_cumulative_data.empty:
                vehicle_cumulative = vehicle_cumulative_data['orders'].sum()
                
                if vehicle_cumulative > 0:
                    change_rate = (cm2_cumulative - vehicle_cumulative) / vehicle_cumulative
                    if abs(change_rate) > 1.0:  # 100%阈值
                        change_type = "骤增" if change_rate > 0 else "骤降"
                        anomalies.append(f"[累计同周期对比]CM2截至{cm2_date.strftime('%Y-%m-%d')}累计订单相对{vehicle}同期异常{change_type}：CM2累计{cm2_cumulative}单，{vehicle}同期累计{vehicle_cumulative}单，变化幅度{change_rate*100:.1f}%")
    
    return anomalies

# 生成日环比描述数据
def generate_time_series_description(vehicle_data, presale_periods):
    """
    生成CM2车型的日环比描述数据
    包括：CM2日订单数、同期其他车型日订单数、累计订单数对比
    """
    description_data = {
        'cm2_daily': [],
        'cm2_cumulative': [],
        'comparison_daily': {},
        'comparison_cumulative': {}
    }
    
    cm2_data = vehicle_data.get('CM2')
    if cm2_data is None or len(cm2_data) == 0:
        return description_data
    
    # 检查presale_periods是否有效
    if not presale_periods or 'CM2' not in presale_periods:
        return description_data
    
    # 获取CM2的起始日期
    cm2_start = pd.to_datetime(presale_periods['CM2']['start'])
    
    # 准备CM2每日订单数据
    cm2_data_copy = cm2_data.copy()
    cm2_data_copy['date'] = cm2_data_copy['Intention_Payment_Time'].dt.date
    cm2_daily = cm2_data_copy.groupby('date').size().reset_index(name='orders')
    cm2_daily['date'] = pd.to_datetime(cm2_daily['date'])
    cm2_daily = cm2_daily.sort_values('date')
    
    # 计算CM2累计订单数
    cm2_daily['cumulative'] = cm2_daily['orders'].cumsum()
    
    # 存储CM2数据（仅保留最后一天）
    if not cm2_daily.empty:
        last_row = cm2_daily.iloc[-1]
        days_from_start = (last_row['date'] - cm2_start).days + 1
        description_data['cm2_daily'].append({
            'date': last_row['date'].strftime('%Y-%m-%d'),
            'day_n': days_from_start,
            'orders': last_row['orders']
        })
        description_data['cm2_cumulative'].append({
            'date': last_row['date'].strftime('%Y-%m-%d'),
            'day_n': days_from_start,
            'cumulative_orders': last_row['cumulative']
        })
    
    # 处理其他车型的同期数据
    for vehicle in ['CM0', 'CM1', 'DM0', 'DM1']:
        if vehicle not in vehicle_data or len(vehicle_data[vehicle]) == 0:
            continue
        
        # 检查该车型的presale_periods是否存在
        if vehicle not in presale_periods:
            continue
            
        vehicle_start = pd.to_datetime(presale_periods[vehicle]['start'])
        vehicle_data_copy = vehicle_data[vehicle].copy()
        vehicle_data_copy['date'] = vehicle_data_copy['Intention_Payment_Time'].dt.date
        vehicle_daily = vehicle_data_copy.groupby('date').size().reset_index(name='orders')
        vehicle_daily['date'] = pd.to_datetime(vehicle_daily['date'])
        vehicle_daily = vehicle_daily.sort_values('date')
        vehicle_daily['cumulative'] = vehicle_daily['orders'].cumsum()
        
        description_data['comparison_daily'][vehicle] = []
        description_data['comparison_cumulative'][vehicle] = []
        
        # 对比相同相对天数的数据（仅保留最后一天）
        if not cm2_daily.empty:
            cm2_row = cm2_daily.iloc[-1]
            cm2_date = cm2_row['date']
            days_from_start = (cm2_date - cm2_start).days
            
            # 找到对应的历史车型日期
            target_date = vehicle_start + pd.Timedelta(days=days_from_start)
            
            # 查找对应日期的订单量
            vehicle_orders_on_date = vehicle_daily[vehicle_daily['date'] == target_date]
            
            if not vehicle_orders_on_date.empty:
                vehicle_orders = vehicle_orders_on_date.iloc[0]['orders']
                vehicle_cumulative = vehicle_daily[vehicle_daily['date'] <= target_date]['orders'].sum()
                
                description_data['comparison_daily'][vehicle].append({
                    'cm2_date': cm2_date.strftime('%Y-%m-%d'),
                    'vehicle_date': target_date.strftime('%Y-%m-%d'),
                    'day_n': days_from_start + 1,
                    'orders': vehicle_orders
                })
                description_data['comparison_cumulative'][vehicle].append({
                    'cm2_date': cm2_date.strftime('%Y-%m-%d'),
                    'vehicle_date': target_date.strftime('%Y-%m-%d'),
                    'day_n': days_from_start + 1,
                    'cumulative_orders': vehicle_cumulative
                })
    
    return description_data

# 生成退订描述数据
def generate_refund_description(vehicle_data, presale_periods):
    """
    生成CM2车型的退订描述数据
    包括：CM2日退订数、同期其他车型日退订数、累计退订数对比
    """

    
    refund_data = {
        'cm2_daily': [],
        'cm2_cumulative': [],
        'comparison_daily': {},
        'comparison_cumulative': {}
    }
    
    cm2_data = vehicle_data.get('CM2')
    if cm2_data is None or len(cm2_data) == 0:
        return refund_data
    
    # 检查presale_periods是否有效
    if not presale_periods or 'CM2' not in presale_periods:
        return refund_data
    
    # 过滤有退订时间的数据
    cm2_refund_data = cm2_data[cm2_data['intention_refund_time'].notna()].copy()
    if len(cm2_refund_data) == 0:
        return refund_data
    
    # 获取CM2的起始日期
    cm2_start = pd.to_datetime(presale_periods['CM2']['start'])
    
    # 准备CM2每日退订数据
    cm2_refund_data['refund_date'] = cm2_refund_data['intention_refund_time'].dt.date
    cm2_daily_refund = cm2_refund_data.groupby('refund_date').size().reset_index(name='refunds')
    cm2_daily_refund['refund_date'] = pd.to_datetime(cm2_daily_refund['refund_date'])
    cm2_daily_refund = cm2_daily_refund.sort_values('refund_date')
    
    # 计算CM2累计退订数
    cm2_daily_refund['cumulative'] = cm2_daily_refund['refunds'].cumsum()
    
    # 存储CM2数据（仅保留最后一天）
    if not cm2_daily_refund.empty:
        last_row = cm2_daily_refund.iloc[-1]
        days_from_start = (last_row['refund_date'] - cm2_start).days + 1
        refund_data['cm2_daily'].append({
            'date': last_row['refund_date'].strftime('%Y-%m-%d'),
            'day_n': days_from_start,
            'refunds': last_row['refunds']
        })
        refund_data['cm2_cumulative'].append({
            'date': last_row['refund_date'].strftime('%Y-%m-%d'),
            'day_n': days_from_start,
            'cumulative_refunds': last_row['cumulative']
        })
    
    # 处理其他车型的同期退订数据
    for vehicle in ['CM0', 'CM1', 'DM0', 'DM1']:
        if vehicle not in vehicle_data or len(vehicle_data[vehicle]) == 0:
            continue
        
        # 检查该车型的presale_periods是否存在
        if vehicle not in presale_periods:
            continue
            
        vehicle_start = pd.to_datetime(presale_periods[vehicle]['start'])
        vehicle_refund_data = vehicle_data[vehicle][vehicle_data[vehicle]['intention_refund_time'].notna()].copy()
        
        if len(vehicle_refund_data) == 0:
            continue
            
        vehicle_refund_data['refund_date'] = vehicle_refund_data['intention_refund_time'].dt.date
        vehicle_daily_refund = vehicle_refund_data.groupby('refund_date').size().reset_index(name='refunds')
        vehicle_daily_refund['refund_date'] = pd.to_datetime(vehicle_daily_refund['refund_date'])
        vehicle_daily_refund = vehicle_daily_refund.sort_values('refund_date')
        vehicle_daily_refund['cumulative'] = vehicle_daily_refund['refunds'].cumsum()
        
        refund_data['comparison_daily'][vehicle] = []
        refund_data['comparison_cumulative'][vehicle] = []
        
        # 对比相同相对天数的数据（仅保留最后一天）
        if not cm2_daily_refund.empty:
            cm2_row = cm2_daily_refund.iloc[-1]
            cm2_refund_date = cm2_row['refund_date']
            days_from_start = (cm2_refund_date - cm2_start).days
            
            # 找到对应的历史车型日期
            target_date = vehicle_start + pd.Timedelta(days=days_from_start)
            
            # 查找对应日期的退订量
            vehicle_refunds_on_date = vehicle_daily_refund[vehicle_daily_refund['refund_date'] == target_date]
            
            if not vehicle_refunds_on_date.empty:
                vehicle_refunds = vehicle_refunds_on_date.iloc[0]['refunds']
                vehicle_cumulative = vehicle_daily_refund[vehicle_daily_refund['refund_date'] <= target_date]['refunds'].sum()
                
                refund_data['comparison_daily'][vehicle].append({
                    'cm2_date': cm2_refund_date.strftime('%Y-%m-%d'),
                    'vehicle_date': target_date.strftime('%Y-%m-%d'),
                    'day_n': days_from_start + 1,
                    'refunds': vehicle_refunds
                })
                refund_data['comparison_cumulative'][vehicle].append({
                    'cm2_date': cm2_refund_date.strftime('%Y-%m-%d'),
                    'vehicle_date': target_date.strftime('%Y-%m-%d'),
                    'day_n': days_from_start + 1,
                    'cumulative_refunds': vehicle_cumulative
                })
    
    return refund_data

def generate_previous_day_data(vehicle_data, presale_periods):
    """
    生成CM2车型前一天的订单和退订数据
    包括：CM2前一天订单数、退订数，同期其他车型对比，以及累计数据
    """
    previous_day_data = {
        'cm2_previous_day': None,
        'comparison_previous_day': {},
        'cm2_cumulative_previous': None,
        'comparison_cumulative_previous': {}
    }
    
    cm2_data = vehicle_data.get('CM2')
    if cm2_data is None or len(cm2_data) == 0:
        return previous_day_data
    
    # 检查presale_periods是否有效
    if not presale_periods or 'CM2' not in presale_periods:
        return previous_day_data
    
    # 获取CM2的起始日期
    cm2_start = pd.to_datetime(presale_periods['CM2']['start'])
    
    # 准备CM2每日订单数据
    cm2_data_copy = cm2_data.copy()
    cm2_data_copy['date'] = cm2_data_copy['Intention_Payment_Time'].dt.date
    cm2_daily = cm2_data_copy.groupby('date').size().reset_index(name='orders')
    cm2_daily['date'] = pd.to_datetime(cm2_daily['date'])
    cm2_daily = cm2_daily.sort_values('date')
    cm2_daily['cumulative'] = cm2_daily['orders'].cumsum()
    
    # 准备CM2退订数据
    cm2_refund_data = cm2_data[cm2_data['intention_refund_time'].notna()].copy()
    if len(cm2_refund_data) > 0:
        cm2_refund_data['refund_date'] = cm2_refund_data['intention_refund_time'].dt.date
        cm2_daily_refund = cm2_refund_data.groupby('refund_date').size().reset_index(name='refunds')
        cm2_daily_refund['refund_date'] = pd.to_datetime(cm2_daily_refund['refund_date'])
        cm2_daily_refund = cm2_daily_refund.sort_values('refund_date')
        cm2_daily_refund['cumulative_refunds'] = cm2_daily_refund['refunds'].cumsum()
    else:
        cm2_daily_refund = pd.DataFrame(columns=['refund_date', 'refunds', 'cumulative_refunds'])
    
    # 获取前一天数据（倒数第二天）
    if len(cm2_daily) >= 2:
        previous_day_row = cm2_daily.iloc[-2]  # 倒数第二天
        previous_day_date = previous_day_row['date']
        days_from_start = (previous_day_date - cm2_start).days + 1
        
        # 获取前一天的退订数
        previous_day_refunds = 0
        if len(cm2_daily_refund) > 0:
            refund_on_date = cm2_daily_refund[cm2_daily_refund['refund_date'] == previous_day_date]
            if not refund_on_date.empty:
                previous_day_refunds = refund_on_date.iloc[0]['refunds']
        
        previous_day_data['cm2_previous_day'] = {
            'date': previous_day_date.strftime('%Y-%m-%d'),
            'day_n': days_from_start,
            'orders': previous_day_row['orders'],
            'refunds': previous_day_refunds
        }
        
        # 获取前一天的累计数据
        cumulative_orders = previous_day_row['cumulative']
        cumulative_refunds = 0
        if len(cm2_daily_refund) > 0:
            cumulative_refunds = cm2_daily_refund[cm2_daily_refund['refund_date'] <= previous_day_date]['refunds'].sum()
        
        previous_day_data['cm2_cumulative_previous'] = {
            'day_n': days_from_start,
            'cumulative_orders': cumulative_orders,
            'cumulative_refunds': cumulative_refunds
        }
        
        # 处理其他车型的同期数据
        for vehicle in ['CM0', 'CM1', 'DM0', 'DM1']:
            if vehicle not in vehicle_data or len(vehicle_data[vehicle]) == 0:
                continue
            
            # 检查该车型的presale_periods是否存在
            if vehicle not in presale_periods:
                continue
                
            vehicle_start = pd.to_datetime(presale_periods[vehicle]['start'])
            
            # 准备车型订单数据
            vehicle_data_copy = vehicle_data[vehicle].copy()
            vehicle_data_copy['date'] = vehicle_data_copy['Intention_Payment_Time'].dt.date
            vehicle_daily = vehicle_data_copy.groupby('date').size().reset_index(name='orders')
            vehicle_daily['date'] = pd.to_datetime(vehicle_daily['date'])
            vehicle_daily = vehicle_daily.sort_values('date')
            vehicle_daily['cumulative'] = vehicle_daily['orders'].cumsum()
            
            # 准备车型退订数据
            vehicle_refund_data = vehicle_data[vehicle][vehicle_data[vehicle]['intention_refund_time'].notna()].copy()
            if len(vehicle_refund_data) > 0:
                vehicle_refund_data['refund_date'] = vehicle_refund_data['intention_refund_time'].dt.date
                vehicle_daily_refund = vehicle_refund_data.groupby('refund_date').size().reset_index(name='refunds')
                vehicle_daily_refund['refund_date'] = pd.to_datetime(vehicle_daily_refund['refund_date'])
                vehicle_daily_refund = vehicle_daily_refund.sort_values('refund_date')
            else:
                vehicle_daily_refund = pd.DataFrame(columns=['refund_date', 'refunds'])
            
            # 找到对应的历史车型日期（前一天）
            target_date = vehicle_start + pd.Timedelta(days=days_from_start-1)
            
            # 查找对应日期的订单量和退订量
            vehicle_orders_on_date = vehicle_daily[vehicle_daily['date'] == target_date]
            vehicle_refunds_on_date = 0
            if len(vehicle_daily_refund) > 0:
                refund_on_date = vehicle_daily_refund[vehicle_daily_refund['refund_date'] == target_date]
                if not refund_on_date.empty:
                    vehicle_refunds_on_date = refund_on_date.iloc[0]['refunds']
            
            if not vehicle_orders_on_date.empty:
                vehicle_orders = vehicle_orders_on_date.iloc[0]['orders']
                vehicle_cumulative_orders = vehicle_daily[vehicle_daily['date'] <= target_date]['orders'].sum()
                vehicle_cumulative_refunds = 0
                if len(vehicle_daily_refund) > 0:
                    vehicle_cumulative_refunds = vehicle_daily_refund[vehicle_daily_refund['refund_date'] <= target_date]['refunds'].sum()
                
                previous_day_data['comparison_previous_day'][vehicle] = {
                    'vehicle_date': target_date.strftime('%Y-%m-%d'),
                    'day_n': days_from_start,
                    'orders': vehicle_orders,
                    'refunds': vehicle_refunds_on_date
                }
                
                previous_day_data['comparison_cumulative_previous'][vehicle] = {
                    'day_n': days_from_start,
                    'cumulative_orders': vehicle_cumulative_orders,
                    'cumulative_refunds': vehicle_cumulative_refunds
                }
    
    return previous_day_data

# 生成结构检查报告
def generate_structure_report(state, vehicle_data, anomalies):
    """
    生成结构检查报告
    """
    from datetime import datetime
    
    report_content = f"""# 结构检查报告

## 报告概览
- **生成时间**: {datetime.now().isoformat()}
- **分析车型**: CM2 vs 历史车型(CM0, DM0, CM1) + CM2 vs CM1直接对比
- **检查维度**: 地区分布、渠道结构、人群结构

## 数据概况
"""
    
    # 添加各车型数据量统计
    for vehicle, data in vehicle_data.items():
        report_content += f"- **{vehicle}车型**: {len(data)} 条订单数据\n"
    
    report_content += "\n## 结构异常检测结果\n\n"
    
    # 分类异常 - 区分历史对比和CM1对比
    region_anomalies_hist = [a for a in anomalies if '[历史对比]' in a and any(region_type in a for region_type in ['Parent Region Name', 'License Province', 'license_city_level', 'License City'])]
    region_anomalies_cm1 = [a for a in anomalies if '[CM1对比]' in a and any(region_type in a for region_type in ['Parent Region Name', 'License Province', 'license_city_level', 'License City'])]
    
    channel_anomalies_hist = [a for a in anomalies if '[历史对比]' in a and '渠道' in a]
    channel_anomalies_cm1 = [a for a in anomalies if '[CM1对比]' in a and '渠道' in a]
    
    demographic_anomalies_hist = [a for a in anomalies if '[历史对比]' in a and any(demo_type in a for demo_type in ['性别', '年龄段'])]
    demographic_anomalies_cm1 = [a for a in anomalies if '[CM1对比]' in a and any(demo_type in a for demo_type in ['性别', '年龄段'])]
    
    # 同比/环比异常
    time_series_anomalies_daily = [a for a in anomalies if '[日环比]' in a]
    time_series_anomalies_period = [a for a in anomalies if '[同周期对比]' in a]
    time_series_anomalies_cumulative = [a for a in anomalies if '[累计同周期对比]' in a]
    
    # 地区分布异常
    report_content += "### 🌍 地区分布异常检测\n\n"
    
    # 历史对比结果
    report_content += "#### 📊 CM2 vs 历史平均对比\n\n"
    if region_anomalies_hist:
        report_content += "**🚨 发现地区分布异常:**\n\n"
        report_content += "| 序号 | 地区类型 | 地区名称 | CM2占比 | 历史平均占比 | 变化幅度 | 异常类型 |\n"
        report_content += "|------|----------|----------|---------|-------------|----------|----------|\n"
        
        for i, anomaly in enumerate(region_anomalies_hist, 1):
            clean_anomaly = anomaly.replace('[历史对比]', '')
            
            # 解析异常信息
            region_type_match = re.search(r'(Parent Region Name|License Province|license_city_level|License City)中(.+?)地区', clean_anomaly)
            cm2_match = re.search(r'CM2为([\d.]+%)', clean_anomaly)
            hist_match = re.search(r'历史平均为([\d.]+%)', clean_anomaly)
            change_match = re.search(r'变化幅度([\d.]+%)', clean_anomaly)
            
            region_type = region_type_match.group(1) if region_type_match else ""
            region_name = region_type_match.group(2) if region_type_match else ""
            cm2_ratio = cm2_match.group(1) if cm2_match else ""
            hist_ratio = hist_match.group(1) if hist_match else ""
            change_rate = change_match.group(1) if change_match else ""
            
            # 判断异常类型并添加emoji
            if "异常增长" in clean_anomaly:
                anomaly_type = "📈 增长"
            elif "异常下降" in clean_anomaly:
                anomaly_type = "📉 下降"
            else:
                anomaly_type = "异常"
            
            report_content += f"| {i} | {region_type} | {region_name} | {cm2_ratio} | {hist_ratio} | {change_rate} | {anomaly_type} |\n"
    else:
        report_content += "**✅ 地区分布正常:** 所有地区订单占比变化均在20%阈值范围内。\n"
    
    # CM1对比结果
    report_content += "\n#### 🔄 CM2 vs CM1直接对比\n\n"
    if region_anomalies_cm1:
        report_content += "**🚨 发现地区分布异常:**\n\n"
        report_content += "| 序号 | 地区类型 | 地区名称 | CM2占比 | CM1占比 | 变化幅度 | 异常类型 |\n"
        report_content += "|------|----------|----------|---------|---------|----------|----------|\n"
        
        for i, anomaly in enumerate(region_anomalies_cm1, 1):
            clean_anomaly = anomaly.replace('[CM1对比]', '')
            
            # 解析异常信息
            region_type_match = re.search(r'(Parent Region Name|License Province|license_city_level|License City)中(.+?)地区', clean_anomaly)
            cm2_match = re.search(r'CM2为([\d.]+%)', clean_anomaly)
            cm1_match = re.search(r'CM1为([\d.]+%)', clean_anomaly)
            change_match = re.search(r'变化幅度([\d.]+%)', clean_anomaly)
            
            region_type = region_type_match.group(1) if region_type_match else ""
            region_name = region_type_match.group(2) if region_type_match else ""
            cm2_ratio = cm2_match.group(1) if cm2_match else ""
            cm1_ratio = cm1_match.group(1) if cm1_match else ""
            change_rate = change_match.group(1) if change_match else ""
            
            # 判断异常类型并添加emoji
            if "异常增长" in clean_anomaly:
                anomaly_type = "📈 增长"
            elif "异常下降" in clean_anomaly:
                anomaly_type = "📉 下降"
            else:
                anomaly_type = "异常"
            
            report_content += f"| {i} | {region_type} | {region_name} | {cm2_ratio} | {cm1_ratio} | {change_rate} | {anomaly_type} |\n"
    else:
        report_content += "**✅ 地区分布正常:** 相比CM1，所有地区订单占比变化均在20%阈值范围内。\n"
    
    # 渠道结构异常
    report_content += "\n### 🛒 渠道结构异常检测\n\n"
    
    # 历史对比结果
    report_content += "#### 📊 CM2 vs 历史平均对比\n\n"
    if channel_anomalies_hist:
        report_content += "**🚨 发现渠道结构异常:**\n\n"
        report_content += "| 序号 | 渠道名称 | CM2占比 | 历史平均占比 | 变化幅度 | 异常类型 |\n"
        report_content += "|------|----------|---------|-------------|----------|----------|\n"
        
        for i, anomaly in enumerate(channel_anomalies_hist, 1):
            clean_anomaly = anomaly.replace('[历史对比]', '')
            
            # 解析异常信息
            channel_match = re.search(r'渠道(.+?)销量占比', clean_anomaly)
            cm2_match = re.search(r'CM2为([\d.]+%)', clean_anomaly)
            hist_match = re.search(r'历史平均为([\d.]+%)', clean_anomaly)
            change_match = re.search(r'变化幅度([\d.]+%)', clean_anomaly)
            
            channel_name = channel_match.group(1) if channel_match else ""
            cm2_ratio = cm2_match.group(1) if cm2_match else ""
            hist_ratio = hist_match.group(1) if hist_match else ""
            change_rate = change_match.group(1) if change_match else ""
            
            # 判断异常类型并添加emoji
            if "异常增长" in clean_anomaly:
                anomaly_type = "📈 增长"
            elif "异常下降" in clean_anomaly:
                anomaly_type = "📉 下降"
            else:
                anomaly_type = "异常"
            
            report_content += f"| {i} | {channel_name} | {cm2_ratio} | {hist_ratio} | {change_rate} | {anomaly_type} |\n"
    else:
        report_content += "**✅ 渠道结构正常:** 所有渠道销量占比变化均在15%阈值范围内。\n"
    
    # CM1对比结果
    report_content += "\n#### 🔄 CM2 vs CM1直接对比\n\n"
    if channel_anomalies_cm1:
        report_content += "**🚨 发现渠道结构异常:**\n\n"
        report_content += "| 序号 | 渠道名称 | CM2占比 | CM1占比 | 变化幅度 | 异常类型 |\n"
        report_content += "|------|----------|---------|---------|----------|----------|\n"
        
        for i, anomaly in enumerate(channel_anomalies_cm1, 1):
            clean_anomaly = anomaly.replace('[CM1对比]', '')
            
            # 解析异常信息
            channel_match = re.search(r'渠道(.+?)销量占比', clean_anomaly)
            cm2_match = re.search(r'CM2为([\d.]+%)', clean_anomaly)
            cm1_match = re.search(r'CM1为([\d.]+%)', clean_anomaly)
            change_match = re.search(r'变化幅度([\d.]+%)', clean_anomaly)
            
            channel_name = channel_match.group(1) if channel_match else ""
            cm2_ratio = cm2_match.group(1) if cm2_match else ""
            cm1_ratio = cm1_match.group(1) if cm1_match else ""
            change_rate = change_match.group(1) if change_match else ""
            
            # 判断异常类型并添加emoji
            if "异常增长" in clean_anomaly:
                anomaly_type = "📈 增长"
            elif "异常下降" in clean_anomaly:
                anomaly_type = "📉 下降"
            else:
                anomaly_type = "异常"
            
            report_content += f"| {i} | {channel_name} | {cm2_ratio} | {cm1_ratio} | {change_rate} | {anomaly_type} |\n"
    else:
        report_content += "**✅ 渠道结构正常:** 相比CM1，所有渠道销量占比变化均在15%阈值范围内。\n"
    
    # 人群结构异常
    report_content += "\n### 👥 人群结构异常检测\n\n"
    
    # 历史对比结果
    report_content += "#### 📊 CM2 vs 历史平均对比\n\n"
    if demographic_anomalies_hist:
        report_content += "**🚨 发现人群结构异常:**\n\n"
        report_content += "| 序号 | 人群类型 | 人群名称 | CM2占比 | 历史平均占比 | 变化幅度 | 异常类型 |\n"
        report_content += "|------|----------|----------|---------|-------------|----------|----------|\n"
        
        for i, anomaly in enumerate(demographic_anomalies_hist, 1):
            clean_anomaly = anomaly.replace('[历史对比]', '')
            
            # 解析异常信息
            demo_match = re.search(r'(性别|年龄段)(.+?)比例异常', clean_anomaly)
            cm2_match = re.search(r'CM2为([\d.]+%)', clean_anomaly)
            hist_match = re.search(r'历史平均为([\d.]+%)', clean_anomaly)
            change_match = re.search(r'变化幅度([\d.]+%)', clean_anomaly)
            
            demo_type = demo_match.group(1) if demo_match else ""
            demo_name = demo_match.group(2) if demo_match else ""
            cm2_ratio = cm2_match.group(1) if cm2_match else ""
            hist_ratio = hist_match.group(1) if hist_match else ""
            change_rate = change_match.group(1) if change_match else ""
            
            # 判断异常类型并添加emoji
            if "异常增长" in clean_anomaly:
                anomaly_type = "📈 增长"
            elif "异常下降" in clean_anomaly:
                anomaly_type = "📉 下降"
            else:
                anomaly_type = "异常"
            
            report_content += f"| {i} | {demo_type} | {demo_name} | {cm2_ratio} | {hist_ratio} | {change_rate} | {anomaly_type} |\n"
    else:
        report_content += "**✅ 人群结构正常:** 所有性别比例和年龄段结构变化均在10%阈值范围内。\n"
    
    # CM1对比结果
    report_content += "\n#### 🔄 CM2 vs CM1直接对比\n\n"
    if demographic_anomalies_cm1:
        report_content += "**🚨 发现人群结构异常:**\n\n"
        report_content += "| 序号 | 人群类型 | 人群名称 | CM2占比 | CM1占比 | 变化幅度 | 异常类型 |\n"
        report_content += "|------|----------|----------|---------|---------|----------|----------|\n"
        
        for i, anomaly in enumerate(demographic_anomalies_cm1, 1):
            clean_anomaly = anomaly.replace('[CM1对比]', '')
            
            # 解析异常信息
            demo_match = re.search(r'(性别|年龄段)(.+?)比例异常', clean_anomaly)
            cm2_match = re.search(r'CM2为([\d.]+%)', clean_anomaly)
            cm1_match = re.search(r'CM1为([\d.]+%)', clean_anomaly)
            change_match = re.search(r'变化幅度([\d.]+%)', clean_anomaly)
            
            demo_type = demo_match.group(1) if demo_match else ""
            demo_name = demo_match.group(2) if demo_match else ""
            cm2_ratio = cm2_match.group(1) if cm2_match else ""
            cm1_ratio = cm1_match.group(1) if cm1_match else ""
            change_rate = change_match.group(1) if change_match else ""
            
            # 判断异常类型并添加emoji
            if "异常增长" in clean_anomaly:
                anomaly_type = "📈 增长"
            elif "异常下降" in clean_anomaly:
                anomaly_type = "📉 下降"
            else:
                anomaly_type = "异常"
            
            report_content += f"| {i} | {demo_type} | {demo_name} | {cm2_ratio} | {cm1_ratio} | {change_rate} | {anomaly_type} |\n"
    else:
        report_content += "**✅ 人群结构正常:** 相比CM1，所有性别比例和年龄段结构变化均在10%阈值范围内。\n"
    
    # 同比/环比异常
    report_content += "\n### 📈 同比/环比异常检测\n\n"
    
    # 生成日环比描述数据
    time_series_desc = generate_time_series_description(vehicle_data, state.get('presale_periods', {}))
    
    # 将时间序列数据存储到state中，供退订率日环比异常检测使用
    if 'time_series_data' not in state:
        state['time_series_data'] = {}
    
    # 准备CM2每日数据用于退订率计算
    cm2_data = vehicle_data.get('CM2')
    if cm2_data is not None and len(cm2_data) > 0:
        presale_periods = state.get('presale_periods', {})
        if presale_periods and 'CM2' in presale_periods:
            cm2_start = pd.to_datetime(presale_periods['CM2']['start'])
            
            # 准备CM2每日订单数据
            cm2_data_copy = cm2_data.copy()
            cm2_data_copy['date'] = cm2_data_copy['Intention_Payment_Time'].dt.date
            cm2_daily = cm2_data_copy.groupby('date').size().reset_index(name='orders')
            cm2_daily['date'] = pd.to_datetime(cm2_daily['date'])
            cm2_daily = cm2_daily.sort_values('date')
            cm2_daily['cumulative_orders'] = cm2_daily['orders'].cumsum()
            
            # 准备CM2退订数据
            cm2_refund_data = cm2_data[cm2_data['intention_refund_time'].notna()].copy()
            if len(cm2_refund_data) > 0:
                cm2_refund_data['refund_date'] = cm2_refund_data['intention_refund_time'].dt.date
                cm2_daily_refund = cm2_refund_data.groupby('refund_date').size().reset_index(name='refunds')
                cm2_daily_refund['refund_date'] = pd.to_datetime(cm2_daily_refund['refund_date'])
                cm2_daily_refund = cm2_daily_refund.sort_values('refund_date')
                cm2_daily_refund['cumulative_refunds'] = cm2_daily_refund['refunds'].cumsum()
                
                # 合并订单和退订数据
                cm2_daily = cm2_daily.merge(cm2_daily_refund[['refund_date', 'cumulative_refunds']], 
                                          left_on='date', right_on='refund_date', how='left')
                cm2_daily['cumulative_refunds'] = cm2_daily['cumulative_refunds'].fillna(method='ffill').fillna(0)
            else:
                cm2_daily['cumulative_refunds'] = 0
            
            state['time_series_data']['cm2_daily_data'] = cm2_daily
    
    # 日环比描述
    report_content += "#### 📊 日环比描述\n\n"
    
    # CM2日订单数描述
    if time_series_desc['cm2_daily']:
        report_content += "**CM2车型日订单数:**\n\n"
        for data in time_series_desc['cm2_daily']:  # 显示所有天数
            report_content += f"- 第{data['day_n']}日 ({data['date']}): {data['orders']}单\n"
    
    # 同期其他车型日订单数
    report_content += "\n**同期其他车型日订单数对比:**\n\n"
    for vehicle in ['CM0', 'CM1', 'DM0', 'DM1']:
        if vehicle in time_series_desc['comparison_daily'] and time_series_desc['comparison_daily'][vehicle]:
            report_content += f"- **{vehicle}车型**: "
            for data in time_series_desc['comparison_daily'][vehicle]:  # 显示所有天数
                report_content += f"第{data['day_n']}日({data['vehicle_date']}):{data['orders']}单; "
            report_content += "\n"
    
    # CM2累计订单数描述
    if time_series_desc['cm2_cumulative']:
        report_content += "\n**CM2车型累计订单数:**\n\n"
        for data in time_series_desc['cm2_cumulative']:  # 显示所有天数
            report_content += f"- 第{data['day_n']}日 ({data['date']}): 累计{data['cumulative_orders']}单\n"
    
    # 同期其他车型累计订单数
    report_content += "\n**同期其他车型累计订单数对比:**\n\n"
    for vehicle in ['CM0', 'CM1', 'DM0', 'DM1']:
        if vehicle in time_series_desc['comparison_cumulative'] and time_series_desc['comparison_cumulative'][vehicle]:
            report_content += f"- **{vehicle}车型**: "
            for data in time_series_desc['comparison_cumulative'][vehicle]:  # 显示所有天数
                report_content += f"第{data['day_n']}日({data['vehicle_date']}):累计{data['cumulative_orders']}单; "
            report_content += "\n"
    
    # 退订描述
    report_content += "\n#### 🔄 退订情况描述\n\n"
    
    # 生成退订描述数据
    refund_desc = generate_refund_description(vehicle_data, state.get('presale_periods', {}))

    
    # CM2日退订数描述
    if refund_desc['cm2_daily']:
        report_content += "**CM2车型日退订数:**\n\n"
        for data in refund_desc['cm2_daily']:
            report_content += f"- 第{data['day_n']}日 ({data['date']}): {data['refunds']}单\n"
    else:
        report_content += "**CM2车型日退订数:** 暂无退订数据\n\n"
    
    # 同期其他车型日退订数对比
    if refund_desc['comparison_daily']:
        report_content += "\n**同期其他车型日退订数对比:**\n\n"
        for vehicle in ['CM0', 'CM1', 'DM0', 'DM1']:
            if vehicle in refund_desc['comparison_daily'] and refund_desc['comparison_daily'][vehicle]:
                report_content += f"- **{vehicle}车型**: "
                for data in refund_desc['comparison_daily'][vehicle]:
                    report_content += f"第{data['day_n']}日({data['vehicle_date']}):{data['refunds']}单; "
                report_content += "\n"
    
    # CM2累计退订数描述
    if refund_desc['cm2_cumulative']:
        report_content += "\n**CM2车型累计退订数:**\n\n"
        for data in refund_desc['cm2_cumulative']:
            report_content += f"- 第{data['day_n']}日 ({data['date']}): 累计{data['cumulative_refunds']}单\n"
    
    # 同期其他车型累计退订数对比
    if refund_desc['comparison_cumulative']:
        report_content += "\n**同期其他车型累计退订数对比:**\n\n"
        for vehicle in ['CM0', 'CM1', 'DM0', 'DM1']:
            if vehicle in refund_desc['comparison_cumulative'] and refund_desc['comparison_cumulative'][vehicle]:
                report_content += f"- **{vehicle}车型**: "
                for data in refund_desc['comparison_cumulative'][vehicle]:
                    report_content += f"第{data['day_n']}日({data['vehicle_date']}):累计{data['cumulative_refunds']}单; "
                report_content += "\n"
    
    # 退订率计算和描述
    report_content += "\n**退订率分析:**\n\n"
    
    # CM2车型退订率
    if refund_desc['cm2_cumulative'] and time_series_desc['cm2_cumulative']:
        cm2_refund_data = refund_desc['cm2_cumulative'][0]
        cm2_order_data = time_series_desc['cm2_cumulative'][0]
        
        cm2_refund_rate = (cm2_refund_data['cumulative_refunds'] / cm2_order_data['cumulative_orders']) * 100
        report_content += f"- **CM2车型**: 第{cm2_refund_data['day_n']}日 ({cm2_refund_data['date']}) 退订率: {cm2_refund_rate:.2f}% (累计退订{cm2_refund_data['cumulative_refunds']}单/累计订单{cm2_order_data['cumulative_orders']}单)\n"
    
    # 同期其他车型退订率对比
    if refund_desc['comparison_cumulative'] and time_series_desc['comparison_cumulative']:
        report_content += "\n**同期其他车型退订率对比:**\n\n"
        for vehicle in ['CM0', 'CM1', 'DM0', 'DM1']:
            if (vehicle in refund_desc['comparison_cumulative'] and 
                refund_desc['comparison_cumulative'][vehicle] and
                vehicle in time_series_desc['comparison_cumulative'] and 
                time_series_desc['comparison_cumulative'][vehicle]):
                
                refund_data = refund_desc['comparison_cumulative'][vehicle][0]
                order_data = time_series_desc['comparison_cumulative'][vehicle][0]
                
                if order_data['cumulative_orders'] > 0:
                    refund_rate = (refund_data['cumulative_refunds'] / order_data['cumulative_orders']) * 100
                    report_content += f"- **{vehicle}车型**: 第{refund_data['day_n']}日({refund_data['vehicle_date']}) 退订率: {refund_rate:.2f}% (累计退订{refund_data['cumulative_refunds']}单/累计订单{order_data['cumulative_orders']}单)\n"
    
    # 前一天数据描述
    report_content += "\n#### 📅 前一天数据分析\n\n"
    
    # 生成前一天数据
    previous_day_data = generate_previous_day_data(vehicle_data, state.get('presale_periods', {}))
    
    # CM2前一天订单数和退订数
    if previous_day_data['cm2_previous_day']:
        cm2_prev = previous_day_data['cm2_previous_day']
        report_content += f"**CM2车型前一天数据 (第{cm2_prev['day_n']}日, {cm2_prev['date']}):**\n\n"
        report_content += f"- 订单数: {cm2_prev['orders']}单\n"
        report_content += f"- 退订数: {cm2_prev['refunds']}单\n"
    
    # 同期其他车型前一天数据对比
    if previous_day_data['comparison_previous_day']:
        report_content += "\n**同期其他车型前一天数据对比:**\n\n"
        for vehicle in ['CM0', 'CM1', 'DM0', 'DM1']:
            if vehicle in previous_day_data['comparison_previous_day']:
                data = previous_day_data['comparison_previous_day'][vehicle]
                report_content += f"- **{vehicle}车型** (第{data['day_n']}日, {data['vehicle_date']}): 订单{data['orders']}单, 退订{data['refunds']}单\n"
    
    # CM2前N-1日累计数据
    if previous_day_data['cm2_cumulative_previous']:
        cm2_cum_prev = previous_day_data['cm2_cumulative_previous']
        report_content += f"\n**CM2车型前{cm2_cum_prev['day_n']}日累计数据:**\n\n"
        report_content += f"- 累计订单数: {cm2_cum_prev['cumulative_orders']}单\n"
        report_content += f"- 累计退订数: {cm2_cum_prev['cumulative_refunds']}单\n"
        
        # 计算前N-1日退订率
        if cm2_cum_prev['cumulative_orders'] > 0:
            previous_refund_rate = (cm2_cum_prev['cumulative_refunds'] / cm2_cum_prev['cumulative_orders']) * 100
            report_content += f"- 退订率: {previous_refund_rate:.2f}%\n"
    
    # 同期其他车型前N-1日累计数据对比
    if previous_day_data['comparison_cumulative_previous']:
        report_content += "\n**同期其他车型前N-1日累计数据对比:**\n\n"
        for vehicle in ['CM0', 'CM1', 'DM0', 'DM1']:
            if vehicle in previous_day_data['comparison_cumulative_previous']:
                data = previous_day_data['comparison_cumulative_previous'][vehicle]
                vehicle_refund_rate = 0
                if data['cumulative_orders'] > 0:
                    vehicle_refund_rate = (data['cumulative_refunds'] / data['cumulative_orders']) * 100
                report_content += f"- **{vehicle}车型** (第{data['day_n']}日): 累计订单{data['cumulative_orders']}单, 累计退订{data['cumulative_refunds']}单, 退订率{vehicle_refund_rate:.2f}%\n"

    # 日环比异常
    report_content += "\n#### 📅 日环比异常检测\n\n"
    if time_series_anomalies_daily:
        report_content += "**🚨 发现日环比异常:**\n\n"
        for i, anomaly in enumerate(time_series_anomalies_daily, 1):
            clean_anomaly = anomaly.replace('[日环比]', '')
            report_content += f"{i}. {clean_anomaly}\n"
    else:
        report_content += "**✅ 日环比正常:** CM2车型日订单量变化均在50%阈值范围内。\n"
    
    # 退订率日环比异常检测
    report_content += "\n#### 📊 退订率日环比异常检测\n\n"
    
    # 获取当前日期(第N日)的退订率数据
    time_series_data = state.get('time_series_data', {})
    if time_series_data and 'cm2_daily_data' in time_series_data:
        cm2_daily_data = time_series_data['cm2_daily_data']
        if len(cm2_daily_data) > 0:
            # 获取最后一天的退订率
            last_day_data = cm2_daily_data.iloc[-1]
            current_refund_rate = 0
            if last_day_data['cumulative_orders'] > 0:
                current_refund_rate = (last_day_data['cumulative_refunds'] / last_day_data['cumulative_orders']) * 100
            
            # 获取前一天的退订率
            previous_refund_rate = 0
            if previous_day_data['cm2_cumulative_previous'] and previous_day_data['cm2_cumulative_previous']['cumulative_orders'] > 0:
                previous_refund_rate = (previous_day_data['cm2_cumulative_previous']['cumulative_refunds'] / previous_day_data['cm2_cumulative_previous']['cumulative_orders']) * 100
            
            # 计算CM2退订率变化幅度
            cm2_refund_rate_change = 0
            if previous_refund_rate > 0:
                cm2_refund_rate_change = ((current_refund_rate - previous_refund_rate) / previous_refund_rate) * 100
            
            # 计算历史车型的退订率日环比变化幅度作为基准
            historical_changes = []
            for vehicle in ['CM0', 'CM1', 'DM0', 'DM1']:
                if vehicle in previous_day_data['comparison_cumulative_previous']:
                    # 获取历史车型当前日期的退订率（模拟第N日）
                    vehicle_current_data = previous_day_data['comparison_cumulative_previous'][vehicle]
                    vehicle_current_refund_rate = 0
                    if vehicle_current_data['cumulative_orders'] > 0:
                        vehicle_current_refund_rate = (vehicle_current_data['cumulative_refunds'] / vehicle_current_data['cumulative_orders']) * 100
                    
                    # 获取历史车型前一天的退订率（模拟第N-1日）
                    # 这里需要计算第N-1日的累计数据
                    if vehicle in vehicle_data and len(vehicle_data[vehicle]) > 0:
                        vehicle_start = pd.to_datetime(state.get('presale_periods', {}).get(vehicle, {}).get('start'))
                        if vehicle_start is not None:
                            target_date_n_minus_1 = vehicle_start + pd.Timedelta(days=previous_day_data['cm2_cumulative_previous']['day_n']-2)
                            
                            # 计算第N-1日的累计订单和退订
                            vehicle_data_copy = vehicle_data[vehicle].copy()
                            vehicle_data_copy['date'] = vehicle_data_copy['Intention_Payment_Time'].dt.date
                            vehicle_orders_n_minus_1 = vehicle_data_copy[pd.to_datetime(vehicle_data_copy['date']) <= target_date_n_minus_1].shape[0]
                            
                            vehicle_refunds_n_minus_1 = 0
                            vehicle_refund_data = vehicle_data[vehicle][vehicle_data[vehicle]['intention_refund_time'].notna()].copy()
                            if len(vehicle_refund_data) > 0:
                                vehicle_refund_data['refund_date'] = vehicle_refund_data['intention_refund_time'].dt.date
                                vehicle_refunds_n_minus_1 = vehicle_refund_data[pd.to_datetime(vehicle_refund_data['refund_date']) <= target_date_n_minus_1].shape[0]
                            
                            vehicle_previous_refund_rate = 0
                            if vehicle_orders_n_minus_1 > 0:
                                vehicle_previous_refund_rate = (vehicle_refunds_n_minus_1 / vehicle_orders_n_minus_1) * 100
                            
                            # 计算历史车型的退订率变化幅度
                            if vehicle_previous_refund_rate > 0:
                                vehicle_change = ((vehicle_current_refund_rate - vehicle_previous_refund_rate) / vehicle_previous_refund_rate) * 100
                                historical_changes.append(abs(vehicle_change))
            
            # 计算历史车型变化幅度的平均值作为基准
            if historical_changes:
                avg_historical_change = sum(historical_changes) / len(historical_changes)
                threshold = max(20, avg_historical_change * 1.5)  # 阈值为20%或历史平均变化的1.5倍，取较大值
                
                # 检测CM2异常
                if abs(cm2_refund_rate_change) > threshold:
                    if cm2_refund_rate_change > 0:
                        report_content += f"**🚨 发现退订率异常骤增:** CM2当前退订率{current_refund_rate:.2f}%，前日退订率{previous_refund_rate:.2f}%，变化幅度{cm2_refund_rate_change:+.1f}%\n"
                        report_content += f"**📊 历史基准:** 历史车型平均变化幅度{avg_historical_change:.1f}%，异常阈值{threshold:.1f}%\n"
                    else:
                        report_content += f"**🚨 发现退订率异常骤降:** CM2当前退订率{current_refund_rate:.2f}%，前日退订率{previous_refund_rate:.2f}%，变化幅度{cm2_refund_rate_change:+.1f}%\n"
                        report_content += f"**📊 历史基准:** 历史车型平均变化幅度{avg_historical_change:.1f}%，异常阈值{threshold:.1f}%\n"
                else:
                    report_content += f"**✅ 退订率日环比正常:** CM2当前退订率{current_refund_rate:.2f}%，前日退订率{previous_refund_rate:.2f}%，变化幅度{cm2_refund_rate_change:+.1f}%\n"
                    report_content += f"**📊 历史基准:** 历史车型平均变化幅度{avg_historical_change:.1f}%，异常阈值{threshold:.1f}%\n"
            else:
                # 如果没有历史数据，使用固定20%阈值
                if abs(cm2_refund_rate_change) > 20:
                    if cm2_refund_rate_change > 0:
                        report_content += f"**🚨 发现退订率异常骤增:** CM2当前退订率{current_refund_rate:.2f}%，前日退订率{previous_refund_rate:.2f}%，变化幅度{cm2_refund_rate_change:+.1f}%\n"
                    else:
                        report_content += f"**🚨 发现退订率异常骤降:** CM2当前退订率{current_refund_rate:.2f}%，前日退订率{previous_refund_rate:.2f}%，变化幅度{cm2_refund_rate_change:+.1f}%\n"
                else:
                    report_content += f"**✅ 退订率日环比正常:** CM2当前退订率{current_refund_rate:.2f}%，前日退订率{previous_refund_rate:.2f}%，变化幅度{cm2_refund_rate_change:+.1f}%\n"
                report_content += f"**📊 使用固定阈值:** 20%（缺少历史车型数据）\n"
        else:
            report_content += "**⚠️ 无法进行退订率日环比检测:** 缺少时间序列数据\n"
    else:
        report_content += "**⚠️ 无法进行退订率日环比检测:** 缺少时间序列数据\n"
    
    # 添加各车型每日退订率对比表格
    report_content += "\n#### 📈 各车型每日退订率对比表\n\n"
    
    # 构建退订率对比表格
    refund_rate_table_data = {}
    max_days = 0
    
    # 处理所有车型的退订率数据
    for vehicle in ['CM2', 'CM0', 'CM1', 'DM0', 'DM1']:
        if vehicle not in vehicle_data or len(vehicle_data[vehicle]) == 0:
            continue
            
        presale_periods = state.get('presale_periods', {})
        if vehicle not in presale_periods:
            continue
            
        vehicle_start = pd.to_datetime(presale_periods[vehicle]['start'])
        
        # 准备车型订单数据
        vehicle_data_copy = vehicle_data[vehicle].copy()
        vehicle_data_copy['date'] = vehicle_data_copy['Intention_Payment_Time'].dt.date
        vehicle_daily = vehicle_data_copy.groupby('date').size().reset_index(name='orders')
        vehicle_daily['date'] = pd.to_datetime(vehicle_daily['date'])
        vehicle_daily = vehicle_daily.sort_values('date')
        vehicle_daily['cumulative_orders'] = vehicle_daily['orders'].cumsum()
        
        # 准备车型退订数据
        vehicle_refund_data = vehicle_data[vehicle][vehicle_data[vehicle]['intention_refund_time'].notna()].copy()
        if len(vehicle_refund_data) > 0:
            vehicle_refund_data['refund_date'] = vehicle_refund_data['intention_refund_time'].dt.date
            vehicle_daily_refund = vehicle_refund_data.groupby('refund_date').size().reset_index(name='refunds')
            vehicle_daily_refund['refund_date'] = pd.to_datetime(vehicle_daily_refund['refund_date'])
            vehicle_daily_refund = vehicle_daily_refund.sort_values('refund_date')
            vehicle_daily_refund['cumulative_refunds'] = vehicle_daily_refund['refunds'].cumsum()
            
            # 合并订单和退订数据
            vehicle_daily = vehicle_daily.merge(vehicle_daily_refund[['refund_date', 'cumulative_refunds']], 
                                              left_on='date', right_on='refund_date', how='left')
            vehicle_daily['cumulative_refunds'] = vehicle_daily['cumulative_refunds'].ffill().fillna(0)
        else:
            vehicle_daily['cumulative_refunds'] = 0
        
        # 计算每日退订率
        vehicle_daily['refund_rate'] = 0.0
        vehicle_daily.loc[vehicle_daily['cumulative_orders'] > 0, 'refund_rate'] = (
            vehicle_daily['cumulative_refunds'] / vehicle_daily['cumulative_orders'] * 100
        )
        
        # 计算从预售开始的天数
        vehicle_daily['days_from_start'] = (vehicle_daily['date'] - vehicle_start).dt.days
        
        # 存储数据
        refund_rate_table_data[vehicle] = {}
        for _, row in vehicle_daily.iterrows():
            day_num = row['days_from_start']
            if day_num >= 0:  # 只包含预售开始后的数据
                refund_rate_table_data[vehicle][day_num] = row['refund_rate']
                max_days = max(max_days, day_num)
    
    # 生成表格（车型作为列，日期作为行）
    if refund_rate_table_data and max_days > 0:
        # 表头
        header = "| 日期 |"
        separator = "|------|"
        vehicles_with_data = []
        for vehicle in ['CM2', 'CM0', 'CM1', 'DM0', 'DM1']:
            if vehicle in refund_rate_table_data:
                vehicles_with_data.append(vehicle)
                header += f" **{vehicle}** |"
                separator += "-------|"
        
        report_content += header + "\n"
        report_content += separator + "\n"
        
        # 表格内容（按日期行展示）
        for day in range(max_days + 1):  # 完整展示所有天数
            row = f"| 第{day}日 |"
            for vehicle in vehicles_with_data:
                if day in refund_rate_table_data[vehicle]:
                    rate = refund_rate_table_data[vehicle][day]
                    row += f" {rate:.2f}% |"
                else:
                    row += " - |"
            report_content += row + "\n"
        
        report_content += "\n*注：表格显示各车型从预售开始每日的累计退订率，'-' 表示该日无数据*\n\n"
        
        # 生成归一化时间对比表格
        report_content += "#### 📈 各车型归一化预售周期退订率对比表\n\n"
        
        # 计算每个车型的预售周期长度和归一化数据
        normalized_data = {}
        presale_periods = state.get('presale_periods', {})
        
        for vehicle in vehicles_with_data:
            vehicle_days = list(refund_rate_table_data[vehicle].keys())
            if vehicle_days and vehicle in presale_periods:
                # 使用预售周期定义计算总天数
                vehicle_start = pd.to_datetime(presale_periods[vehicle]['start'])
                vehicle_end = pd.to_datetime(presale_periods[vehicle]['end'])
                total_presale_days = (vehicle_end - vehicle_start).days
                
                normalized_data[vehicle] = {}
                
                # 为每个百分比点计算对应的天数和退订率
                max_available_day = max(vehicle_days) if vehicle_days else 0
                max_available_pct = int((max_available_day / total_presale_days) * 100) if total_presale_days > 0 else 0
                
                for pct in range(0, 101, 10):  # 0%, 10%, 20%, ..., 100%
                    target_day = int(total_presale_days * pct / 100)
                    
                    # 如果目标百分比超出当前可用数据范围，跳过
                    if pct > max_available_pct and vehicle == 'CM2':
                        continue  # 不添加数据，后续会显示为 "-"
                    
                    # 找到最接近的有数据的天数
                    closest_day = None
                    min_diff = float('inf')
                    for day in vehicle_days:
                        if day <= target_day:  # 只考虑不超过目标天数的数据
                            diff = abs(day - target_day)
                            if diff < min_diff:
                                min_diff = diff
                                closest_day = day
                    
                    # 如果没有找到不超过目标天数的数据，取最小的天数
                    if closest_day is None and vehicle_days:
                        closest_day = min([d for d in vehicle_days if d >= 0])
                    
                    if closest_day is not None:
                        normalized_data[vehicle][pct] = refund_rate_table_data[vehicle][closest_day]
        
        # 生成归一化表格
        if normalized_data:
            # 表头
            norm_header = "| 预售进度 |"
            norm_separator = "|-------|"
            for vehicle in vehicles_with_data:
                if vehicle in normalized_data:
                    norm_header += f" **{vehicle}** |"
                    norm_separator += "-------|"
            
            report_content += norm_header + "\n"
            report_content += norm_separator + "\n"
            
            # 表格内容
            for pct in range(0, 101, 10):
                row = f"| {pct}% |"
                for vehicle in vehicles_with_data:
                    if vehicle in normalized_data and pct in normalized_data[vehicle]:
                        rate = normalized_data[vehicle][pct]
                        row += f" {rate:.2f}% |"
                    else:
                        row += " - |"
                report_content += row + "\n"
            
            report_content += "\n*注：表格显示各车型在预售周期不同进度下的累计退订率，便于跨车型对比*\n"
        else:
            report_content += "**⚠️ 无法生成归一化对比表:** 缺少归一化数据\n"
    else:
        report_content += "**⚠️ 无法生成退订率对比表:** 缺少车型数据\n"
    
    # 同周期对比异常
    report_content += "\n#### 🔄 同周期对比异常检测\n\n"
    if time_series_anomalies_period:
        report_content += "**🚨 发现同周期对比异常:**\n\n"
        report_content += "| 序号 | 日期 | CM2订单数 | 对比车型 | 对比车型订单数 | 变化幅度 | 异常类型 |\n"
        report_content += "|------|------|----------|----------|----------------|----------|----------|\n"
        for i, anomaly in enumerate(time_series_anomalies_period, 1):
            clean_anomaly = anomaly.replace('[同周期对比]', '')
            
            # 解析异常信息
            parts = clean_anomaly.split('：')
            if len(parts) >= 2:
                desc_part = parts[0]
                data_part = parts[1]
                
                # 提取日期
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', desc_part)
                date = date_match.group(1) if date_match else ""
                
                # 提取对比车型
                vehicle_match = re.search(r'相对(CM\d|DM\d)同期', desc_part)
                compare_vehicle = vehicle_match.group(1) if vehicle_match else ""
                
                # 提取异常类型并添加emoji
                if "骤增" in desc_part:
                    anomaly_type = "📈 骤增"
                else:
                    anomaly_type = "📉 骤降"
                
                # 提取数据
                cm2_match = re.search(r'CM2为(\d+)单', data_part)
                compare_match = re.search(r'{}同期为(\d+)单'.format(compare_vehicle), data_part)
                change_match = re.search(r'变化幅度([+-]?\d+\.\d+)%', data_part)
                
                cm2_orders = cm2_match.group(1) if cm2_match else ""
                compare_orders = compare_match.group(1) if compare_match else ""
                change_rate = change_match.group(1) + "%" if change_match else ""
                
                report_content += f"| {i} | {date} | {cm2_orders} | {compare_vehicle} | {compare_orders} | {change_rate} | {anomaly_type} |\n"
    else:
        report_content += "**✅ 同周期对比正常:** CM2相对于历史车型同期订单量变化均在100%阈值范围内。\n"
    
    # 累计同周期对比异常
    report_content += "\n#### 📊 累计同周期对比异常检测\n\n"
    if time_series_anomalies_cumulative:
        report_content += "**🚨 发现累计同周期对比异常:**\n\n"
        report_content += "| 序号 | 截至日期 | CM2累计订单数 | 对比车型 | 对比车型累计订单数 | 变化幅度 | 异常类型 |\n"
        report_content += "|------|----------|---------------|----------|-------------------|----------|----------|\n"
        for i, anomaly in enumerate(time_series_anomalies_cumulative, 1):
            clean_anomaly = anomaly.replace('[累计同周期对比]', '')
            
            # 解析异常信息
            parts = clean_anomaly.split('：')
            if len(parts) >= 2:
                desc_part = parts[0]
                data_part = parts[1]
                
                # 提取日期
                date_match = re.search(r'截至(\d{4}-\d{2}-\d{2})', desc_part)
                date = date_match.group(1) if date_match else ""
                
                # 提取对比车型
                vehicle_match = re.search(r'相对(CM\d|DM\d)同期', desc_part)
                compare_vehicle = vehicle_match.group(1) if vehicle_match else ""
                
                # 提取异常类型并添加emoji
                if "骤增" in desc_part:
                    anomaly_type = "📈 骤增"
                else:
                    anomaly_type = "📉 骤降"
                
                # 提取数据
                cm2_match = re.search(r'CM2累计(\d+)单', data_part)
                compare_match = re.search(r'{}同期累计(\d+)单'.format(compare_vehicle), data_part)
                change_match = re.search(r'变化幅度([+-]?\d+\.\d+)%', data_part)
                
                cm2_orders = cm2_match.group(1) if cm2_match else ""
                compare_orders = compare_match.group(1) if compare_match else ""
                change_rate = change_match.group(1) + "%" if change_match else ""
                
                report_content += f"| {i} | {date} | {cm2_orders} | {compare_vehicle} | {compare_orders} | {change_rate} | {anomaly_type} |\n"
    else:
        report_content += "**✅ 累计同周期对比正常:** CM2累计订单量相对于历史车型同期变化均在100%阈值范围内。\n"
    
    report_content += "\n## 检查说明\n\n"
    report_content += "### 📊 历史平均对比\n"
    report_content += "- **地区分布异常**: 检查CM2相对于历史车型(CM0, DM0, CM1, DM1)平均值的各地区订单占比变化超过20%的情况，且该地区占比超过1%\n"
    report_content += "- **渠道结构异常**: 检查CM2相对于历史车型平均值的渠道销量占比变化超过15%的情况，且该渠道占比超过1%\n"
    report_content += "- **人群结构异常**: 检查CM2相对于历史车型平均值的性别比例和年龄段结构变化超过10%的情况\n\n"
    report_content += "### 🔄 CM1直接对比\n"
    report_content += "- **地区分布异常**: 检查CM2相对于CM1车型的各地区订单占比变化超过20%的情况，且该地区占比超过1%\n"
    report_content += "- **渠道结构异常**: 检查CM2相对于CM1车型的渠道销量占比变化超过15%的情况，且该渠道占比超过1%\n"
    report_content += "- **人群结构异常**: 检查CM2相对于CM1车型的性别比例和年龄段结构变化超过10%的情况\n\n"
    report_content += "### 📈 同比/环比检测\n"
    report_content += "- **日环比异常**: 检查CM2车型内部相邻日期订单量变化超过50%的情况\n"
    report_content += "- **同周期对比异常**: 检查CM2相对于历史车型(CM0, CM1, DM0, DM1)相同相对天数的订单量变化超过50%的情况\n"
    report_content += "- **累计同周期对比异常**: 检查CM2累计订单量相对于历史车型同期累计订单量变化超过50%的情况\n"
    
    # 保存报告
    report_path = "/Users/zihao_/Documents/github/W35_workflow/structure_check_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"结构检查报告已生成: {report_path}")
    print(f"\n📋 结构检查报告已生成: {report_path}")
    
    # 更新状态
    state["structure_anomalies"] = anomalies
    state["structure_report_path"] = report_path

# 创建工作流图
def create_workflow():
    """
    创建LangGraph异常检测工作流
    """
    workflow = StateGraph(WorkflowState)
    
    # 添加节点
    workflow.add_node("anomaly_detection", anomaly_detection_node)
    workflow.add_node("structure_check", structure_check_node)
    workflow.add_node("generate_complete_report", lambda state: generate_complete_report(state) or state)
    workflow.add_node("update_readme", update_readme_mermaid_node)
    
    # 设置入口点
    workflow.set_entry_point("anomaly_detection")
    
    # 添加边
    workflow.add_edge("anomaly_detection", "structure_check")
    workflow.add_edge("structure_check", "generate_complete_report")
    workflow.add_edge("generate_complete_report", "update_readme")
    workflow.add_edge("update_readme", END)
    
    return workflow.compile()

# 主函数
def main():
    """
    主函数 - 运行工作流
    """
    logger.info("启动LangGraph异常检测工作流...")
    
    # 创建工作流
    app = create_workflow()
    
    # 初始化状态
    initial_state = {
        "data": None,
        "data_path": "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet",
        "integrity_check_results": {},
        "errors": [],
        "status": "initialized",
        "metadata": {}
    }
    
    # 运行工作流
    try:
        result = app.invoke(initial_state)
        
        if result["status"] == "failed":
            logger.error("工作流执行失败")
            for error in result["errors"]:
                logger.error(f"错误: {error}")
        else:
            logger.info("工作流执行成功")
            logger.info(f"最终状态: {result['status']}")
            
    except Exception as e:
        logger.error(f"工作流执行过程中发生异常: {str(e)}")

if __name__ == "__main__":
    main()
