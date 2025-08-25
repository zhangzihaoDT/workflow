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
        
        # 生成结构检查报告
        generate_structure_report(state, vehicle_data, anomalies)
        
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
    
    # 地区分布异常
    report_content += "### 🌍 地区分布异常检测\n\n"
    
    # 历史对比结果
    report_content += "#### 📊 CM2 vs 历史平均对比\n\n"
    if region_anomalies_hist:
        report_content += "**🚨 发现地区分布异常:**\n\n"
        for i, anomaly in enumerate(region_anomalies_hist, 1):
            clean_anomaly = anomaly.replace('[历史对比]', '')
            report_content += f"{i}. {clean_anomaly}\n"
    else:
        report_content += "**✅ 地区分布正常:** 所有地区订单占比变化均在20%阈值范围内。\n"
    
    # CM1对比结果
    report_content += "\n#### 🔄 CM2 vs CM1直接对比\n\n"
    if region_anomalies_cm1:
        report_content += "**🚨 发现地区分布异常:**\n\n"
        for i, anomaly in enumerate(region_anomalies_cm1, 1):
            clean_anomaly = anomaly.replace('[CM1对比]', '')
            report_content += f"{i}. {clean_anomaly}\n"
    else:
        report_content += "**✅ 地区分布正常:** 相比CM1，所有地区订单占比变化均在20%阈值范围内。\n"
    
    # 渠道结构异常
    report_content += "\n### 🛒 渠道结构异常检测\n\n"
    
    # 历史对比结果
    report_content += "#### 📊 CM2 vs 历史平均对比\n\n"
    if channel_anomalies_hist:
        report_content += "**🚨 发现渠道结构异常:**\n\n"
        for i, anomaly in enumerate(channel_anomalies_hist, 1):
            clean_anomaly = anomaly.replace('[历史对比]', '')
            report_content += f"{i}. {clean_anomaly}\n"
    else:
        report_content += "**✅ 渠道结构正常:** 所有渠道销量占比变化均在15%阈值范围内。\n"
    
    # CM1对比结果
    report_content += "\n#### 🔄 CM2 vs CM1直接对比\n\n"
    if channel_anomalies_cm1:
        report_content += "**🚨 发现渠道结构异常:**\n\n"
        for i, anomaly in enumerate(channel_anomalies_cm1, 1):
            clean_anomaly = anomaly.replace('[CM1对比]', '')
            report_content += f"{i}. {clean_anomaly}\n"
    else:
        report_content += "**✅ 渠道结构正常:** 相比CM1，所有渠道销量占比变化均在15%阈值范围内。\n"
    
    # 人群结构异常
    report_content += "\n### 👥 人群结构异常检测\n\n"
    
    # 历史对比结果
    report_content += "#### 📊 CM2 vs 历史平均对比\n\n"
    if demographic_anomalies_hist:
        report_content += "**🚨 发现人群结构异常:**\n\n"
        for i, anomaly in enumerate(demographic_anomalies_hist, 1):
            clean_anomaly = anomaly.replace('[历史对比]', '')
            report_content += f"{i}. {clean_anomaly}\n"
    else:
        report_content += "**✅ 人群结构正常:** 所有性别比例和年龄段结构变化均在10%阈值范围内。\n"
    
    # CM1对比结果
    report_content += "\n#### 🔄 CM2 vs CM1直接对比\n\n"
    if demographic_anomalies_cm1:
        report_content += "**🚨 发现人群结构异常:**\n\n"
        for i, anomaly in enumerate(demographic_anomalies_cm1, 1):
            clean_anomaly = anomaly.replace('[CM1对比]', '')
            report_content += f"{i}. {clean_anomaly}\n"
    else:
        report_content += "**✅ 人群结构正常:** 相比CM1，所有性别比例和年龄段结构变化均在10%阈值范围内。\n"
    
    report_content += "\n## 检查说明\n\n"
    report_content += "### 📊 历史平均对比\n"
    report_content += "- **地区分布异常**: 检查CM2相对于历史车型(CM0, DM0, CM1, DM1)平均值的各地区订单占比变化超过20%的情况，且该地区占比超过1%\n"
    report_content += "- **渠道结构异常**: 检查CM2相对于历史车型平均值的渠道销量占比变化超过15%的情况，且该渠道占比超过1%\n"
    report_content += "- **人群结构异常**: 检查CM2相对于历史车型平均值的性别比例和年龄段结构变化超过10%的情况\n\n"
    report_content += "### 🔄 CM1直接对比\n"
    report_content += "- **地区分布异常**: 检查CM2相对于CM1车型的各地区订单占比变化超过20%的情况，且该地区占比超过1%\n"
    report_content += "- **渠道结构异常**: 检查CM2相对于CM1车型的渠道销量占比变化超过15%的情况，且该渠道占比超过1%\n"
    report_content += "- **人群结构异常**: 检查CM2相对于CM1车型的性别比例和年龄段结构变化超过10%的情况\n"
    
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
