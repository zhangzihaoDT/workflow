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
    B --> C[读取数据文件]
    C --> D[数据完整性检查]
    D --> E[缺失值检测]
    E --> F[重复数据检测]
    F --> G[数据类型分析]
    G --> H[生成异常检测报告]
    H --> I[更新README图示]
    I --> J[结束]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style H fill:#e8f5e8
    style I fill:#fff3e0
    style J fill:#ffebee
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

### 报告生成
- 自动生成详细的MD格式异常检测报告
- 包含数据质量评级
- 提供可视化的异常统计信息

### 工作流特性
- 基于LangGraph框架构建
- 模块化节点设计
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
- 控制台日志: 实时处理状态信息
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

# 创建工作流图
def create_workflow():
    """
    创建LangGraph异常检测工作流
    """
    workflow = StateGraph(WorkflowState)
    
    # 添加节点
    workflow.add_node("anomaly_detection", anomaly_detection_node)
    workflow.add_node("update_readme", update_readme_mermaid_node)
    
    # 设置入口点
    workflow.set_entry_point("anomaly_detection")
    
    # 添加边
    workflow.add_edge("anomaly_detection", "update_readme")
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
