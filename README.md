# W35 异常检测工作流

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
