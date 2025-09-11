#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
销售代理特定条件分析脚本
筛选2025-09-09浙闽赣区的销售代理匹配数据并生成分析报告
"""

import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据路径配置
DATA_PATH = "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"
SALES_INFO_PATH = "/Users/zihao_/Documents/coding/dataset/formatted/sales_info_data.json"
REPORT_PATH = "/Users/zihao_/Documents/github/W35_workflow/sales_agent_specific_analysis_report.md"

def load_data() -> Tuple[pd.DataFrame, List[Dict]]:
    """
    加载订单数据和销售代理数据
    """
    logger.info("开始加载数据...")
    
    # 加载订单数据
    try:
        df = pd.read_parquet(DATA_PATH)
        logger.info(f"订单数据加载成功，共{len(df):,}条记录")
    except Exception as e:
        logger.error(f"加载订单数据失败: {str(e)}")
        raise
    
    # 加载销售代理数据
    try:
        with open(SALES_INFO_PATH, 'r', encoding='utf-8') as f:
            sales_info_json = json.load(f)
            sales_info_data = sales_info_json.get('data', [])
        logger.info(f"销售代理数据加载成功，共{len(sales_info_data):,}条记录")
    except Exception as e:
        logger.error(f"加载销售代理数据失败: {str(e)}")
        raise
    
    return df, sales_info_data

def filter_target_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    筛选目标数据：2025-09-09 且 浙闽赣区
    """
    logger.info("开始筛选目标数据...")
    
    # 确保日期字段为datetime类型
    df['Intention_Payment_Time'] = pd.to_datetime(df['Intention_Payment_Time'])
    
    # 筛选条件
    target_date = pd.to_datetime('2025-09-09')
    target_region = '浙闽赣区'
    
    # 应用筛选条件
    filtered_df = df[
        (df['Intention_Payment_Time'].dt.date == target_date.date()) &
        (df['Parent Region Name'] == target_region)
    ].copy()
    
    logger.info(f"筛选后数据量: {len(filtered_df):,}条记录")
    
    if len(filtered_df) == 0:
        logger.warning("筛选后无数据，请检查筛选条件")
    
    return filtered_df



def build_sales_agent_dict(sales_info_data: List[Dict]) -> Dict:
    """
    构建销售代理字典，包含经销商信息
    """
    logger.info("构建销售代理字典...")
    
    sales_agents_dict = {}
    valid_count = 0
    
    for item in sales_info_data:
        if isinstance(item, dict):
            member_name = str(item.get('Member Name', '')).strip() if item.get('Member Name') else ''
            member_code = str(item.get('Member Code', '')).strip() if item.get('Member Code') else ''
            id_card = str(item.get('Id Card', '')).strip() if item.get('Id Card') else ''
            dealer_name_fc = str(item.get('Dealer Name Fc', '')).strip() if item.get('Dealer Name Fc') else ''
            
            if member_name and member_code and id_card:
                key = (member_name, member_code, id_card)
                sales_agents_dict[key] = {
                    'dealer_name_fc': dealer_name_fc,
                    'dealer_type': item.get('Dealer_type', ''),
                    'status': item.get('Status', ''),
                    'phone': item.get('Phone', '')
                }
                valid_count += 1
    
    logger.info(f"有效销售代理记录: {valid_count:,}条")
    return sales_agents_dict

def analyze_sales_agent_matches(filtered_df: pd.DataFrame, sales_agents_dict: Dict) -> Dict:
    """
    分析销售代理匹配情况
    """
    logger.info("开始分析销售代理匹配情况...")
    
    if len(filtered_df) == 0:
        return {
            'total_orders': 0,
            'matched_orders': 0,
            'match_ratio': 0.0,
            'matched_data': pd.DataFrame(),
            'unmatched_data': pd.DataFrame(),
            'dealer_summary': pd.DataFrame(),
            'agent_summary': pd.DataFrame()
        }
    
    # 预处理字段
    filtered_df = filtered_df.copy()
    filtered_df['clean_store_agent_name'] = filtered_df['Store Agent Name'].fillna('').astype(str).str.strip()
    filtered_df['clean_store_agent_id'] = filtered_df['Store Agent Id'].fillna('').astype(str).str.strip()
    filtered_df['clean_buyer_identity'] = filtered_df['Buyer Identity No'].fillna('').astype(str).str.strip()
    
    # 创建组合字段用于匹配
    filtered_df['agent_combo'] = list(zip(
        filtered_df['clean_store_agent_name'],
        filtered_df['clean_store_agent_id'],
        filtered_df['clean_buyer_identity']
    ))
    
    # 执行匹配并添加经销商信息
    def get_dealer_info(combo):
        if combo in sales_agents_dict:
            return sales_agents_dict[combo]['dealer_name_fc']
        return ''
    
    filtered_df['is_matched'] = filtered_df['agent_combo'].apply(lambda x: x in sales_agents_dict)
    filtered_df['dealer_name_fc'] = filtered_df['agent_combo'].apply(get_dealer_info)
    
    # 统计结果
    total_orders = len(filtered_df)
    matched_orders = filtered_df['is_matched'].sum()
    match_ratio = matched_orders / total_orders if total_orders > 0 else 0.0
    
    # 分离匹配和未匹配数据
    matched_data = filtered_df[filtered_df['is_matched']].copy()
    unmatched_data = filtered_df[~filtered_df['is_matched']].copy()
    
    # 经销商汇总分析（基于匹配的数据）
    if len(matched_data) > 0 and 'dealer_name_fc' in matched_data.columns:
        dealer_summary = matched_data.groupby('dealer_name_fc').agg({
            'Order Number': 'count'
        }).reset_index()
        dealer_summary.columns = ['经销商名称', '订单总数']
        dealer_summary = dealer_summary.sort_values('订单总数', ascending=False)
    else:
        dealer_summary = pd.DataFrame(columns=['经销商名称', '订单总数'])
    
    # 销售代理汇总分析（仅匹配的数据）
    if len(matched_data) > 0:
        agent_summary = matched_data.groupby(['Store Agent Name', 'Store Agent Id', 'Buyer Identity No']).agg({
            'Order Number': 'count',
            'dealer_name_fc': 'first'
        }).reset_index()
        agent_summary.columns = ['销售代理姓名', '销售代理ID', '身份证号', '订单数量', '关联经销商']
        agent_summary = agent_summary.sort_values('订单数量', ascending=False)
    else:
        agent_summary = pd.DataFrame(columns=['销售代理姓名', '销售代理ID', '身份证号', '订单数量', '关联经销商'])
    
    logger.info(f"匹配分析完成 - 总订单: {total_orders:,}, 匹配订单: {matched_orders:,}, 匹配率: {match_ratio:.2%}")
    
    return {
        'total_orders': total_orders,
        'matched_orders': matched_orders,
        'match_ratio': match_ratio,
        'matched_data': matched_data,
        'unmatched_data': unmatched_data,
        'dealer_summary': dealer_summary,
        'agent_summary': agent_summary
    }

def generate_analysis_report(analysis_results: Dict, target_date: str = '2025-09-09', target_region: str = '浙闽赣区'):
    """
    生成分析报告
    """
    logger.info("生成分析报告...")
    
    report_content = f"""# 销售代理特定条件分析报告

**分析时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 筛选条件

- **目标日期：** {target_date}
- **目标地区：** {target_region}
- **匹配条件：**
  - Store Agent Name = Member Name
  - Store Agent Id = Member Code  
  - Buyer Identity No = Id Card

## 整体分析结果

- **总订单数：** {analysis_results['total_orders']:,}
- **匹配订单数：** {analysis_results['matched_orders']:,}
- **匹配率：** {analysis_results['match_ratio']:.2%}
- **未匹配订单数：** {analysis_results['total_orders'] - analysis_results['matched_orders']:,}

"""

    # 经销商分析
    if len(analysis_results['dealer_summary']) > 0:
        report_content += "\n## 经销商分析\n\n"
        report_content += "### 各经销商订单及匹配情况\n\n"
        
        dealer_table = analysis_results['dealer_summary'].to_string(index=False)
        report_content += f"```\n{dealer_table}\n```\n\n"
        
        # 经销商关键洞察
        top_dealer = analysis_results['dealer_summary'].iloc[0]
        
        report_content += "### 经销商关键洞察\n\n"
        report_content += f"- **订单量最大经销商：** {top_dealer['经销商名称']}（{top_dealer['订单总数']:,}单）\n"
        report_content += f"- **涉及经销商总数：** {len(analysis_results['dealer_summary'])}家\n"
    
    # 销售代理分析
    if len(analysis_results['agent_summary']) > 0:
        report_content += "\n## 销售代理分析\n\n"
        report_content += "### 匹配的销售代理详情\n\n"
        
        agent_table = analysis_results['agent_summary'].to_string(index=False)
        report_content += f"```\n{agent_table}\n```\n\n"
        
        # 销售代理关键洞察
        total_agents = len(analysis_results['agent_summary'])
        top_agent = analysis_results['agent_summary'].iloc[0] if total_agents > 0 else None
        multi_order_agents = analysis_results['agent_summary'][analysis_results['agent_summary']['订单数量'] > 1]
        
        report_content += "### 销售代理关键洞察\n\n"
        report_content += f"- **匹配的销售代理总数：** {total_agents}人\n"
        
        if top_agent is not None:
            report_content += f"- **订单量最多的销售代理：** {top_agent['销售代理姓名']}（{top_agent['订单数量']}单）\n"
        
        report_content += f"- **多订单销售代理：** {len(multi_order_agents)}人（订单数>1）\n"
        
        if len(multi_order_agents) > 0:
            avg_orders = multi_order_agents['订单数量'].mean()
            report_content += f"- **多订单销售代理平均订单数：** {avg_orders:.1f}单\n"
    
    # 数据质量分析
    report_content += "\n## 数据质量分析\n\n"
    
    if analysis_results['total_orders'] > 0:
        unmatched_ratio = (analysis_results['total_orders'] - analysis_results['matched_orders']) / analysis_results['total_orders']
        
        if unmatched_ratio > 0.5:
            quality_level = "较低"
            quality_desc = "超过50%的订单未能匹配到销售代理信息"
        elif unmatched_ratio > 0.2:
            quality_level = "中等"
            quality_desc = "20%-50%的订单未能匹配到销售代理信息"
        else:
            quality_level = "较高"
            quality_desc = "大部分订单都能匹配到销售代理信息"
        
        report_content += f"- **数据匹配质量：** {quality_level}\n"
        report_content += f"- **质量描述：** {quality_desc}\n"
        report_content += f"- **未匹配率：** {unmatched_ratio:.1%}\n"
    else:
        report_content += "- **数据匹配质量：** 无法评估（无目标数据）\n"
    
    # 建议措施
    report_content += "\n## 建议措施\n\n"
    
    if analysis_results['total_orders'] == 0:
        report_content += "- 检查筛选条件是否正确，确认目标日期和地区的数据是否存在\n"
        report_content += "- 验证数据源的完整性和时效性\n"
    elif analysis_results['match_ratio'] < 0.3:
        report_content += "- 检查销售代理数据的完整性和准确性\n"
        report_content += "- 验证匹配字段的数据格式是否一致\n"
        report_content += "- 考虑优化销售代理信息的录入流程\n"
    else:
        report_content += "- 继续保持当前的数据质量水平\n"
        report_content += "- 定期更新销售代理信息数据\n"
        report_content += "- 建立数据质量监控机制\n"
    
    report_content += "\n---\n"
    report_content += f"\n*报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
    
    # 保存报告
    try:
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logger.info(f"分析报告已保存至: {REPORT_PATH}")
    except Exception as e:
        logger.error(f"保存报告失败: {str(e)}")
        raise
    
    return report_content

def main():
    """
    主函数
    """
    try:
        logger.info("=== 销售代理特定条件分析开始 ===")
        
        # 1. 加载数据
        df, sales_info_data = load_data()
        
        # 2. 筛选目标数据
        filtered_df = filter_target_data(df)
        
        # 3. 构建销售代理字典
        sales_agents_dict = build_sales_agent_dict(sales_info_data)
        
        # 4. 执行匹配分析
        analysis_results = analyze_sales_agent_matches(filtered_df, sales_agents_dict)
        
        # 5. 生成分析报告
        report_content = generate_analysis_report(analysis_results)
        
        # 6. 输出关键结果
        logger.info("=== 分析结果摘要 ===")
        logger.info(f"总订单数: {analysis_results['total_orders']:,}")
        logger.info(f"匹配订单数: {analysis_results['matched_orders']:,}")
        logger.info(f"匹配率: {analysis_results['match_ratio']:.2%}")
        logger.info(f"匹配的销售代理数: {len(analysis_results['agent_summary'])}")
        logger.info(f"涉及经销商数: {len(analysis_results['dealer_summary'])}")
        
        logger.info("=== 销售代理特定条件分析完成 ===")
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"分析过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()