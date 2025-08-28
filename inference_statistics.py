#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CM2车型纯电/增程比例推断统计脚本

基于小样本监督学习和分布校准的方法，推断CM2全量订单的纯电/增程比例
使用三个数据样本：
1. CM1全量纯电用户（参考基准）
2. CM2全量混合用户（待推断目标）
3. CM2小样本标签用户（监督信号）
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CM2InferenceEngine:
    """
    CM2车型纯电/增程比例推断引擎
    """
    
    def __init__(self, data_path, business_definition_path):
        self.data_path = data_path
        self.business_definition_path = business_definition_path
        self.df = None
        self.business_periods = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def load_data(self):
        """加载数据和业务定义"""
        logger.info("加载数据...")
        
        # 加载主数据
        self.df = pd.read_parquet(self.data_path)
        logger.info(f"成功加载数据，数据形状: {self.df.shape}")
        
        # 加载业务周期定义
        with open(self.business_definition_path, 'r') as f:
            self.business_periods = json.load(f)
        logger.info("成功加载业务周期定义")
        
    def filter_vehicle_data(self):
        """根据业务周期筛选CM1和CM2车型数据"""
        logger.info("筛选CM1和CM2车型数据...")
        
        # 转换时间列
        self.df['Intention_Payment_Time'] = pd.to_datetime(self.df['Intention_Payment_Time'])
        
        # 筛选CM1数据
        cm1_start = pd.to_datetime(self.business_periods['CM1']['start'])
        cm1_end = pd.to_datetime(self.business_periods['CM1']['end'])
        cm1_mask = (self.df['Intention_Payment_Time'] >= cm1_start) & \
                   (self.df['Intention_Payment_Time'] <= cm1_end)
        self.cm1_data = self.df[cm1_mask].copy()
        
        # 筛选CM2数据
        cm2_start = pd.to_datetime(self.business_periods['CM2']['start'])
        cm2_end = pd.to_datetime(self.business_periods['CM2']['end'])
        cm2_mask = (self.df['Intention_Payment_Time'] >= cm2_start) & \
                   (self.df['Intention_Payment_Time'] <= cm2_end)
        self.cm2_data = self.df[cm2_mask].copy()
        
        logger.info(f"CM1数据量: {len(self.cm1_data)}")
        logger.info(f"CM2数据量: {len(self.cm2_data)}")
        
    def prepare_samples(self):
        """准备三个数据样本"""
        logger.info("准备数据样本...")
        
        # 选择特征列
        feature_cols = ['buyer_age', 'order_gender', 'License City', 'Parent Region Name']
        
        # 1. CM1全量纯电用户（假设CM1全部为纯电）
        self.sample1_cm1_pure = self.cm1_data[feature_cols].copy()
        self.sample1_cm1_pure['vehicle_type'] = 'pure_electric'  # CM1假设全为纯电
        self.sample1_cm1_pure['sample_type'] = 'cm1_pure_reference'
        
        # 2. CM2全量混合用户（待推断）
        self.sample2_cm2_all = self.cm2_data[feature_cols].copy()
        self.sample2_cm2_all['sample_type'] = 'cm2_all_mixed'
        
        # 3. CM2小样本标签用户（有pre_vehicle_model_type标签）
        cm2_labeled_mask = self.cm2_data['pre_vehicle_model_type'].notna()
        self.sample3_cm2_labeled = self.cm2_data[cm2_labeled_mask][feature_cols + ['pre_vehicle_model_type']].copy()
        
        # 将pre_vehicle_model_type映射为纯电/增程
        # 这里需要根据实际的标签值进行映射，假设包含'纯电'或'EV'的为纯电，其他为增程
        def map_vehicle_type(pre_type):
            if pd.isna(pre_type):
                return None
            pre_type_str = str(pre_type).lower()
            if '纯电' in pre_type_str or 'ev' in pre_type_str or 'bev' in pre_type_str:
                return 'pure_electric'
            else:
                return 'range_extender'
        
        self.sample3_cm2_labeled['vehicle_type'] = self.sample3_cm2_labeled['pre_vehicle_model_type'].apply(map_vehicle_type)
        self.sample3_cm2_labeled['sample_type'] = 'cm2_labeled_sample'
        
        # 移除无法映射的数据
        self.sample3_cm2_labeled = self.sample3_cm2_labeled[self.sample3_cm2_labeled['vehicle_type'].notna()]
        
        logger.info(f"样本1 - CM1全量纯电用户: {len(self.sample1_cm1_pure)}")
        logger.info(f"样本2 - CM2全量混合用户: {len(self.sample2_cm2_all)}")
        logger.info(f"样本3 - CM2小样本标签用户: {len(self.sample3_cm2_labeled)}")
        
        if len(self.sample3_cm2_labeled) == 0:
            raise ValueError("CM2小样本标签用户数量为0，无法进行监督学习")
            
        # 计算小样本标签比例
        labeled_ratio = len(self.sample3_cm2_labeled) / len(self.sample2_cm2_all)
        logger.info(f"CM2标签样本占比: {labeled_ratio:.2%}")
        
    def feature_engineering(self):
        """特征工程：编码和标准化"""
        logger.info("进行特征工程...")
        
        # 合并所有数据进行统一编码
        all_data = pd.concat([
            self.sample1_cm1_pure[['buyer_age', 'order_gender', 'License City', 'Parent Region Name']],
            self.sample2_cm2_all[['buyer_age', 'order_gender', 'License City', 'Parent Region Name']],
            self.sample3_cm2_labeled[['buyer_age', 'order_gender', 'License City', 'Parent Region Name']]
        ], ignore_index=True)
        
        # 处理分类特征
        categorical_cols = ['order_gender', 'License City', 'Parent Region Name']
        for col in categorical_cols:
            # 处理分类变量的缺失值
            if all_data[col].dtype.name == 'category':
                # 如果是分类类型，先添加'Unknown'到categories中
                if 'Unknown' not in all_data[col].cat.categories:
                    all_data[col] = all_data[col].cat.add_categories(['Unknown'])
                all_data[col] = all_data[col].fillna('Unknown')
            else:
                # 如果不是分类类型，直接填充
                all_data[col] = all_data[col].fillna('Unknown')
            
            # 标签编码
            le = LabelEncoder()
            all_data[col + '_encoded'] = le.fit_transform(all_data[col])
            self.label_encoders[col] = le
        
        # 处理数值特征
        all_data['buyer_age'] = pd.to_numeric(all_data['buyer_age'], errors='coerce')
        all_data['buyer_age'] = all_data['buyer_age'].fillna(all_data['buyer_age'].median())
        
        # 准备特征矩阵
        feature_cols = ['buyer_age'] + [col + '_encoded' for col in categorical_cols]
        
        # 分别为三个样本准备特征
        n1 = len(self.sample1_cm1_pure)
        n2 = len(self.sample2_cm2_all)
        n3 = len(self.sample3_cm2_labeled)
        
        self.X_cm1 = all_data.iloc[:n1][feature_cols].values
        self.X_cm2_all = all_data.iloc[n1:n1+n2][feature_cols].values
        self.X_cm2_labeled = all_data.iloc[n1+n2:n1+n2+n3][feature_cols].values
        
        # 标准化特征
        X_all = np.vstack([self.X_cm1, self.X_cm2_all, self.X_cm2_labeled])
        self.scaler.fit(X_all)
        
        self.X_cm1 = self.scaler.transform(self.X_cm1)
        self.X_cm2_all = self.scaler.transform(self.X_cm2_all)
        self.X_cm2_labeled = self.scaler.transform(self.X_cm2_labeled)
        
        # 准备标签
        self.y_cm2_labeled = (self.sample3_cm2_labeled['vehicle_type'] == 'pure_electric').astype(int)
        
        logger.info(f"特征维度: {self.X_cm1.shape[1]}")
        logger.info(f"CM2标签样本中纯电比例: {self.y_cm2_labeled.mean():.2%}")
        
    def train_model(self):
        """训练监督学习模型"""
        logger.info("训练监督学习模型...")
        
        # 使用CM2标签样本训练模型
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_cm2_labeled, self.y_cm2_labeled, 
            test_size=0.3, random_state=42, stratify=self.y_cm2_labeled
        )
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        logger.info("模型评估结果:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # 特征重要性
        feature_names = ['buyer_age', 'order_gender', 'License_City', 'Parent_Region_Name']
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("特征重要性:")
        for _, row in feature_importance.iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")
            
    def calculate_distribution_weights(self):
        """计算分布校准权重"""
        logger.info("计算分布校准权重...")
        
        # 比较CM2小样本与CM2全量的特征分布差异
        # 使用特征的统计量进行简单的重要性加权
        
        # 计算CM2全量和CM2小样本的特征均值
        cm2_all_mean = np.mean(self.X_cm2_all, axis=0)
        cm2_labeled_mean = np.mean(self.X_cm2_labeled, axis=0)
        
        # 计算分布差异（欧氏距离）
        distribution_diff = np.linalg.norm(cm2_all_mean - cm2_labeled_mean)
        logger.info(f"CM2全量与小样本特征分布差异: {distribution_diff:.4f}")
        
        # 简单的校准策略：如果分布差异较大，则降低置信度
        self.calibration_factor = max(0.5, 1.0 - distribution_diff * 0.1)
        logger.info(f"分布校准因子: {self.calibration_factor:.4f}")
        
    def predict_cm2_ratios(self):
        """预测CM2全量的纯电/增程比例"""
        logger.info("预测CM2全量纯电/增程比例...")
        
        # 预测CM2全量用户的纯电概率
        pure_electric_probs = self.model.predict_proba(self.X_cm2_all)[:, 1]
        
        # 应用分布校准
        calibrated_probs = pure_electric_probs * self.calibration_factor + \
                          (1 - self.calibration_factor) * np.mean(self.y_cm2_labeled)
        
        # 计算最终比例
        pure_electric_ratio = np.mean(calibrated_probs)
        range_extender_ratio = 1 - pure_electric_ratio
        
        self.results = {
            'pure_electric_ratio': pure_electric_ratio,
            'range_extender_ratio': range_extender_ratio,
            'total_cm2_orders': len(self.sample2_cm2_all),
            'estimated_pure_electric_orders': int(pure_electric_ratio * len(self.sample2_cm2_all)),
            'estimated_range_extender_orders': int(range_extender_ratio * len(self.sample2_cm2_all)),
            'calibration_factor': self.calibration_factor,
            'labeled_sample_ratio': len(self.sample3_cm2_labeled) / len(self.sample2_cm2_all)
        }
        
        return self.results
        
    def analyze_by_segments(self):
        """按细分维度分析纯电/增程比例"""
        logger.info("按细分维度分析...")
        
        # 预测概率
        pure_electric_probs = self.model.predict_proba(self.X_cm2_all)[:, 1]
        
        # 添加预测结果到CM2数据
        cm2_with_pred = self.sample2_cm2_all.copy()
        cm2_with_pred['pure_electric_prob'] = pure_electric_probs
        cm2_with_pred['predicted_type'] = (pure_electric_probs > 0.5).astype(int)
        
        segment_analysis = {}
        
        # 按性别分析
        if 'order_gender' in cm2_with_pred.columns:
            gender_analysis = cm2_with_pred.groupby('order_gender').agg({
                'pure_electric_prob': ['mean', 'count']
            }).round(4)
            segment_analysis['gender'] = gender_analysis
        
        # 按地区分析
        if 'Parent Region Name' in cm2_with_pred.columns:
            region_analysis = cm2_with_pred.groupby('Parent Region Name').agg({
                'pure_electric_prob': ['mean', 'count']
            }).round(4)
            # 只显示订单量较多的地区
            region_analysis = region_analysis[region_analysis[('pure_electric_prob', 'count')] >= 50]
            segment_analysis['region'] = region_analysis
        
        # 按年龄段分析
        if 'buyer_age' in cm2_with_pred.columns:
            cm2_with_pred['age_group'] = pd.cut(cm2_with_pred['buyer_age'], 
                                               bins=[0, 25, 35, 45, 55, 100], 
                                               labels=['<25', '25-35', '35-45', '45-55', '55+'])
            age_analysis = cm2_with_pred.groupby('age_group').agg({
                'pure_electric_prob': ['mean', 'count']
            }).round(4)
            segment_analysis['age'] = age_analysis
        
        self.segment_results = segment_analysis
        return segment_analysis
        
    def generate_report(self):
        """生成推断结果报告"""
        logger.info("生成推断结果报告...")
        
        report = f"""
# CM2车型纯电/增程比例推断报告

## 📊 总体推断结果

- **CM2总订单数**: {self.results['total_cm2_orders']:,}单
- **预测纯电比例**: {self.results['pure_electric_ratio']:.2%}
- **预测增程比例**: {self.results['range_extender_ratio']:.2%}
- **预测纯电订单数**: {self.results['estimated_pure_electric_orders']:,}单
- **预测增程订单数**: {self.results['estimated_range_extender_orders']:,}单

## 🔧 模型参数

- **标签样本占比**: {self.results['labeled_sample_ratio']:.2%}
- **分布校准因子**: {self.results['calibration_factor']:.4f}
- **模型类型**: 随机森林分类器

## 📈 细分维度分析

### 按性别分析
"""
        
        if 'gender' in self.segment_results:
            report += "\n| 性别 | 纯电比例 | 订单数 |\n|------|----------|--------|\n"
            for gender, data in self.segment_results['gender'].iterrows():
                ratio = data[('pure_electric_prob', 'mean')]
                count = int(data[('pure_electric_prob', 'count')])
                report += f"| {gender} | {ratio:.2%} | {count:,} |\n"
        
        if 'age' in self.segment_results:
            report += "\n### 按年龄段分析\n\n| 年龄段 | 纯电比例 | 订单数 |\n|--------|----------|--------|\n"
            for age_group, data in self.segment_results['age'].iterrows():
                ratio = data[('pure_electric_prob', 'mean')]
                count = int(data[('pure_electric_prob', 'count')])
                report += f"| {age_group} | {ratio:.2%} | {count:,} |\n"
        
        if 'region' in self.segment_results:
            report += "\n### 按地区分析（订单数≥50）\n\n| 地区 | 纯电比例 | 订单数 |\n|------|----------|--------|\n"
            for region, data in self.segment_results['region'].iterrows():
                ratio = data[('pure_electric_prob', 'mean')]
                count = int(data[('pure_electric_prob', 'count')])
                report += f"| {region} | {ratio:.2%} | {count:,} |\n"
        
        report += f"""

## 📝 方法说明

本推断基于以下方法：

1. **监督学习**: 使用CM2小样本标签数据训练随机森林分类器
2. **特征工程**: 基于年龄、性别、城市、地区等特征进行编码和标准化
3. **分布校准**: 通过比较小样本与全量样本的特征分布进行校准
4. **细分分析**: 按不同维度分析纯电/增程偏好差异

## ⚠️ 注意事项

- 推断结果基于现有标签样本的代表性
- 分布校准因子反映了小样本与全量样本的差异程度
- 建议结合业务知识对结果进行验证和调整

---
*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 保存报告
        report_path = "/Users/zihao_/Documents/github/W35_workflow/cm2_inference_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"推断报告已保存: {report_path}")
        return report
        
    def run_inference(self):
        """运行完整的推断流程"""
        logger.info("开始CM2纯电/增程比例推断...")
        
        try:
            # 1. 加载数据
            self.load_data()
            
            # 2. 筛选车型数据
            self.filter_vehicle_data()
            
            # 3. 准备样本
            self.prepare_samples()
            
            # 4. 特征工程
            self.feature_engineering()
            
            # 5. 训练模型
            self.train_model()
            
            # 6. 分布校准
            self.calculate_distribution_weights()
            
            # 7. 预测比例
            results = self.predict_cm2_ratios()
            
            # 8. 细分分析
            self.analyze_by_segments()
            
            # 9. 生成报告
            report = self.generate_report()
            
            logger.info("推断完成！")
            logger.info(f"CM2纯电比例: {results['pure_electric_ratio']:.2%}")
            logger.info(f"CM2增程比例: {results['range_extender_ratio']:.2%}")
            
            return results, report
            
        except Exception as e:
            logger.error(f"推断过程中发生错误: {str(e)}")
            raise

def main():
    """主函数"""
    # 数据路径
    data_path = "/Users/zihao_/Documents/coding/dataset/formatted/intention_order_analysis.parquet"
    business_definition_path = "/Users/zihao_/Documents/github/W35_workflow/business_definition.json"
    
    # 创建推断引擎
    engine = CM2InferenceEngine(data_path, business_definition_path)
    
    # 运行推断
    results, report = engine.run_inference()
    
    print("\n" + "="*50)
    print("CM2车型纯电/增程比例推断结果")
    print("="*50)
    print(f"总订单数: {results['total_cm2_orders']:,}")
    print(f"纯电比例: {results['pure_electric_ratio']:.2%}")
    print(f"增程比例: {results['range_extender_ratio']:.2%}")
    print(f"预测纯电订单: {results['estimated_pure_electric_orders']:,}")
    print(f"预测增程订单: {results['estimated_range_extender_orders']:,}")
    print("="*50)
    
if __name__ == "__main__":
    main()