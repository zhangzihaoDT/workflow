"""
测试批量内容提取器
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(__file__))

from batch_content_extractor import BatchContentExtractor


def test_small_batch():
    """测试小批量提取（前3页）"""
    print("🧪 开始测试批量提取功能（前3页）")
    print("=" * 50)
    
    # 创建批量提取器
    batch_extractor = BatchContentExtractor()
    
    try:
        # 测试提取前3页
        batch_extractor.extract_batch(
            start_num=1,
            end_num=3,
            delay=1.0  # 测试时使用较短的延迟
        )
        
        print("\n✅ 小批量测试完成！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_small_batch()
    
    if success:
        print("\n🎉 测试成功！批量提取脚本工作正常。")
        print("现在可以运行完整的65页提取任务。")
    else:
        print("\n⚠️ 测试失败，请检查脚本配置。")