"""
批量内容提取器

动态修改URL中的数字（1-65），批量提取懂车帝社区内容
"""

import sys
import os
import time
import json
from datetime import datetime
from typing import List, Dict, Any

# 添加skills目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'skills'))

from enhanced_content_extractor import EnhancedContentExtractor


class BatchContentExtractor:
    """批量内容提取器"""
    
    def __init__(self):
        self.extractor = EnhancedContentExtractor()
        self.base_url = "https://www.dongchedi.com/community/9118/hot-release-{}"
        self.all_posts = []
        self.failed_urls = []
        self.success_count = 0
        self.total_count = 0
        
    def extract_batch(self, start_num: int = 1, end_num: int = 65, delay: float = 2.0):
        """
        批量提取内容
        
        Args:
            start_num: 起始数字
            end_num: 结束数字
            delay: 每次请求之间的延迟（秒）
        """
        print(f"开始批量提取内容，范围: {start_num}-{end_num}")
        print(f"每次请求间隔: {delay}秒")
        print("=" * 60)
        
        self.total_count = end_num - start_num + 1
        
        for num in range(start_num, end_num + 1):
            url = self.base_url.format(num)
            
            print(f"\n[{num}/{end_num}] 正在处理: {url}")
            
            try:
                # 执行内容提取
                result = self.extractor.analyze_and_extract(url)
                
                if result["success"]:
                    posts = result.get("posts", [])
                    print(f"✅ 成功提取 {len(posts)} 条内容")
                    
                    # 为每个帖子添加来源URL信息
                    for post in posts:
                        post['source_url'] = url
                        post['page_number'] = num
                    
                    self.all_posts.extend(posts)
                    self.success_count += 1
                    
                else:
                    error_msg = result.get("error", "未知错误")
                    print(f"❌ 提取失败: {error_msg}")
                    self.failed_urls.append({"url": url, "error": error_msg})
                
            except Exception as e:
                print(f"❌ 处理异常: {str(e)}")
                self.failed_urls.append({"url": url, "error": str(e)})
            
            # 显示进度统计
            self._print_progress_stats()
            
            # 延迟以避免过于频繁的请求
            if num < end_num:
                print(f"等待 {delay} 秒...")
                time.sleep(delay)
        
        print("\n" + "=" * 60)
        print("批量提取完成！")
        self._print_final_stats()
        
        # 保存结果
        self._save_results()
    
    def _print_progress_stats(self):
        """打印进度统计"""
        processed = self.success_count + len(self.failed_urls)
        success_rate = (self.success_count / processed * 100) if processed > 0 else 0
        
        print(f"进度: {processed}/{self.total_count} | "
              f"成功: {self.success_count} | "
              f"失败: {len(self.failed_urls)} | "
              f"成功率: {success_rate:.1f}% | "
              f"总内容数: {len(self.all_posts)}")
    
    def _print_final_stats(self):
        """打印最终统计"""
        print(f"📊 最终统计:")
        print(f"   总页面数: {self.total_count}")
        print(f"   成功页面: {self.success_count}")
        print(f"   失败页面: {len(self.failed_urls)}")
        print(f"   成功率: {(self.success_count / self.total_count * 100):.1f}%")
        print(f"   总内容数: {len(self.all_posts)}")
        
        if self.failed_urls:
            print(f"\n❌ 失败的URL:")
            for failed in self.failed_urls:
                print(f"   {failed['url']} - {failed['error']}")
    
    def _save_results(self):
        """保存提取结果"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 确保report目录存在
        report_dir = os.path.join(os.path.dirname(__file__), 'report')
        os.makedirs(report_dir, exist_ok=True)
        
        # 保存JSON格式
        json_filename = f"batch_extracted_content_{timestamp}.json"
        json_filepath = os.path.join(report_dir, json_filename)
        
        json_data = {
            "extraction_info": {
                "timestamp": timestamp,
                "total_pages": self.total_count,
                "successful_pages": self.success_count,
                "failed_pages": len(self.failed_urls),
                "total_posts": len(self.all_posts),
                "base_url_pattern": self.base_url,
                "extraction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "failed_urls": self.failed_urls,
            "posts": self.all_posts
        }
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"📄 JSON文件已保存: {json_filepath}")
        
        # 保存文本格式
        txt_filename = f"batch_extracted_content_{timestamp}.txt"
        txt_filepath = os.path.join(report_dir, txt_filename)
        
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write("懂车帝社区批量内容提取结果\n")
            f.write("=" * 60 + "\n\n")
            
            # 写入提取信息
            f.write(f"提取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"URL模式: {self.base_url}\n")
            f.write(f"页面范围: 1-65\n")
            f.write(f"总页面数: {self.total_count}\n")
            f.write(f"成功页面: {self.success_count}\n")
            f.write(f"失败页面: {len(self.failed_urls)}\n")
            f.write(f"总内容数: {len(self.all_posts)}\n")
            f.write("=" * 60 + "\n\n")
            
            # 写入失败的URL（如果有）
            if self.failed_urls:
                f.write("失败的URL列表:\n")
                f.write("-" * 30 + "\n")
                for failed in self.failed_urls:
                    f.write(f"URL: {failed['url']}\n")
                    f.write(f"错误: {failed['error']}\n\n")
                f.write("=" * 60 + "\n\n")
            
            # 写入所有提取的内容
            f.write("提取的内容列表:\n")
            f.write("=" * 60 + "\n\n")
            
            for i, post in enumerate(self.all_posts, 1):
                f.write(f"【内容 {i}】\n")
                f.write(f"来源页面: {post.get('page_number', 'N/A')}\n")
                f.write(f"来源URL: {post.get('source_url', 'N/A')}\n")
                
                if post.get('username'):
                    f.write(f"用户: {post['username']}\n")
                
                if post.get('timestamp'):
                    f.write(f"时间: {post['timestamp']}\n")
                
                interactions = post.get('interactions', {})
                if interactions:
                    interaction_parts = []
                    for key, value in interactions.items():
                        if key == 'likes':
                            interaction_parts.append(f"likes: {value}")
                        elif key == 'comments':
                            interaction_parts.append(f"comments: {value}")
                        elif key == 'shares':
                            interaction_parts.append(f"shares: {value}")
                        elif key == 'views':
                            interaction_parts.append(f"views: {value}")
                    
                    if interaction_parts:
                        f.write(f"互动: {', '.join(interaction_parts)}\n")
                
                f.write(f"内容: {post.get('content', '')}\n")
                
                if post.get('images'):
                    f.write(f"图片: {len(post['images'])} 张\n")
                
                if 'quality_score' in post:
                    f.write(f"质量分数: {post['quality_score']:.2f}\n")
                
                f.write("-" * 60 + "\n\n")
        
        print(f"📄 文本文件已保存: {txt_filepath}")
        
        return json_filepath, txt_filepath


def main():
    """主函数"""
    print("🚗 懂车帝社区批量内容提取器")
    print("=" * 60)
    
    # 创建批量提取器
    batch_extractor = BatchContentExtractor()
    
    # 开始批量提取
    try:
        batch_extractor.extract_batch(
            start_num=1,
            end_num=65,
            delay=2.0  # 2秒延迟，避免请求过于频繁
        )
        
        print("\n🎉 批量提取任务完成！")
        print("所有结果已保存到 report 目录中。")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断了提取过程")
        print("正在保存已提取的内容...")
        batch_extractor._save_results()
        
    except Exception as e:
        print(f"\n❌ 批量提取过程中发生错误: {str(e)}")
        print("正在保存已提取的内容...")
        batch_extractor._save_results()


if __name__ == "__main__":
    main()