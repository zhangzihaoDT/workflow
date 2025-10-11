"""
懂车帝内容提取器 - Gradio Web应用

一个用户友好的前端界面，用于提取和展示懂车帝社区页面的用户发布内容
包含AI问答检索功能，基于DeepSeek API
"""

import gradio as gr
import sys
import os
import time
import json
import requests
from datetime import datetime
from typing import Tuple, List, Dict, Any

# 添加skills目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'skills'))

from enhanced_content_extractor import EnhancedContentExtractor


class AIQARetriever:
    """AI问答检索器"""
    
    def __init__(self):
        self.api_key = "sk-d7b34ad3d34e40218e1674383b400702"
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.data_file = "/Users/zihao_/Documents/github/W35_workflow/report/batch_extracted_content_20251010_135757.json"
        self.posts_data = None
        self._load_data()
    
    def _load_data(self):
        """加载JSON数据"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.posts_data = data.get('posts', [])
                print(f"✅ 成功加载 {len(self.posts_data)} 条内容数据")
        except Exception as e:
            print(f"❌ 加载数据失败: {str(e)}")
            self.posts_data = []
    
    def _filter_posts_by_time(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """
        根据时间范围过滤帖子数据
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD 格式)
            end_date: 结束日期 (YYYY-MM-DD 格式)
            
        Returns:
            过滤后的帖子列表
        """
        if not self.posts_data:
            return []
        
        filtered_posts = []
        
        for post in self.posts_data:
            timestamp = post.get('timestamp', '')
            if not timestamp:
                continue
            
            # 提取日期部分 (YYYY-MM-DD)
            post_date = timestamp.split(' ')[0] if ' ' in timestamp else timestamp
            
            # 检查是否在时间范围内
            include_post = True
            
            if start_date and post_date < start_date:
                include_post = False
            
            if end_date and post_date > end_date:
                include_post = False
            
            if include_post:
                filtered_posts.append(post)
        
        return filtered_posts
    
    def get_time_range_info(self) -> Tuple[str, str, int]:
        """
        获取数据的时间范围信息
        
        Returns:
            Tuple[最早日期, 最晚日期, 总数据量]
        """
        if not self.posts_data:
            return "", "", 0
        
        dates = []
        for post in self.posts_data:
            timestamp = post.get('timestamp', '')
            if timestamp:
                post_date = timestamp.split(' ')[0] if ' ' in timestamp else timestamp
                dates.append(post_date)
        
        if dates:
            dates.sort()
            return dates[0], dates[-1], len(self.posts_data)
        
        return "", "", len(self.posts_data)
    
    def _save_filtered_dataset(self, filtered_posts: List[Dict], question: str, start_date: str = None, end_date: str = None) -> str:
        """
        保存筛选后的数据集为JSON文件
        
        Args:
            filtered_posts: 筛选后的帖子数据
            question: 用户问题
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            保存的文件路径
        """
        # 确保report目录存在
        report_dir = os.path.join(os.path.dirname(__file__), 'report')
        os.makedirs(report_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"filtered_dataset_{timestamp}.json"
        file_path = os.path.join(report_dir, filename)
        
        # 构建数据集信息
        dataset_info = {
            "filter_info": {
                "question": question,
                "start_date": start_date or "不限",
                "end_date": end_date or "不限",
                "filter_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "original_count": len(self.posts_data),
                "filtered_count": len(filtered_posts)
            },
            "time_range": {
                "earliest": min([post.get('timestamp', '') for post in filtered_posts]) if filtered_posts else "",
                "latest": max([post.get('timestamp', '') for post in filtered_posts]) if filtered_posts else ""
            },
            "posts": filtered_posts
        }
        
        # 保存文件
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 筛选数据集已保存: {filename}")
            return file_path
        except Exception as e:
            print(f"❌ 保存筛选数据集失败: {str(e)}")
            return ""
    
    def _call_deepseek_api(self, messages: List[Dict[str, str]]) -> str:
        """调用DeepSeek API，包含重试机制"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        # 重试机制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 增加超时时间到60秒
                response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                return result['choices'][0]['message']['content']
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"⚠️ API调用超时，正在重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(2)  # 等待2秒后重试
                    continue
                else:
                    return f"API调用失败: 连接超时，已重试{max_retries}次"
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ API调用出错，正在重试 ({attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(2)  # 等待2秒后重试
                    continue
                else:
                    return f"API调用失败: {str(e)}"
    
    def search_and_answer(self, question: str, start_date: str = None, end_date: str = None, progress=gr.Progress()) -> Tuple[str, str, str]:
        """
        基于内容数据回答用户问题
        
        Args:
            question: 用户问题
            start_date: 开始日期 (YYYY-MM-DD 格式)
            end_date: 结束日期 (YYYY-MM-DD 格式)
            progress: Gradio进度条
            
        Returns:
            Tuple[回答内容, 处理日志, 筛选数据集文件路径]
        """
        if not question or not question.strip():
            return "❌ 请输入有效的问题", "", ""
        
        if not self.posts_data:
            return "❌ 数据文件未加载成功，请检查文件路径", "", ""
        
        progress(0.1, desc="正在分析问题...")
        
        # 根据时间范围过滤数据
        filtered_posts = self._filter_posts_by_time(start_date, end_date)
        
        if not filtered_posts:
            time_info = f"时间范围：{start_date or '不限'} 到 {end_date or '不限'}"
            return f"❌ 在指定的时间范围内没有找到数据。{time_info}", "", ""
        
        print(f"🔍 时间过滤：从 {len(self.posts_data)} 条数据中筛选出 {len(filtered_posts)} 条")
        
        # 使用过滤后的数据替换原始数据进行分析
        original_posts = self.posts_data
        self.posts_data = filtered_posts
        
        # 构建系统提示
        system_prompt = """你是一个专业的汽车行业分析师，擅长分析用户评论和反馈。
你将基于提供的懂车帝社区用户评论数据来回答问题。
请仔细分析所有相关内容，提供准确、客观、有见地的回答。
回答要求：
1. 基于实际数据内容回答，不要编造信息
2. 如果数据中没有相关信息，请明确说明
3. 提供具体的数据支撑和用户评论引用
4. 分析要客观中立，避免主观偏见
5. 回答要结构清晰，便于理解"""
        
        # 分批处理内容，避免单次请求过大 - "一目十行"
        batch_size = 10
        total_batches = (len(self.posts_data) + batch_size - 1) // batch_size
        
        relevant_contents = []
        processing_log = []
        
        progress(0.2, desc="开始分析内容数据...")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(self.posts_data))
            batch_posts = self.posts_data[start_idx:end_idx]
            
            # 更新进度
            current_progress = 0.2 + (batch_idx / total_batches) * 0.6
            progress(current_progress, desc=f"正在分析第 {batch_idx + 1}/{total_batches} 批内容...")
            
            print(f"🔍 正在分析第 {batch_idx + 1}/{total_batches} 批内容 ({start_idx + 1}-{end_idx}/{len(self.posts_data)})")
            processing_log.append(f"分析第 {batch_idx + 1}/{total_batches} 批内容 ({start_idx + 1}-{end_idx}/{len(self.posts_data)})")
            
            # 构建批次内容摘要
            batch_content = ""
            for i, post in enumerate(batch_posts):
                content_preview = post.get('content', '')[:200] + "..." if len(post.get('content', '')) > 200 else post.get('content', '')
                batch_content += f"""
内容 {start_idx + i + 1}:
用户: {post.get('username', '未知')}
时间: {post.get('timestamp', '未知')}
互动: 点赞{post.get('interactions', {}).get('likes', 0)}次, 评论{post.get('interactions', {}).get('comments', 0)}次
内容: {content_preview}
质量分数: {post.get('quality_score', 0)}
---
"""
            
            # 询问AI这批内容是否与问题相关
            relevance_messages = [
                {"role": "system", "content": "你需要判断以下用户评论内容是否与用户问题相关。如果相关，请提取关键信息；如果不相关，请回答'不相关'。"},
                {"role": "user", "content": f"用户问题：{question}\n\n评论内容：{batch_content}\n\n请判断这些内容是否与问题相关，如果相关请提取关键信息。"}
            ]
            
            relevance_result = self._call_deepseek_api(relevance_messages)
            
            if "不相关" not in relevance_result and len(relevance_result) > 10:
                relevant_contents.append(relevance_result)
                processing_log.append(f"✅ 第 {batch_idx + 1} 批发现相关内容")
            else:
                processing_log.append(f"⚪ 第 {batch_idx + 1} 批无相关内容")
        
        progress(0.8, desc="正在生成最终回答...")
        
        if not relevant_contents:
            final_answer = "根据分析所有925条用户评论，没有找到与您问题直接相关的内容。请尝试调整问题或提供更具体的关键词。"
            processing_log.append("❌ 未找到相关内容")
        else:
            # 基于相关内容生成最终回答
            combined_content = "\n\n".join(relevant_contents)
            final_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"用户问题：{question}\n\n相关内容分析：\n{combined_content}\n\n请基于以上内容给出详细、准确的回答。"}
            ]
            
            final_answer = self._call_deepseek_api(final_messages)
            processing_log.append(f"✅ 基于 {len(relevant_contents)} 批相关内容生成回答")
        
        progress(1.0, desc="分析完成！")
        
        # 保存筛选后的数据集
        filtered_dataset_path = self._save_filtered_dataset(filtered_posts, question, start_date, end_date)
        
        # 恢复原始数据
        self.posts_data = original_posts
        
        # 格式化处理日志
        log_text = "\n".join(processing_log)
        
        return final_answer, log_text, filtered_dataset_path


class ContentExtractorApp:
    """内容提取器应用"""
    
    def __init__(self):
        self.extractor = EnhancedContentExtractor()
        self.ai_qa = AIQARetriever()
        self.last_result = None
    
    def extract_content(self, url: str, progress=gr.Progress()) -> Tuple[str, str, str]:
        """
        提取内容的主要函数
        
        Args:
            url: 要提取的网页URL
            progress: Gradio进度条
            
        Returns:
            Tuple[统计信息, 格式化内容, 下载链接]
        """
        if not url or not url.strip():
            return "❌ 请输入有效的URL", "", ""
        
        url = url.strip()
        
        # 验证URL格式
        if not (url.startswith('http://') or url.startswith('https://')):
            return "❌ URL格式不正确，请输入完整的网址（包含http://或https://）", "", ""
        
        try:
            progress(0.1, desc="正在连接网站...")
            
            # 执行内容提取
            progress(0.3, desc="正在分析网页结构...")
            result = self.extractor.analyze_and_extract(url)
            
            if not result["success"]:
                return f"❌ 提取失败: {result.get('error', '未知错误')}", "", ""
            
            progress(0.7, desc="正在格式化内容...")
            
            # 保存结果供下载
            self.last_result = result
            
            # 生成统计信息
            stats_info = self._generate_stats_info(result)
            
            # 生成格式化内容展示
            formatted_content = self._generate_formatted_content(result)
            
            progress(0.9, desc="正在生成下载文件...")
            
            # 生成下载文件
            download_file = self._save_result_file(result)
            
            progress(1.0, desc="提取完成！")
            
            return stats_info, formatted_content, download_file
            
        except Exception as e:
            error_msg = f"❌ 处理过程中发生错误: {str(e)}"
            return error_msg, "", ""
    
    def _generate_stats_info(self, result: Dict[str, Any]) -> str:
        """生成统计信息"""
        summary = result.get('summary', {})
        
        stats = f"""
## 📊 提取统计信息

**🔗 URL:** {result['url']}
**⏰ 提取时间:** {result['analysis_time']}

### 📈 数据概览
- **原始内容数:** {result['total_raw_posts']} 条
- **高质量内容数:** {result['total_posts']} 条
- **过滤率:** {((result['total_raw_posts'] - result['total_posts']) / result['total_raw_posts'] * 100):.1f}% (已过滤低质量内容)

### 📝 内容分析
- **总字符数:** {summary.get('total_content_length', 0):,} 字符
- **平均长度:** {summary.get('average_content_length', 0):.1f} 字符/条
- **平均质量分数:** {summary.get('average_quality_score', 0):.2f}/10

### 🏷️ 内容特征
- **包含用户名:** {summary.get('posts_with_username', 0)} 条
- **包含时间戳:** {summary.get('posts_with_timestamp', 0)} 条
- **包含图片:** {summary.get('posts_with_images', 0)} 条
- **包含互动数据:** {summary.get('posts_with_interactions', 0)} 条
"""
        return stats
    
    def _generate_formatted_content(self, result: Dict[str, Any]) -> str:
        """生成格式化的内容展示"""
        posts = result.get('posts', [])
        
        if not posts:
            return "❌ 未提取到有效内容"
        
        content_display = f"## 📋 提取到的用户内容 ({len(posts)} 条)\n\n"
        
        for i, post in enumerate(posts[:10], 1):  # 只显示前10条
            content_display += f"### 【内容 {i}】\n"
            
            # 用户信息
            if post.get('username'):
                content_display += f"**👤 用户:** {post['username']}\n"
            
            # 时间信息
            if post.get('timestamp'):
                content_display += f"**🕒 时间:** {post['timestamp']}\n"
            
            # 互动数据
            interactions = post.get('interactions', {})
            if interactions:
                interaction_parts = []
                if 'likes' in interactions:
                    interaction_parts.append(f"👍 {interactions['likes']}点赞")
                if 'comments' in interactions:
                    interaction_parts.append(f"💬 {interactions['comments']}评论")
                if 'shares' in interactions:
                    interaction_parts.append(f"🔄 {interactions['shares']}分享")
                
                if interaction_parts:
                    content_display += f"**📊 互动:** {' | '.join(interaction_parts)}\n"
            
            # 图片信息
            if post.get('images'):
                content_display += f"**🖼️ 图片:** {len(post['images'])} 张\n"
            
            # 质量分数
            if 'quality_score' in post:
                content_display += f"**⭐ 质量分数:** {post['quality_score']:.1f}/10\n"
            
            # 内容文本
            content = post.get('content', '')
            if len(content) > 300:
                content = content[:300] + "..."
            content_display += f"**📄 内容:**\n{content}\n"
            
            content_display += "\n---\n\n"
        
        if len(posts) > 10:
            content_display += f"*注：仅显示前10条内容，完整内容请下载文件查看（共{len(posts)}条）*\n"
        
        return content_display
    
    def _save_result_file(self, result: Dict[str, Any]) -> str:
        """保存结果文件并返回文件路径"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"extracted_content_{timestamp}.txt"
        
        # 设置保存路径为 report 目录
        report_dir = os.path.join(os.path.dirname(__file__), 'report')
        
        # 确保 report 目录存在
        os.makedirs(report_dir, exist_ok=True)
        
        filepath = os.path.join(report_dir, filename)
        
        posts = result.get('posts', [])
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("懂车帝用户内容提取结果\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"URL: {result['url']}\n")
            f.write(f"提取时间: {result['analysis_time']}\n")
            f.write(f"总计: {len(posts)} 条内容\n")
            f.write("=" * 50 + "\n\n")
            
            for i, post in enumerate(posts, 1):
                f.write(f"【内容 {i}】\n")
                
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
                    
                    if interaction_parts:
                        f.write(f"互动: {', '.join(interaction_parts)}\n")
                
                f.write(f"内容: {post.get('content', '')}\n")
                
                if post.get('images'):
                    f.write(f"图片: {len(post['images'])} 张\n")
                
                f.write("-" * 50 + "\n\n")
        
        return filepath


def create_app():
    """创建Gradio应用"""
    app = ContentExtractorApp()
    
    # 自定义CSS样式
    custom_css = """
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #C8D0D9 0%, #ffffff 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stats-box {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .content-box {
        background: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=custom_css, title="懂车帝内容提取器 & AI问答检索") as demo:
        gr.HTML("""
        <div class="header">
            <h1>🚗 懂车帝内容提取器 & AI问答检索</h1>
            <p>智能提取懂车帝社区页面的用户发布内容，支持内容过滤和格式化，以及基于DeepSeek AI的问答检索</p>
        </div>
        """)
        
        with gr.Tabs():
            # 第一个标签页：内容提取
            with gr.TabItem("📥 内容提取"):
                with gr.Row():
                    with gr.Column(scale=3):
                        url_input = gr.Textbox(
                            label="🔗 请输入懂车帝社区页面URL",
                            placeholder="例如: https://www.dongchedi.com/community/9118/hot-release-49",
                            lines=1
                        )
                    
                    with gr.Column(scale=1):
                        extract_btn = gr.Button(
                            "🚀 开始提取",
                            variant="primary",
                            size="lg"
                        )
                
                # 示例URL
                gr.HTML("""
                <div style="margin: 10px 0; padding: 10px; background: #e3f2fd; border-radius: 5px;">
                    <strong>💡 示例URL:</strong><br>
                    • https://www.dongchedi.com/community/9118/hot-release-49<br>
                    • https://www.dongchedi.com/community/[其他社区ID]/hot-release-[页面ID]
                </div>
                """)
                
                # 结果展示区域
                with gr.Row():
                    with gr.Column(scale=1):
                        stats_output = gr.Markdown(
                            label="📊 统计信息",
                            elem_classes=["stats-box"]
                        )
                    
                    with gr.Column(scale=2):
                        content_output = gr.Markdown(
                            label="📋 提取内容",
                            elem_classes=["content-box"]
                        )
                
                # 下载区域
                download_file = gr.File(
                    label="📥 下载完整结果文件",
                    visible=False
                )
                
                # 绑定事件
                extract_btn.click(
                    fn=app.extract_content,
                    inputs=[url_input],
                    outputs=[stats_output, content_output, download_file],
                    show_progress=True
                ).then(
                    fn=lambda: gr.update(visible=True),
                    outputs=[download_file]
                )
                
                # 添加说明
                gr.HTML("""
                <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                    <h3>📖 使用说明</h3>
                    <ul>
                        <li><strong>输入URL:</strong> 粘贴懂车帝社区页面的完整链接</li>
                        <li><strong>点击提取:</strong> 系统会自动分析页面并提取用户发布的内容</li>
                        <li><strong>查看结果:</strong> 左侧显示统计信息，右侧显示格式化的内容预览</li>
                        <li><strong>下载文件:</strong> 可下载包含所有提取内容的完整文本文件</li>
                    </ul>
                    
                    <h3>✨ 功能特点</h3>
                    <ul>
                        <li><strong>智能过滤:</strong> 自动过滤UI元素和低质量内容，只保留真实用户发布的内容</li>
                        <li><strong>质量评分:</strong> 基于多维度指标对内容进行质量评分</li>
                        <li><strong>结构化输出:</strong> 包含用户名、时间、互动数据、图片信息等完整信息</li>
                        <li><strong>格式化展示:</strong> 清晰的格式化展示，便于阅读和分析</li>
                    </ul>
                </div>
                """)
            
            # 第二个标签页：AI问答检索
            with gr.TabItem("🤖 AI问答检索"):
                gr.Markdown("""
                基于已提取的懂车帝用户评论数据，使用DeepSeek AI进行智能问答
                
                💡 **使用说明：**
                - 输入您想了解的问题，AI将分析用户评论数据
                - 可选择时间范围来过滤数据，控制分析的数据量
                - 系统会逐批分析内容，并显示实时进度
                - 基于相关内容生成准确、客观的回答
                
                ⏰ **时间过滤：**
                - 留空表示不限制时间范围，分析所有数据
                - 设置开始/结束日期可缩小分析范围，提高效率
                - 日期格式：YYYY-MM-DD（如：2024-01-01）
                """)
                
                with gr.Row():
                    with gr.Column(scale=4):
                        question_input = gr.Textbox(
                            label="💬 请输入您的问题",
                            placeholder="例如：用户对这款车的油耗评价如何？有哪些常见的质量问题？",
                            lines=2
                        )
                    with gr.Column(scale=1):
                        qa_btn = gr.Button("🔍 AI分析", variant="primary", size="lg")
                
                # 时间范围过滤控件
                with gr.Row():
                    with gr.Column(scale=2):
                        start_date = gr.Textbox(
                            label="📅 开始日期 (可选)",
                            placeholder="YYYY-MM-DD，如：2024-01-01",
                            value="",
                            lines=1
                        )
                    with gr.Column(scale=2):
                        end_date = gr.Textbox(
                            label="📅 结束日期 (可选)",
                            placeholder="YYYY-MM-DD，如：2024-12-31",
                            value="",
                            lines=1
                        )
                    with gr.Column(scale=1):
                        # 获取数据时间范围信息
                        min_date, max_date, total_count = app.ai_qa.get_time_range_info()
                        time_info = gr.HTML(f"""
                        <div style="padding: 10px; background: #e8f5e8; border-radius: 5px; font-size: 12px;">
                            <strong>📊 数据范围：</strong><br>
                            • 最早：{min_date}<br>
                            • 最晚：{max_date}<br>
                            • 总计：{total_count} 条
                        </div>
                        """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        answer_output = gr.Textbox(
                            label="🎯 AI回答",
                            lines=20,
                            max_lines=25,
                            show_copy_button=True
                        )
                    with gr.Column(scale=1):
                        log_output = gr.Textbox(
                            label="📊 分析进度",
                            lines=20,
                            max_lines=25
                        )
                
                # 下载筛选数据集区域
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.HTML("""
                        <div style="padding: 10px; background: #f0f8ff; border-radius: 5px; margin: 10px 0;">
                            <strong>📥 筛选数据集下载</strong><br>
                            <small>AI分析完成后，可下载本次使用的筛选数据集，方便复查大模型处理的具体内容</small>
                        </div>
                        """)
                    with gr.Column(scale=1):
                        filtered_dataset_file = gr.File(
                            label="📊 筛选数据集",
                            visible=False,
                            file_count="single"
                        )
                
                # 绑定AI问答事件
                qa_btn.click(
                    fn=app.ai_qa.search_and_answer,
                    inputs=[question_input, start_date, end_date],
                    outputs=[answer_output, log_output, filtered_dataset_file],
                    show_progress=True
                ).then(
                    fn=lambda file_path: gr.update(visible=bool(file_path)),
                    inputs=[filtered_dataset_file],
                    outputs=[filtered_dataset_file]
                )
                
                # AI问答示例
                gr.Examples(
                    examples=[
                        ["用户对这款车的油耗表现如何评价？"],
                        ["有哪些用户反映的质量问题？"],
                        ["用户对车辆的外观设计有什么看法？"],
                        ["关于车辆的性价比，用户怎么说？"],
                        ["用户对售后服务的评价如何？"]
                    ],
                    inputs=[question_input]
                )
                
                # 数据源信息
                gr.Markdown(f"""
                📊 **数据源信息：**
                - 数据文件：`batch_extracted_content_20251010_135757.json`
                - 数据来源：懂车帝社区用户评论
                - AI模型：DeepSeek Chat
                
                📥 **筛选数据集：**
                - 每次AI分析后会生成对应的筛选数据集文件
                - 包含本次分析使用的具体数据内容
                - 便于用户复查和验证AI分析结果
                - 文件格式：JSON，包含过滤条件、时间范围、数据统计等信息
                """)
    
    return demo


if __name__ == "__main__":
    # 创建并启动应用
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )