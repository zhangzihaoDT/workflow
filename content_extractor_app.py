"""
懂车帝内容提取器 - Gradio Web应用

一个用户友好的前端界面，用于提取和展示懂车帝社区页面的用户发布内容
"""

import gradio as gr
import sys
import os
import time
from typing import Tuple, List, Dict, Any

# 添加skills目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'skills'))

from enhanced_content_extractor import EnhancedContentExtractor


class ContentExtractorApp:
    """内容提取器应用"""
    
    def __init__(self):
        self.extractor = EnhancedContentExtractor()
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
    
    with gr.Blocks(css=custom_css, title="懂车帝内容提取器") as demo:
        gr.HTML("""
        <div class="header">
            <h1>🚗 懂车帝内容提取器</h1>
            <p>智能提取懂车帝社区页面的用户发布内容，支持内容过滤和格式化</p>
        </div>
        """)
        
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