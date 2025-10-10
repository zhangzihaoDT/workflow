"""
增强版用户内容提取器

专门用于从懂车帝社区页面提取用户发布的实际内容文本
"""

import requests
from bs4 import BeautifulSoup
import json
import re
from typing import Dict, List, Any
import time
from datetime import datetime


class EnhancedContentExtractor:
    """增强版内容提取器"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def _convert_timestamp_to_date(self, timestamp: int) -> str:
        """将Unix时间戳转换为可读的日期格式"""
        try:
            # 处理秒级时间戳
            if timestamp > 1000000000000:  # 毫秒级时间戳
                timestamp = timestamp / 1000
            
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, OSError):
            return ""
    
    def fetch_content(self, url: str) -> BeautifulSoup:
        """获取网页内容"""
        try:
            print(f"正在获取网页内容: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            
            soup = BeautifulSoup(response.text, 'html.parser')
            print(f"成功获取网页内容，页面大小: {len(response.text)} 字符")
            return soup
            
        except requests.RequestException as e:
            print(f"获取网页内容失败: {e}")
            raise
    
    def extract_post_content(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        提取帖子内容
        
        重点关注提取实际的用户发布文本内容
        """
        print("正在提取用户发布的内容...")
        
        posts = []
        
        # 首先尝试从页面的JSON数据中提取结构化内容
        try:
            json_posts = self._extract_from_json_data(soup)
            if json_posts:
                print(f"从JSON数据中提取到 {len(json_posts)} 条内容")
                posts.extend(json_posts)
        except Exception as e:
            print(f"从JSON数据提取失败: {e}")
        
        # 如果JSON提取失败或内容不足，使用HTML解析作为备选
        if len(posts) < 5:
            print("JSON提取内容不足，使用HTML解析作为补充")
            html_posts = self._extract_from_html(soup)
            posts.extend(html_posts)
        
        # 去重和排序
        unique_posts = self._deduplicate_posts(posts)
        
        print(f"提取到 {len(unique_posts)} 条用户内容")
        return unique_posts
    
    def _extract_from_html(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        从HTML中提取帖子内容（原有的提取逻辑）
        """
        posts = []
        
        # 方法1: 查找所有包含中文文本的段落和div，过滤出可能的用户内容
        all_text_elements = soup.find_all(['p', 'div', 'span'], string=True)
        
        for element in all_text_elements:
            text = element.get_text(strip=True)
            
            # 基本过滤条件
            if not self._is_valid_content(text):
                continue
            
            # 查找父容器以获取更多上下文信息
            post_container = self._find_post_container(element)
            
            if post_container:
                post_data = self._extract_post_info(post_container, text)
                if post_data:
                    posts.append(post_data)
        
        # 方法2: 专门查找可能包含帖子内容的特定结构
        # 基于懂车帝的页面结构，查找可能的帖子容器
        potential_post_containers = soup.find_all('div', class_=re.compile(r'item|post|content|feed'))
        
        for container in potential_post_containers:
            post_text = self._extract_main_text_from_container(container)
            if post_text and self._is_valid_content(post_text):
                post_data = self._extract_post_info(container, post_text)
                if post_data and not self._is_duplicate(post_data, posts):
                    posts.append(post_data)
        
        # 方法3: 查找包含特定关键词的内容块
        content_keywords = ['智己', 'LS6', '车', '驾驶', '体验', '感受', '推荐', '不错', '满意', '问题']
        
        for keyword in content_keywords:
            keyword_elements = soup.find_all(string=re.compile(keyword))
            for element in keyword_elements:
                if hasattr(element, 'parent'):
                    parent = element.parent
                    text = parent.get_text(strip=True)
                    if self._is_valid_content(text) and len(text) > 50:
                        post_data = self._extract_post_info(parent, text)
                        if post_data and not self._is_duplicate(post_data, posts):
                            posts.append(post_data)
        
        return posts
    
    def _is_valid_content(self, text: str) -> bool:
        """判断文本是否是有效的用户内容"""
        if not text or len(text) < 20:
            return False
        
        # 排除明显的UI元素
        ui_keywords = [
            '登录', '注册', '搜索', '首页', '导航', '菜单', '广告',
            '版权', '备案', '联系我们', '关于我们', '隐私政策',
            '点赞', '评论', '分享', '收藏', '关注', '取消关注',
            'Copyright', '©', '京ICP', '沪ICP', 'App', '小程序',
            '业务合作', '下载', '发布作品', '车友圈', '圈主'
        ]
        
        for keyword in ui_keywords:
            if keyword in text and len(text) < 100:  # 短文本包含UI关键词的排除
                return False
        
        # 排除纯数字、纯符号的内容
        if re.match(r'^[\d\s\-\.\,\:：]+$', text):
            return False
        
        # 排除只有标点符号和空格的内容
        if re.match(r'^[\s\.\,\!\?\-\:：；""''（）\(\)]+$', text):
            return False
        
        # 必须包含中文
        if not re.search(r'[\u4e00-\u9fff]', text):
            return False
        
        # 检查是否包含用户内容的特征词
        content_indicators = [
            '我', '觉得', '认为', '感觉', '体验', '使用', '开了', '买了',
            '推荐', '不推荐', '满意', '不满意', '喜欢', '不喜欢',
            '性价比', '质量', '服务', '态度', '效果', '问题', '建议',
            '智己', 'LS6', '这车', '这款车', '驾驶', '操控', '续航',
            '充电', '空间', '内饰', '外观', '配置', '价格'
        ]
        
        has_content_indicators = any(indicator in text for indicator in content_indicators)
        
        return has_content_indicators
    
    def _find_post_container(self, element):
        """查找包含帖子的容器元素"""
        current = element
        max_depth = 5  # 最多向上查找5层
        
        for _ in range(max_depth):
            if not current or not hasattr(current, 'parent'):
                break
                
            current = current.parent
            
            # 检查是否是可能的帖子容器
            if current.name in ['article', 'section']:
                return current
            
            if current.name == 'div':
                classes = current.get('class', [])
                class_str = ' '.join(classes)
                
                # 查找可能的帖子容器类名
                if any(keyword in class_str.lower() for keyword in 
                      ['post', 'item', 'content', 'feed', 'card', 'message']):
                    return current
                
                # 如果容器有合理的大小和结构，也可能是帖子容器
                text_length = len(current.get_text(strip=True))
                child_count = len(current.find_all())
                
                if 100 <= text_length <= 2000 and 5 <= child_count <= 50:
                    return current
        
        return element.parent if hasattr(element, 'parent') else element
    
    def _extract_main_text_from_container(self, container) -> str:
        """从容器中提取主要文本内容"""
        # 移除脚本和样式标签
        for script in container.find_all(['script', 'style']):
            script.decompose()
        
        # 查找可能包含主要内容的元素
        main_text_elements = []
        
        # 查找段落
        paragraphs = container.find_all('p')
        for p in paragraphs:
            text = p.get_text(strip=True)
            if len(text) > 20:
                main_text_elements.append(text)
        
        # 查找包含较多文本的div
        divs = container.find_all('div')
        for div in divs:
            # 只获取直接文本，不包括子元素
            direct_text = ''.join(div.find_all(string=True, recursive=False)).strip()
            if len(direct_text) > 20:
                main_text_elements.append(direct_text)
        
        # 如果没有找到段落，获取整个容器的文本
        if not main_text_elements:
            full_text = container.get_text(separator=' ', strip=True)
            if len(full_text) > 20:
                return full_text
        
        # 返回最长的文本片段
        if main_text_elements:
            return max(main_text_elements, key=len)
        
        return ""
    
    def _extract_post_info(self, container, main_text: str) -> Dict[str, Any]:
        """从容器中提取帖子信息"""
        # 提取用户名
        username = self._extract_username_from_container(container)
        
        # 提取时间
        timestamp = self._extract_timestamp_from_container(container)
        
        # 提取互动数据
        interactions = self._extract_interactions(container)
        
        # 提取图片
        images = [img.get('src') for img in container.find_all('img', src=True) 
                 if img.get('src') and not img.get('src').startswith('data:')]
        
        return {
            'content': main_text,
            'username': username,
            'timestamp': timestamp,
            'interactions': interactions,
            'images': images,
            'content_length': len(main_text),
            'word_count': len(main_text.split()),
            'extracted_at': time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _extract_username_from_container(self, container) -> str:
        """从容器中提取用户名"""
        # 查找可能包含用户名的元素
        username_selectors = [
            '[class*="user"]',
            '[class*="name"]',
            '[class*="author"]',
            'a[href*="/user/"]',
            'span[class*="nick"]'
        ]
        
        for selector in username_selectors:
            elements = container.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                if text and len(text) < 50 and not any(char in text for char in ['/', '\\', '?', '&']):
                    return text
        
        return ""
    
    def _extract_timestamp_from_container(self, container) -> str:
        """从容器中提取时间戳"""
        text = container.get_text()
        
        # 时间模式
        time_patterns = [
            r'\d{4}-\d{1,2}-\d{1,2}',
            r'\d{1,2}-\d{1,2}',
            r'\d+分钟前',
            r'\d+小时前',
            r'\d+天前',
            r'\d{4}年\d{1,2}月\d{1,2}日',
            r'\d{1,2}月\d{1,2}日',
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return ""
    
    def _extract_interactions(self, container) -> Dict[str, int]:
        """提取互动数据（点赞、评论、分享等）"""
        text = container.get_text()
        interactions = {}
        
        # 查找数字+关键词的模式
        patterns = {
            'likes': r'(\d+)点赞',
            'comments': r'(\d+)评论',
            'shares': r'(\d+)分享',
            'views': r'(\d+)浏览'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                interactions[key] = int(match.group(1))
        
        return interactions
    
    def _is_duplicate(self, post_data: Dict, existing_posts: List[Dict]) -> bool:
        """检查是否重复"""
        content = post_data['content']
        for existing in existing_posts:
            if content == existing['content']:
                return True
            # 检查内容相似度（前100字符）
            if len(content) > 100 and len(existing['content']) > 100:
                if content[:100] == existing['content'][:100]:
                    return True
        return False
    
    def _deduplicate_posts(self, posts: List[Dict]) -> List[Dict]:
        """去重并按质量排序"""
        unique_posts = []
        seen_contents = set()
        
        # 按内容长度排序，优先保留较长的内容
        posts.sort(key=lambda x: x['content_length'], reverse=True)
        
        for post in posts:
            content_hash = hash(post['content'][:100])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_posts.append(post)
        
        return unique_posts
    
    def _extract_from_json_data(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """从页面的JSON数据中提取帖子内容"""
        posts = []
        
        # 查找页面中的script标签，寻找包含帖子数据的JSON
        script_tags = soup.find_all('script')
        
        for script in script_tags:
            if script.string:
                script_content = script.string.strip()
                
                # 寻找包含cheyou_list或类似结构的JSON数据
                if 'cheyou_list' in script_content or 'content' in script_content:
                    try:
                        # 尝试提取JSON数据
                        json_matches = re.findall(r'\{.*?"cheyou_list".*?\}', script_content, re.DOTALL)
                        
                        for json_str in json_matches:
                            try:
                                # 尝试解析JSON
                                data = json.loads(json_str)
                                posts.extend(self._parse_cheyou_list(data))
                            except json.JSONDecodeError:
                                continue
                                
                        # 如果没有找到完整的JSON，尝试查找部分数据
                        if not posts:
                            posts.extend(self._extract_partial_json_data(script_content))
                            
                    except Exception as e:
                        print(f"解析JSON数据时出错: {e}")
                        continue
        
        return posts
    
    def _parse_cheyou_list(self, data: Dict) -> List[Dict[str, Any]]:
        """解析cheyou_list数据结构"""
        posts = []
        
        try:
            # 查找cheyou_list
            cheyou_list = None
            if 'cheyou_list' in data:
                cheyou_list = data['cheyou_list']
            elif 'props' in data and 'pageProps' in data['props'] and 'cheyouList' in data['props']['pageProps']:
                cheyou_list = data['props']['pageProps']['cheyouList'].get('cheyou_list', [])
            
            if cheyou_list:
                for item in cheyou_list:
                    if isinstance(item, dict) and 'content' in item:
                        post_data = self._parse_single_post(item)
                        if post_data:
                            posts.append(post_data)
                            
        except Exception as e:
            print(f"解析cheyou_list时出错: {e}")
        
        return posts
    
    def _parse_single_post(self, item: Dict) -> Dict[str, Any]:
        """解析单个帖子数据"""
        try:
            content = item.get('content', '').strip()
            if not content or len(content) < 20:
                return None
            
            # 提取用户信息
            username = ""
            if 'profile_info' in item and 'name' in item['profile_info']:
                username = item['profile_info']['name']
            
            # 提取时间戳 - 这是关键修复
            timestamp = ""
            timestamp_int = None
            
            # 优先使用display_time
            if 'display_time' in item and item['display_time']:
                timestamp_int = item['display_time']
            # 备选create_time
            elif 'create_time' in item and item['create_time']:
                timestamp_int = item['create_time']
            # 备选latest_comment_time
            elif 'latest_comment_time' in item and item['latest_comment_time']:
                timestamp_int = item['latest_comment_time']
            
            # 转换时间戳
            if timestamp_int:
                timestamp = self._convert_timestamp_to_date(timestamp_int)
            
            # 提取互动数据
            interactions = {}
            if 'digg_count' in item:
                interactions['likes'] = item['digg_count']
            if 'comment_count' in item:
                interactions['comments'] = item['comment_count']
            if 'read_count' in item and item['read_count']:
                interactions['views'] = item['read_count']
            
            # 提取图片
            images = []
            if 'image_urls' in item and isinstance(item['image_urls'], list):
                for img in item['image_urls']:
                    if isinstance(img, dict) and 'url' in img:
                        images.append(img['url'])
                    elif isinstance(img, str):
                        images.append(img)
            
            return {
                'content': content,
                'username': username,
                'timestamp': timestamp,
                'interactions': interactions,
                'images': images,
                'content_length': len(content),
                'word_count': len(content.split()),
                'extracted_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'source': 'json_data'  # 标记数据来源
            }
            
        except Exception as e:
            print(f"解析单个帖子时出错: {e}")
            return None
    
    def _extract_partial_json_data(self, script_content: str) -> List[Dict[str, Any]]:
        """从部分JSON数据中提取内容"""
        posts = []
        
        try:
            # 更复杂的正则表达式来提取更多信息
            # 根据实际数据结构调整：content -> digg_count -> comment_count -> profile_info -> name
            post_pattern = r'"content":\s*"([^"]+)"[^}]*?"digg_count":\s*(\d+)[^}]*?"comment_count":\s*(\d+)[^}]*?"profile_info":\s*{[^}]*?"name":\s*"([^"]*)"[^}]*?}.*?"display_time":\s*(\d+)'
            matches = re.findall(post_pattern, script_content, re.DOTALL)
            
            for content, digg_count, comment_count, username, timestamp_str in matches:
                if len(content) > 20:
                    timestamp = self._convert_timestamp_to_date(int(timestamp_str))
                    interactions = {}
                    if digg_count and digg_count != '0':
                        interactions['likes'] = int(digg_count)
                    if comment_count and comment_count != '0':
                        interactions['comments'] = int(comment_count)
                    
                    posts.append({
                        'content': content,
                        'username': username.strip() if username else "",
                        'timestamp': timestamp,
                        'interactions': interactions,
                        'images': [],
                        'content_length': len(content),
                        'word_count': len(content.split()),
                        'extracted_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'source': 'partial_json'
                    })
            
            # 如果上面的复杂模式没有匹配到，回退到简单模式
            if not posts:
                simple_pattern = r'"content":\s*"([^"]+)".*?"display_time":\s*(\d+)'
                simple_matches = re.findall(simple_pattern, script_content)
                
                for content, timestamp_str in simple_matches:
                    if len(content) > 20:
                        timestamp = self._convert_timestamp_to_date(int(timestamp_str))
                        posts.append({
                            'content': content,
                            'username': "",
                            'timestamp': timestamp,
                            'interactions': {},
                            'images': [],
                            'content_length': len(content),
                            'word_count': len(content.split()),
                            'extracted_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'source': 'partial_json'
                        })
                    
        except Exception as e:
            print(f"提取部分JSON数据时出错: {e}")
        
        return posts
    
    def format_posts(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        格式化帖子，只保留符合Terminal#997-1007结构的高质量内容
        
        筛选条件：
        1. 必须有用户名
        2. 必须有时间戳
        3. 必须有实际的用户发布内容（不是UI元素）
        4. 内容长度要合理（50-2000字符）
        5. 优先保留有互动数据的内容
        """
        print("正在格式化和过滤内容...")
        
        formatted_posts = []
        
        for post in posts:
            # 基本筛选条件
            if not self._is_high_quality_post(post):
                continue
            
            # 清理和格式化内容
            formatted_post = self._clean_and_format_post(post)
            
            if formatted_post:
                formatted_posts.append(formatted_post)
        
        # 按质量排序：优先显示有互动数据、用户信息完整的内容
        formatted_posts.sort(key=self._calculate_post_quality_score, reverse=True)
        
        print(f"格式化完成，保留 {len(formatted_posts)} 条高质量内容")
        return formatted_posts
    
    def _is_high_quality_post(self, post: Dict[str, Any]) -> bool:
        """判断是否是高质量帖子"""
        # 对于从JSON数据提取的内容，放宽用户名要求
        source = post.get('source', '')
        if source in ['json_data', 'partial_json']:
            # JSON数据来源的内容，只要有时间戳就认为是可信的
            if not post.get('timestamp'):
                return False
        else:
            # HTML解析的内容，必须有用户名
            if not post.get('username') or len(post['username']) < 2:
                return False
            
            # 必须有时间戳
            if not post.get('timestamp'):
                return False
        
        # 内容长度要合理
        content = post.get('content', '')
        if len(content) < 50 or len(content) > 2000:
            return False
        
        # 排除明显的UI元素或系统信息
        ui_indicators = [
            '车友圈', '圈主', '懂车帝', '加入圈子', '查看详情',
            '共', '万', '车友等你加入', '认证车主', '问答', '口碑',
            '优秀', '评价', '指导价', '优惠', '省补', '区补', '落地'
        ]
        
        for indicator in ui_indicators:
            if indicator in content and len(content) < 200:
                return False
        
        # 检查是否包含真实的用户体验内容
        experience_keywords = [
            '我', '试驾', '体验', '感觉', '觉得', '认为', '使用', '开了',
            '买了', '订了', '提车', '驾驶', '操控', '续航', '充电',
            '空间', '内饰', '外观', '配置', '价格', '服务', '销售',
            '满意', '不满意', '推荐', '不推荐', '问题', '建议'
        ]
        
        has_experience = any(keyword in content for keyword in experience_keywords)
        
        return has_experience
    
    def _clean_and_format_post(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """清理和格式化帖子内容"""
        # 清理用户名
        username = post.get('username', '').strip()
        # 移除可能的多余信息
        username = re.sub(r'\s+', ' ', username)
        
        # 清理内容
        content = post.get('content', '').strip()
        # 移除多余的空白字符
        content = re.sub(r'\s+', ' ', content)
        # 移除可能的HTML残留
        content = re.sub(r'<[^>]+>', '', content)
        
        # 如果内容太长，进行智能截断
        if len(content) > 1000:
            # 尝试在句号处截断
            sentences = content.split('。')
            truncated_content = ''
            for sentence in sentences:
                if len(truncated_content + sentence + '。') <= 1000:
                    truncated_content += sentence + '。'
                else:
                    break
            if truncated_content:
                content = truncated_content
            else:
                content = content[:1000] + '...'
        
        return {
            'content': content,
            'username': username,
            'timestamp': post.get('timestamp', ''),
            'interactions': post.get('interactions', {}),
            'images': post.get('images', []),
            'content_length': len(content),
            'word_count': len(content.split()),
            'extracted_at': post.get('extracted_at', time.strftime("%Y-%m-%d %H:%M:%S")),
            'quality_score': self._calculate_post_quality_score(post)
        }
    
    def _calculate_post_quality_score(self, post: Dict[str, Any]) -> float:
        """计算帖子质量分数"""
        score = 0.0
        
        # 基础分数
        score += 1.0
        
        # 有用户名加分
        if post.get('username'):
            score += 2.0
            # 用户名包含车主信息加分
            if '车主' in post['username']:
                score += 1.0
            # 用户名包含车龄信息加分
            if '车龄' in post['username']:
                score += 0.5
        
        # 有时间戳加分
        if post.get('timestamp'):
            score += 1.0
        
        # 有互动数据加分
        interactions = post.get('interactions', {})
        if interactions:
            score += 2.0
            # 互动数量加分
            total_interactions = sum(interactions.values())
            if total_interactions > 10:
                score += 1.0
            if total_interactions > 100:
                score += 1.0
        
        # 内容长度适中加分
        content_length = post.get('content_length', 0)
        if 100 <= content_length <= 800:
            score += 1.0
        elif content_length > 800:
            score += 0.5
        
        # 有图片加分
        if post.get('images'):
            score += 0.5
        
        return score

    def analyze_and_extract(self, url: str) -> Dict[str, Any]:
        """分析并提取用户内容"""
        try:
            soup = self.fetch_content(url)
            raw_posts = self.extract_post_content(soup)
            
            # 新增：格式化步骤
            formatted_posts = self.format_posts(raw_posts)
            
            result = {
                "url": url,
                "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "success": True,
                "raw_posts": raw_posts,
                "posts": formatted_posts,  # 使用格式化后的内容
                "total_raw_posts": len(raw_posts),
                "total_posts": len(formatted_posts),
                "summary": {
                    "total_content_length": sum(post['content_length'] for post in formatted_posts),
                    "average_content_length": sum(post['content_length'] for post in formatted_posts) / len(formatted_posts) if formatted_posts else 0,
                    "posts_with_username": len([post for post in formatted_posts if post['username']]),
                    "posts_with_timestamp": len([post for post in formatted_posts if post['timestamp']]),
                    "posts_with_images": len([post for post in formatted_posts if post['images']]),
                    "posts_with_interactions": len([post for post in formatted_posts if post['interactions']]),
                    "average_quality_score": sum(post.get('quality_score', 0) for post in formatted_posts) / len(formatted_posts) if formatted_posts else 0,
                }
            }
            
            return result
            
        except Exception as e:
            return {
                "url": url,
                "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "success": False,
                "error": str(e)
            }
    
    def save_results(self, result: Dict[str, Any], filename: str = None):
        """保存结果"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_content_{timestamp}"
        
        # 保存JSON
        json_filename = f"{filename}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 保存纯文本
        txt_filename = f"{filename}.txt"
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(f"用户内容提取结果（增强版）\n")
            f.write(f"URL: {result['url']}\n")
            f.write(f"提取时间: {result['analysis_time']}\n")
            f.write(f"总计: {result.get('total_posts', 0)} 条内容\n")
            f.write("=" * 60 + "\n\n")
            
            if result.get('success') and result.get('posts'):
                for i, post in enumerate(result['posts'], 1):
                    f.write(f"【内容 {i}】\n")
                    if post['username']:
                        f.write(f"用户: {post['username']}\n")
                    if post['timestamp']:
                        f.write(f"时间: {post['timestamp']}\n")
                    if post['interactions']:
                        interactions_str = ', '.join([f"{k}: {v}" for k, v in post['interactions'].items()])
                        f.write(f"互动: {interactions_str}\n")
                    f.write(f"内容: {post['content']}\n")
                    if post['images']:
                        f.write(f"图片: {len(post['images'])} 张\n")
                    f.write("-" * 50 + "\n\n")
        
        print(f"结果已保存到: {json_filename} 和 {txt_filename}")


def main():
    """主函数"""
    url = "https://www.dongchedi.com/community/9118/hot-release-49"
    
    extractor = EnhancedContentExtractor()
    
    print("开始提取用户发布的内容（增强版）...")
    print("=" * 60)
    
    result = extractor.analyze_and_extract(url)
    
    if result["success"]:
        print("提取完成！")
        print(f"\n=== 提取和格式化统计 ===")
        print(f"原始内容数: {result['total_raw_posts']}")
        print(f"格式化后高质量内容数: {result['total_posts']}")
        print(f"过滤率: {((result['total_raw_posts'] - result['total_posts']) / result['total_raw_posts'] * 100):.1f}%" if result['total_raw_posts'] > 0 else "0%")
        
        if result['total_posts'] > 0:
            summary = result['summary']
            print(f"总字符数: {summary['total_content_length']}")
            print(f"平均长度: {summary['average_content_length']:.1f} 字符")
            print(f"平均质量分数: {summary['average_quality_score']:.2f}")
            print(f"包含用户名: {summary['posts_with_username']} 条")
            print(f"包含时间戳: {summary['posts_with_timestamp']} 条")
            print(f"包含图片: {summary['posts_with_images']} 条")
            print(f"包含互动数据: {summary['posts_with_interactions']} 条")
            
            print(f"\n=== 高质量内容预览（Terminal#997-1007格式）===")
            for i, post in enumerate(result['posts'][:5], 1):  # 显示前5条高质量内容
                print(f"\n【内容 {i}】")
                if post['username']:
                    print(f"用户: {post['username']}")
                if post['timestamp']:
                    print(f"时间: {post['timestamp']}")
                if post['interactions']:
                    interactions_str = ', '.join([f"{k}: {v}" for k, v in post['interactions'].items()])
                    print(f"互动: {interactions_str}")
                content_preview = post['content'][:300] + '...' if len(post['content']) > 300 else post['content']
                print(f"内容: {content_preview}")
                print(f"质量分数: {post.get('quality_score', 0):.2f}")
        
        extractor.save_results(result)
        
    else:
        print(f"提取失败: {result['error']}")


if __name__ == "__main__":
    main()