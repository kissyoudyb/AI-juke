# mytest.py
import sys
import os

# 将项目根目录添加到 Python 路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

from llamafactory.data.template import TEMPLATES
from transformers import AutoTokenizer

# 1. 初始化分词器（任意支持的分词器均可）
tokenizer = AutoTokenizer.from_pretrained("/root/llms/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# 2. 获取模板对象
template_name = "qwen"  # 替换为你需要查看的模板名称
template = TEMPLATES[template_name]

# 3. 修复分词器的 Jinja 模板
template.fix_jinja_template(tokenizer)

# 4. 直接输出模板的 Jinja 格式
print("=" * 40)
print(f"Template [{template_name}] 的 Jinja 格式:")
print("=" * 40)
print(tokenizer.chat_template)