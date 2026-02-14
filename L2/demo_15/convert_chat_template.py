import re
# 原始对话模板配置
original_qwen_chat = dict(
SYSTEM=("<|im_start|>system\n{system}<|im_end|>\n"),
INSTRUCTION=("<|im_start|>user\n{input}<|im_end|>\n" 
             "<|im_start|>assistant\n"),
SUFFIX="<|im_end|>",
SUFFIX_AS_EOS=True,
SEP="\n",
STOP_WORDS=["<|im_end|>", "<|endoftext|>"],
)
# 转换函数
def convert_template(template):
    converted = {}
    for key, value in template.items():
        if isinstance(value, str):
            # 将 {variable} 格式转换为 {{ variable }}
            converted_value = re.sub(r'\{(\w+)\}', r'{{ \1 }}', value)
            converted[key] = converted_value
        else:
            converted[key] = value
    return converted
# 执行转换
jinja2_qwen_chat = convert_template(original_qwen_chat)
print(jinja2_qwen_chat)