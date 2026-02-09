# pip install transformers peft
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# 加载原始模型
model = AutoModelForCausalLM.from_pretrained("/root/llms/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("/root/llms/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
# 注入LoRA权重
model = PeftModel.from_pretrained(model, "/root/LlamaFactory/saves/DeepSeek-R1-1.5B-Distill/lora/train_2026-02-09-09-17-04")
# ruozhiba-qa测试
input_text = "马上要上游泳课了，昨天洗的泳裤还没干，怎么办"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_length=128)[0]))