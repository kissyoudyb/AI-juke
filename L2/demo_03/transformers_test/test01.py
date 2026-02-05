from transformers import AutoModelForCausalLM,AutoTokenizer

# model_name = "uer/gpt2-chinese-cluecorpussmall"
# model_name = "bert-base-chinese"
# model_name = "Qwen/Qwen3-0.6B"
# model_name = "Qwen/Qwen3-1.7B"
model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# cache_dir = "D://transforms/models/model/uer/gpt2-chinese-cluecorpussmall"
# cache_dir = "D://transforms/models/model/Qwen/Qwen3-1.7B"
cache_dir = "D://transforms/models/model/uer/gpt2-chinese-cluecorpussmall"

AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

print(f"模型分词器已下载到: {cache_dir}")


