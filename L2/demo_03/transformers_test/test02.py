from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_dir = r"D://transforms/models/model/uer/gpt2-chinese-cluecorpussmall/models--uer--gpt2-chinese-cluecorpussmall/snapshots/c2c0249d8a2731f269414cc3b22dff021f8e07a3"

model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# output = generator("你好，我是一款语言模型", max_length=50, num_return_sequences=1)

output = generator(
    "你好，我是一款语言模型,",
    max_length=50,
    num_return_sequences=1,
    truncation=True,
    temperature=0.75,
    # do_sample=False,
    top_k=50,
    top_p=0.95,
    clean_up_tokenization_spaces=True

)

print(output)
# print(model)