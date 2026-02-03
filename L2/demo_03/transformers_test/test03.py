from transformers import BertTokenizer, BertForSequenceClassification, pipeline

model_dir = r"D:\transforms\models\model\bert-base-chinese\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

result = classifier("你好，我是一款语言模型")

# output = generator(
#     "你好，我是一款语言模型,",
#     max_length=50,
#     num_return_sequences=1,
#     truncation=True,
#     temperature=0.75,
#     # do_sample=False,
#     top_k=50,
#     top_p=0.95,
#     clean_up_tokenization_spaces=True
#
# )

# print(result)
print(model)