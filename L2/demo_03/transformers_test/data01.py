from datasets import load_dataset

ds = load_dataset("congyanyin0623/chinese_poems.txt", cache_dir="D://transforms/datasets")
print(ds["train"])