from datasets import load_dataset,load_from_disk

#在线加载数据
# dataset = load_dataset(path="NousResearch/hermes-function-calling-v1",cache_dir="data/")
# print(dataset)
#转为csv格式
# dataset.to_csv(path_or_buf=r"D:\PycharmProjects\demo_02\data\ChnSentiCorp.csv")

# 加载缓存数据
datasets = load_from_disk(r"D:\github-code\jukeai\L2\demo_05\data\ChnSentiCorp")
print(datasets)

train_data = datasets["test"]
for data in train_data:
    print(data)

# 扩展：加载CSV格式数据
# dataset = load_dataset(path="csv",data_files=r"D:\jukeai\demo_04\data\hermes-function-calling-v1.csv")
# print(dataset)