import json
from datasets import load_dataset


def convert_arrow_to_llama_factory(arrow_file, output_json):
    # 1. 加载 arrow 文件
    # data_files 参数可以接受单个文件或列表
    dataset = load_dataset("arrow", data_files={"train": arrow_file})

    # 2. 提取训练集数据
    raw_data = dataset["train"]

    alpaca_data = []

    # 3. 遍历转换
    for item in raw_data:
        # 假设 arrow 中的字段名依然是 query, response, system
        # 如果字段名不同，请根据实际情况修改 key 值
        alpaca_item = {
            "instruction": item.get("query", ""),
            "input": "",
            "output": item.get("response", ""),
            "system": item.get("system", "")
        }
        alpaca_data.append(alpaca_item)

    # 4. 保存为 JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

    print(f"转换完成！已保存至: {output_json}")
    print(f"总计条数: {len(alpaca_data)}")


if __name__ == "__main__":
    # 指定你的 arrow 文件名和输出文件名
    convert_arrow_to_llama_factory(
        "/root/datasets/w10442005___ruozhiba_qa/default-de7913bb979851d5/0.0.0/master/ruozhiba_qa-train.arrow",
        "/root/datasets/w10442005___ruozhiba_qa/default-de7913bb979851d5/0.0.0/master/ruozhiba_alpaca.json")
