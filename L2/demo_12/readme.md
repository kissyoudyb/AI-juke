# 实验步骤 不能直接git clone 最新代码，根本跑不了

    conda create -n xtuner python=3.10
    conda activate xtuner
    wget https://github.com/InternLM/xtuner/archive/refs/tags/v0.2.0rc0.zip
    如果没有安装unzip 请安装
    unzip v0.2.0rc0.zip
    cd xtuner-0.2.0rc0
    pip install -e .
    替换runtime.txt避免报错
    pip uninstall datasets
    pip install -e .
    修改配置文件 batch_size = 10 save_steps = 100 evaluation_freq = 100
    /root/xtuner-0.2.0rc0/xtuner/qwen1_5_1_8b_chat_qlora_alpaca_e3.py
    开始微调 搞到1000步就可以停了
    xtuner train /root/xtuner-0.2.0rc0/xtuner/qwen1_5_1_8b_chat_qlora_alpaca_e3.py
    模型转换
    xtuner convert pth_to_hf /root/xtuner-0.2.0rc0/work_dirs/qwen1_5_1_8b_chat_qlora_alpaca_e3/qwen1_5_1_8b_chat_qlora_alpaca_e3.py /root/xtuner-0.2.0rc0/work_dirs/qwen1_5_1_8b_chat_qlora_alpaca_e3/iter_600.pth /root/llms/Qwen/qwen1_5_1_8b_chat_qlora_alpaca_e3
    模型合并
    xtuner convert merge /root/llms/Qwen/Qwen1.5-1.8B-Chat /root/llms/Qwen/qwen1_5_1_8b_chat_qlora_alpaca_e3 /root/llms/Qwen/qwen1_5_1_8b_chat_qlora_merged
    使用vllm测试合并后的模型进行推理
    conda activate vllm
    vllm serve  /root/llms/Qwen/qwen1_5_1_8b_chat_qlora_merged
    python testvllm.py
    
