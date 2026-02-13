# 实验步骤
    1. 修改distill.py 相关模型路径
    2. python distill.py
    3. 测试 使用vllm
    vllm serve /root/llms/Qwen/distilled_qwen --tokenizer /root/llms/Qwen/Qwen1.5-0.5B-Chat
    python testvllm.py