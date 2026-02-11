from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1/",api_key="suibianxie")

chat_completion = client.chat.completions.create(
    messages=[{"role":"user","content":"你好，请介绍下你自己。"}],model="/root/llms/Qwen/Qwen2.5-1.5B-Instruct"
)
print(chat_completion.choices[0])