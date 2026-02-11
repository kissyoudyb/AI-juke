# 实验步骤
git clone https://github.com/ggerganov/llama.cpp.git
pip install -r llama.cpp/requirements.txt

## 如果不量化，保留模型的效果
python /root/llama.cpp/convert_hf_to_gguf.py /root/llms/Qwen/Qwen2.5-1.5B-Instruct --outtype f16 --verbose --outfile Qwen2.5-1.5B-Instruct-gguf.gguf

## 如果需要量化（加速并有损效果），直接执行下面脚本就可以
python /root/llama.cpp/convert_hf_to_gguf.py /root/llms/Qwen/Qwen2.5-1.5B-Instruct --outtype q8_0 --verbose --outfile Qwen2.5-1.5B-Instruct-gguf_q8_0.gguf

ollama serve
## 创建模型
ollama create Qwen2.5-1.5B-Instruct-gguf --file ./ModelFile

## 运行模型
ollama run Qwen2.5-1.5B-Instruct-gguf