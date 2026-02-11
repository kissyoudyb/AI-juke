# 操作说明
## 1-ollama安装和启动-参考第7章
    ollama serve
    ollama run qwen2.5:0.5b

## 2-使用open webui部署模型
    conda create -n open-webui python=3.11

    conda activate open-webui
    pip install -U open-webui torch transformers
    
    # 运行open-webui
    export HF_ENDPOINT=https://hf-mirror.com
    export ENABLE_OLLAMA_API=True
    export OPENAI_API_BASE_URL=http://127.0.0.1:11434/v1
    # 启动服务，这步需要联网下载相当多的包
    open-webui serve